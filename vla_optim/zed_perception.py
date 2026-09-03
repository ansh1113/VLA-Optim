import threading
import time

import numpy as np
import yaml

from vla_optim.obstacles import cluster_to_spheres, crop_to_workspace, filter_self_obstacles

try:
    import pyzed.sl as sl
    _HAS_PYZED = True
except ImportError:
    _HAS_PYZED = False


def quat_to_rotmat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class LiveObstacleFeed:
    """Background thread: grabs ZED point clouds, crops to the workspace,
    transforms into the robot base frame, and clusters into obstacle
    spheres. Call `latest(q, safety_filter)` from your control loop to get
    the most recent obstacle list with the arm itself filtered out.

    Runs independently of both the policy and the safety filter loop --
    `latest()` is non-blocking and just returns whatever was computed on the
    last camera grab.
    """

    def __init__(self, extrinsics_config_path: str, resolution="HD720", fps: int = 30):
        if not _HAS_PYZED:
            raise RuntimeError(
                "pyzed is not installed. Install the ZED SDK from Stereolabs "
                "first (it ships pyzed as part of the SDK install, not via pip)."
            )

        with open(extrinsics_config_path) as f:
            cfg = yaml.safe_load(f)
        R = quat_to_rotmat(cfg["rotation_quaternion"])
        t = np.array(cfg["translation"])
        self._T = np.eye(4)
        self._T[:3, :3] = R
        self._T[:3, 3] = t
        self._workspace_bounds = cfg["workspace_bounds"]

        self._zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = getattr(sl.RESOLUTION, resolution)
        init_params.camera_fps = fps
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        status = self._zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"failed to open ZED camera: {status}")

        self._point_cloud = sl.Mat()
        self._lock = threading.Lock()
        self._latest_obstacles = []
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._zed.close()

    def _loop(self):
        runtime_params = sl.RuntimeParameters()
        while self._running:
            if self._zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                time.sleep(0.005)
                continue

            self._zed.retrieve_measure(self._point_cloud, sl.MEASURE.XYZ)
            xyz = self._point_cloud.get_data()[:, :, :3].reshape(-1, 3)
            xyz = xyz[np.isfinite(xyz).all(axis=1)]

            if len(xyz) == 0:
                with self._lock:
                    self._latest_obstacles = []
                continue

            ones = np.ones((xyz.shape[0], 1))
            xyz_h = np.hstack([xyz, ones])
            xyz_base = (self._T @ xyz_h.T).T[:, :3]

            xyz_base = crop_to_workspace(xyz_base, self._workspace_bounds)
            obstacles = cluster_to_spheres(xyz_base)

            with self._lock:
                self._latest_obstacles = obstacles

    def latest(self, q: np.ndarray, safety_filter) -> list:
        with self._lock:
            obstacles = list(self._latest_obstacles)
        link_spheres = safety_filter.link_spheres(q)
        return filter_self_obstacles(obstacles, link_spheres)
