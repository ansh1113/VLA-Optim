import threading
import time
from typing import Callable, Optional

import numpy as np
import yaml

from vla_optim.obstacles import cluster_to_spheres, crop_to_workspace, filter_self_obstacles


def quat_to_rotmat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class LiveObstacleFeed:
    """Background thread: repeatedly pulls a raw point cloud from whatever
    depth sensor you have, crops it to the workspace, transforms it into
    the robot base frame, and clusters it into obstacle spheres.

    This is intentionally sensor-agnostic. You provide `point_cloud_fn`, a
    zero-argument callable that returns the latest (N, 3) array of points in
    the *camera's own frame* (or None if no new frame is ready yet) -- this
    could wrap any depth camera SDK, an existing perception node's output,
    or a simulator. Everything downstream of that callback (crop, transform,
    cluster, self-filter) is identical regardless of source.

    Call `latest(q, safety_filter)` from your control loop to get the most
    recent obstacle list with the arm's own current pose filtered out. Runs
    independently of both the policy and the safety filter loop -- `latest()`
    is non-blocking and just returns whatever was computed on the last grab.
    """

    def __init__(self, extrinsics_config_path: str,
                 point_cloud_fn: Callable[[], Optional[np.ndarray]],
                 poll_hz: float = 30.0):
        with open(extrinsics_config_path) as f:
            cfg = yaml.safe_load(f)
        R = quat_to_rotmat(cfg["rotation_quaternion"])
        t = np.array(cfg["translation"])
        self._T = np.eye(4)
        self._T[:3, :3] = R
        self._T[:3, 3] = t
        self._workspace_bounds = cfg["workspace_bounds"]

        self._point_cloud_fn = point_cloud_fn
        self._period = 1.0 / poll_hz

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

    def _loop(self):
        while self._running:
            t0 = time.time()
            xyz = self._point_cloud_fn()

            if xyz is None or len(xyz) == 0:
                time.sleep(max(0.0, self._period - (time.time() - t0)))
                continue

            xyz = xyz[np.isfinite(xyz).all(axis=1)]
            ones = np.ones((xyz.shape[0], 1))
            xyz_h = np.hstack([xyz, ones])
            xyz_base = (self._T @ xyz_h.T).T[:, :3]

            xyz_base = crop_to_workspace(xyz_base, self._workspace_bounds)
            obstacles = cluster_to_spheres(xyz_base)

            with self._lock:
                self._latest_obstacles = obstacles

            time.sleep(max(0.0, self._period - (time.time() - t0)))

    def latest(self, q: np.ndarray, safety_filter) -> list:
        with self._lock:
            obstacles = list(self._latest_obstacles)
        link_spheres = safety_filter.link_spheres(q)
        return filter_self_obstacles(obstacles, link_spheres)
