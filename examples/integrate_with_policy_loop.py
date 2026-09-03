"""Skeleton for the two-thread architecture: the policy runs in the
background at its own (slow) rate, the safety filter runs in the foreground
at a much higher rate, always reading the latest policy action and latest
perceived obstacles.

Plug in your own `policy.predict(obs) -> qdot`,
`get_current_joint_positions()` / `send_to_controller(qdot)`, and
`get_point_cloud()` (wrapping whatever depth sensor/SDK you have -- anything
that can hand back an (N, 3) array of points in the camera frame) --
everything here is placeholder except the filter/feed wiring itself.
"""
import threading
import time

import numpy as np

from vla_optim.safety_filter import RealtimeSafetyFilter
from vla_optim.perception import LiveObstacleFeed

URDF_PATH = "path/to/robot.urdf"
LINKS_CONFIG = "config/robot_links.yaml"
CAMERA_CONFIG = "config/camera_extrinsics.yaml"

FILTER_RATE_HZ = 100
POLICY_RATE_HZ = 10

_latest_action_lock = threading.Lock()
_latest_action = None  # set by the policy thread, read by the filter loop


def get_current_joint_positions() -> np.ndarray:
    raise NotImplementedError("hook up to your robot's joint state")


def send_to_controller(qdot_safe: np.ndarray) -> None:
    raise NotImplementedError("hook up to your controller/driver")


def policy_predict(obs) -> np.ndarray:
    raise NotImplementedError("hook up to your PI / lerobot policy")


def get_observation():
    raise NotImplementedError("hook up to your policy's observation pipeline")


def get_point_cloud() -> np.ndarray:
    """Return the latest (N, 3) point cloud in the camera's own frame, or
    None if no new frame is ready. Wrap whatever depth sensor/SDK you have
    here -- this is the only sensor-specific code in the whole pipeline."""
    raise NotImplementedError("hook up to your depth sensor of choice")


def policy_loop():
    global _latest_action
    period = 1.0 / POLICY_RATE_HZ
    while True:
        t0 = time.time()
        obs = get_observation()
        action = policy_predict(obs)
        with _latest_action_lock:
            _latest_action = action
        time.sleep(max(0.0, period - (time.time() - t0)))


def safety_filter_loop():
    filt = RealtimeSafetyFilter(URDF_PATH, LINKS_CONFIG, dt=1.0 / FILTER_RATE_HZ)
    feed = LiveObstacleFeed(CAMERA_CONFIG, point_cloud_fn=get_point_cloud, poll_hz=30)
    feed.start()

    period = 1.0 / FILTER_RATE_HZ
    try:
        while True:
            t0 = time.time()

            with _latest_action_lock:
                qdot_pi = _latest_action

            if qdot_pi is not None:
                q = get_current_joint_positions()
                obstacles = feed.latest(q, filt)
                qdot_safe = filt.filter(q, qdot_pi, obstacles)
                send_to_controller(qdot_safe)

            time.sleep(max(0.0, period - (time.time() - t0)))
    finally:
        feed.stop()


if __name__ == "__main__":
    p1 = threading.Thread(target=policy_loop, daemon=True)
    p1.start()
    safety_filter_loop()
