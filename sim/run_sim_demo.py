"""Visual side-by-side test of RealtimeSafetyFilter: runs the same naive
"policy" (a fixed joint velocity command, standing in for an unaware VLA
action) twice against a MuJoCo-rendered arm -- once applied directly, once
passed through the safety filter -- and saves each as a GIF so the
difference is visible rather than asserted.

    python sim/run_sim_demo.py
"""
import os

import imageio
import mujoco
import numpy as np

from vla_optim.obstacles import Obstacle
from vla_optim.safety_filter import RealtimeSafetyFilter

HERE = os.path.dirname(__file__)
URDF = os.path.join(HERE, "demo_robot.urdf")
LINKS_CONFIG = os.path.join(HERE, "demo_links.yaml")
OUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

DT = 0.02
N_STEPS = 140
QDOT_PI = np.array([0.8, 0.0, 0.0])  # constant "policy" command: swing joint1

model = mujoco.MjModel.from_xml_path(URDF)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=360, width=480)
cam = mujoco.MjvCamera()
cam.lookat = [0.35, 0.25, 0.0]
cam.distance = 1.1
cam.azimuth = -90
cam.elevation = -89.9  # near-top-down: this arm's motion is planar at z=0,
                        # so a straight overhead view removes any camera-
                        # perspective ambiguity about clearance distance


def render_frame(q, obstacle_center, obstacle_radius, obstacle_rgba):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    scene = renderer.scene
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[obstacle_radius, 0, 0],
        pos=obstacle_center,
        mat=np.eye(3).flatten(),
        rgba=obstacle_rgba,
    )
    scene.ngeom += 1
    return renderer.render().copy()


def rollout(filt, obstacle, use_filter):
    q = np.zeros(model.nq)
    frames = []
    for step in range(N_STEPS):
        if use_filter:
            qdot = filt.filter(q, QDOT_PI, [obstacle] if obstacle else [])
        else:
            qdot = QDOT_PI
        q = q + qdot * DT
        rgba = [0.85, 0.15, 0.15, 1] if use_filter else [0.85, 0.15, 0.15, 0.9]
        frames.append(render_frame(q, obstacle.center, obstacle.radius, rgba))
    return frames


# --- Step 1: find where the unfiltered arm actually goes, so the obstacle
# is guaranteed to be in its path rather than guessed at. ---
q = np.zeros(model.nq)
filt_probe = RealtimeSafetyFilter(URDF, LINKS_CONFIG, dt=DT)
midpoint_link3 = None
for step in range(N_STEPS):
    q = q + QDOT_PI * DT
    if step == N_STEPS // 2:
        link_spheres = filt_probe.link_spheres(q)
        midpoint_link3 = link_spheres[-1][0].copy()

obstacle = Obstacle(center=midpoint_link3, radius=0.09)
print(f"obstacle placed at {obstacle.center}, directly in the unfiltered arm's path")

# --- Step 2: two real runs from the same start, same obstacle. ---
print("rendering unfiltered rollout...")
frames_unfiltered = rollout(filt=None, obstacle=obstacle, use_filter=False)

print("rendering filtered rollout...")
filt = RealtimeSafetyFilter(URDF, LINKS_CONFIG, dt=DT)
frames_filtered = rollout(filt=filt, obstacle=obstacle, use_filter=True)

no_filter_path = os.path.join(OUT_DIR, "no_filter.gif")
with_filter_path = os.path.join(OUT_DIR, "with_filter.gif")
imageio.mimsave(no_filter_path, frames_unfiltered, duration=DT, loop=0)
imageio.mimsave(with_filter_path, frames_filtered, duration=DT, loop=0)
print(f"saved {no_filter_path}")
print(f"saved {with_filter_path}")
