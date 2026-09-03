"""Sanity-check the safety filter's math with a synthetic obstacle -- no
robot hardware or camera required.

    python examples/run_safety_filter_demo.py --urdf path/to/robot.urdf
"""
import argparse

import numpy as np

from vla_optim.obstacles import Obstacle
from vla_optim.safety_filter import RealtimeSafetyFilter

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", required=True)
parser.add_argument("--links-config", default="config/robot_links.yaml")
parser.add_argument("--dt", type=float, default=0.02)
args = parser.parse_args()

filt = RealtimeSafetyFilter(args.urdf, args.links_config, dt=args.dt)

q = np.zeros(filt.nq)
qdot_pi = np.zeros(filt.nq)
qdot_pi[0] = 0.8  # drive joint 0 so the end link sweeps toward the obstacle

link_spheres = filt.link_spheres(q)
target_center, target_radius = link_spheres[-1]
obstacle = Obstacle(center=target_center + np.array([0.0, 0.25, 0.0]), radius=0.05)

print(f"{'step':>4}  {'clearance (m)':>14}  {'||qdot_pi||':>12}  {'||qdot_safe||':>13}")
for step in range(200):
    obstacles = [obstacle]
    qdot_safe = filt.filter(q, qdot_pi, obstacles)

    link_spheres = filt.link_spheres(q)
    center, radius = link_spheres[-1]
    clearance = np.linalg.norm(center - obstacle.center) - radius - obstacle.radius

    if step % 10 == 0:
        print(f"{step:>4}  {clearance:>14.4f}  {np.linalg.norm(qdot_pi):>12.4f}  "
              f"{np.linalg.norm(qdot_safe):>13.4f}")

    q = q + qdot_safe * args.dt

print("\nExpected: ||qdot_safe|| tracks ||qdot_pi|| until clearance approaches")
print("d_safe, then drops toward zero as the filter brakes the approach.")
