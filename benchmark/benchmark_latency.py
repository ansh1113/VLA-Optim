"""Per-stage latency benchmark for RealtimeSafetyFilter, so the "is this
faster than our 18ms MoveIt collision check" question is answered by a
measurement instead of an assumption.

    python benchmark/benchmark_latency.py --urdf path/to/robot.urdf \
        --n-obstacles 1 3 5 10 --n-iters 500
"""
import argparse
import time

import numpy as np

from vla_optim.obstacles import Obstacle
from vla_optim.safety_filter import RealtimeSafetyFilter

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", required=True)
parser.add_argument("--links-config", default="config/robot_links.yaml")
parser.add_argument("--dt", type=float, default=0.02)
parser.add_argument("--n-obstacles", type=int, nargs="+", default=[1, 3, 5, 10])
parser.add_argument("--n-iters", type=int, default=500)
parser.add_argument("--moveit-baseline-ms", type=float, default=18.0)
args = parser.parse_args()

filt = RealtimeSafetyFilter(args.urdf, args.links_config, dt=args.dt)
rng = np.random.default_rng(0)


def make_obstacles(n, near_link_center):
    return [
        Obstacle(center=near_link_center + rng.uniform(-0.3, 0.3, size=3), radius=0.05)
        for _ in range(n)
    ]


def percentile(samples_ms, p):
    return float(np.percentile(samples_ms, p))


print(f"{'n_obs':>6}  {'mean (ms)':>10}  {'median (ms)':>12}  {'p95 (ms)':>9}  {'max (ms)':>9}")

for n_obs in args.n_obstacles:
    q = np.zeros(filt.nq)
    qdot_current = np.zeros(filt.nq)  # held fixed every call -- see note below
    qdot_pi = rng.uniform(-0.3, 0.3, size=filt.nq)
    near = filt.link_spheres(q)[-1][0]
    obstacles = make_obstacles(n_obs, near)

    # Pass qdot_current explicitly and keep q fixed so every timed call is an
    # independent, representative instance of "policy asks for this action
    # at this state" rather than letting the filter's internal open-loop
    # velocity state drift across hundreds of calls at an unmoving q -- that
    # drift can wander into a genuinely hard-to-solve region and inflate the
    # measured latency in a way that has nothing to do with real usage.

    # warmup so ProxQP's internal buffers/resize path isn't counted
    for _ in range(10):
        filt.filter(q, qdot_pi, obstacles, qdot_current=qdot_current)

    samples_ms = []
    for _ in range(args.n_iters):
        t0 = time.perf_counter()
        filt.filter(q, qdot_pi, obstacles, qdot_current=qdot_current)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    samples_ms = np.array(samples_ms)
    print(f"{n_obs:>6}  {samples_ms.mean():>10.3f}  {np.median(samples_ms):>12.3f}  "
          f"{percentile(samples_ms, 95):>9.3f}  {samples_ms.max():>9.3f}")

print(f"\nMoveIt baseline (measured, per your team): {args.moveit_baseline_ms:.1f} ms")
print("Compare the mean/p95 columns above against that number directly --")
print("don't assume this filter is faster without this printout showing it.")
