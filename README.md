# VLA-Optim

Real-time safety filter for VLA policies on real arms. Sits between the policy and the
controller: takes the policy's action for this step, returns the closest action that
respects joint limits, velocity limits, and live obstacle clearance.

## Why

Full motion planners and full trajectory optimizers (see
[VLA-Optimization](https://github.com/ansh1113/VLA-Optimization), this repo's sibling)
solve a bigger problem than needed every control step — a whole path, or a whole
multi-waypoint trajectory. This repo only asks: *is the policy's next single action safe,
and if not, what's the smallest fix?* That's a QP with `n_joints` variables, not
`n_waypoints * n_joints`, so it's cheap enough to run every step instead of in the
background.

## How it works

Each tracked link is a sphere at a URDF frame. Each obstacle is a sphere. Pinocchio's
forward kinematics + frame Jacobian give an exact distance and its rate of change per
(link, obstacle) pair, so the whole safety constraint is linear in the commanded joint
velocity:

```
minimize    || q̇ − q̇_policy ||²
subject to  joint position/velocity limits
            clearance(link, obstacle) ≥ 0   for every tracked pair
```

```
perception source (any depth sensor)          policy (PI / lerobot / any VLA)
        |  crop -> cluster -> spheres              |  ~8-10Hz, own thread
        v  + self-filter vs arm pose                v
   LiveObstacleFeed  ------------------->  RealtimeSafetyFilter  ---> controller
                                            ~50-200Hz, reads latest
                                            q / action / obstacles every call
```

Perception is sensor-agnostic: `LiveObstacleFeed` takes a `point_cloud_fn() -> (N,3)
array`, so swapping cameras (or running fabricated obstacles in sim) is a one-function
change.

## Install

```bash
conda create -n vla-optim python=3.9
conda activate vla-optim
conda install -c conda-forge pinocchio=3.9.0
pip install -r requirements.txt
```

## Configure

- `config/robot_links.yaml` — pick a few URDF frames, give each a bounding-sphere radius.
- `config/camera_extrinsics.yaml` — sensor-to-base transform from your own hand-eye
  calibration. Shipped as identity — replace before trusting any obstacle position.

## Usage

No-hardware sanity check:

```bash
python examples/run_safety_filter_demo.py \
  --urdf tests/fixtures/test_robot.urdf --links-config tests/fixtures/test_links.yaml
```

Live loop:

```python
from vla_optim.safety_filter import RealtimeSafetyFilter
from vla_optim.perception import LiveObstacleFeed

filt = RealtimeSafetyFilter("path/to/robot.urdf", "config/robot_links.yaml", dt=0.02)
feed = LiveObstacleFeed("config/camera_extrinsics.yaml", point_cloud_fn=get_point_cloud)
feed.start()

while running:
    q = get_current_joint_positions()
    obstacles = feed.latest(q, filt)
    qdot_safe = filt.filter(q, policy.latest_action(), obstacles)
    send_to_controller(qdot_safe)
```

Full two-thread pattern (policy + filter at independent rates): `examples/integrate_with_policy_loop.py`.

## Benchmark

```bash
python benchmark/benchmark_latency.py --urdf path/to/robot.urdf --n-obstacles 1 3 5 10
```

Reports per-stage timing (distance/Jacobian, QP build, ProxQP solve) — check this against
your own collision-checking latency before assuming it's faster.

## vs. VLA-Optimization (O-VLA)

| | O-VLA V4/V6 | VLA-Optim |
|---|---|---|
| Solves for | full trajectory | single-step correction |
| Latency | 30ms–440ms | sub-step, benchmark to confirm |
| Collision model | full FCL mesh | link/obstacle spheres |
| Best for | offline repair / re-routing | online per-step reactive safety |

## Limitations

- Bounding spheres, not real mesh — no self-collision either.
- Obstacles assumed quasi-static within one control step.
- Self-filtering (dropping the arm's own geometry from detected obstacles) is heuristic.
- Corrects unsafe actions, doesn't route around a fully blocked goal — that's what the
  full O-VLA optimizer is for.
- Fails safe: an infeasible QP returns zero velocity, never an unfiltered command.

## License

MIT — see [LICENSE](LICENSE).
