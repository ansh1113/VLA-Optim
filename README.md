# VLA-Optim

A lightweight, real-time safety filter for VLA / imitation-learning policies running on
real robot arms. It sits between the policy and the controller, taking whatever action the
policy wants to execute and returning the closest action that is guaranteed to respect joint
limits, velocity limits, and live obstacle clearance.

This is a companion project to [VLA-Optimization](https://github.com/ansh1113/VLA-Optimization)
(O-VLA), which does full trajectory optimization (ProxQP + CHOMP + TOPP-RA / CasADi + IPOPT).
That pipeline is great for offline trajectory repair but costs 30ms-440ms per call because it
optimizes an entire multi-waypoint trajectory. This repo asks a narrower question instead:

> Given the policy's action for *this single step*, what is the smallest correction that
> keeps the arm safe?

That narrower question turns into a QP with as many variables as the robot has joints (7 for
a Panda) instead of `n_waypoints * n_joints`, which is what makes it fast enough to sit inside
a per-step control loop rather than a background planner.

## Why this exists

Our policy currently runs collision checking through MoveIt at ~18ms/step. Before replacing
that, or bolting the full O-VLA trajectory optimizer onto the live loop, it's worth
benchmarking a much smaller alternative:

```
policy action (q̇_pi)
        |
        v
  ┌─────────────────────────────┐
  │   RealtimeSafetyFilter      │
  │                              │
  │  minimize  ||q̇ - q̇_pi||²   │
  │  s.t.                        │
  │    joint position limits     │
  │    joint velocity limits     │
  │    clearance(link, obstacle) │  <- control-barrier-style constraint
  │                              │
  └─────────────────────────────┘
        |
        v
  safe action (q̇_safe) -> controller
```

Each robot link is approximated as a sphere centered at a configured frame origin. Each live
obstacle (from the ZED feed) is also a sphere. For every (link, obstacle) pair we get an
analytic distance and its Jacobian for free from Pinocchio's frame Jacobian, so the whole
constraint set is linear in `q̇` and the QP stays tiny: ~7 variables, ~10-30 inequality rows
depending on obstacle count. No sampling, no trajectory, no multi-waypoint optimization.

This is a deliberately simpler model than full mesh-to-mesh FCL distance (which is what
`collision_detector_v3.py` in the O-VLA repo does) traded for speed. See **Limitations**
below for what that trade costs you.

## Architecture

```
                 ZED RGB-D camera
                        |
                        v
              ┌───────────────────┐
              │  LiveObstacleFeed │  background thread, ~15-30Hz
              │  (zed_perception) │  point cloud -> crop -> cluster -> spheres
              └─────────┬─────────┘      + self-filtering vs current arm pose
                        |
                        v  latest obstacle list (thread-safe)
                        |
policy (PI / lerobot) --+--> RealtimeSafetyFilter --> safe action --> robot controller
  runs at ~8-10Hz              runs at 50-200Hz, reads latest q and
  (own thread/process)         latest obstacles every call
```

The policy and the perception feed are decoupled from the filter: the filter always
consumes whatever the *latest* policy action and *latest* obstacle set are, and runs at its
own (much higher) rate. This is what gives you reactive avoidance in between policy
decisions, not just at each policy tick.

## Install

```bash
conda create -n vla-optim python=3.9
conda activate vla-optim
conda install -c conda-forge pinocchio=3.9.0
pip install -r requirements.txt
```

The ZED SDK itself (and its `pyzed` Python bindings) is not pip-installable — install it
from [Stereolabs](https://www.stereolabs.com/developers/release/) matching your camera and
CUDA version first, then `pip install` will pick up `pyzed` from the SDK's bundled wheel.

## Configure

1. **Link spheres** (`config/robot_links.yaml`): pick a handful of frames along your URDF
   (e.g. each link origin, plus the end-effector) and give each a bounding radius that
   comfortably contains the real geometry at that point. Fewer, larger spheres = faster and
   more conservative. This is the main thing you'll want to tune per-robot.

2. **Camera extrinsics** (`config/camera_extrinsics.yaml`): the ZED reports points in its
   own optical frame. You need the camera-to-robot-base transform from your own hand-eye
   calibration; there isn't a way around doing that calibration once per rig. The file is a
   placeholder identity transform — replace it before trusting any live obstacle position.

## Usage

Sanity-check the math with no hardware at all. `tests/fixtures/` has a tiny
3-joint synthetic URDF + link config if you don't have a real one handy yet:

```bash
python examples/run_safety_filter_demo.py \
  --urdf tests/fixtures/test_robot.urdf --links-config tests/fixtures/test_links.yaml
```

Wire it into a live policy loop:

```python
from vla_optim.safety_filter import RealtimeSafetyFilter
from vla_optim.zed_perception import LiveObstacleFeed

filt = RealtimeSafetyFilter("path/to/robot.urdf", "config/robot_links.yaml", dt=0.02)
feed = LiveObstacleFeed("config/camera_extrinsics.yaml")
feed.start()

while running:
    q = get_current_joint_positions()
    qdot_pi = policy.latest_action()          # whatever your policy last produced
    obstacles = feed.latest(q, filt)          # self-filtered against current arm pose
    qdot_safe = filt.filter(q, qdot_pi, obstacles)
    send_to_controller(qdot_safe)
```

See `examples/integrate_with_policy_loop.py` for the full two-thread pattern (policy at
~10Hz, filter at ~50-100Hz).

## Benchmark it against your MoveIt number before trusting it

```bash
python benchmark/benchmark_latency.py --urdf path/to/robot.urdf --n-obstacles 1 3 5 10
```

This reports per-stage timing (distance+Jacobian eval, QP build, ProxQP solve) so you can
see directly whether it beats your measured MoveIt collision-check latency, rather than
assuming it does.

| stage | what it measures |
|---|---|
| distance/jacobian | FK + per-(link,obstacle) distance and Jacobian, analytic |
| qp build | stacking constraint rows into `C, l, u` |
| qp solve | ProxQP dense solve |

## Limitations (read before deploying)

- **Sphere approximation only.** Links are treated as spheres at a single frame origin, not
  their real mesh. This is conservative in the wrong direction if the sphere doesn't fully
  contain the link, and needlessly restrictive if the sphere is much bigger than the link.
  If you need tight non-spherical clearance, use the full FCL pipeline in the O-VLA repo
  instead.
- **No self-collision.** Only robot-vs-obstacle spheres are checked, not link-vs-link.
- **Obstacles assumed quasi-static within one control step.** The CBF constraint doesn't
  account for obstacle velocity; fine at 50-100Hz filter rates for slow-moving obstacles,
  not fine for fast-moving ones.
- **Self-filtering is heuristic.** Points near the arm's own current-pose spheres are
  dropped from the obstacle set; a poorly calibrated extrinsic or a fast-moving arm can
  leak some of the arm itself back in as a false obstacle.
- **Camera extrinsics must be calibrated by you.** The shipped config is an identity
  placeholder.
- This filter only ever makes the policy's action *safer or equal*, never *smarter* — it
  won't route around an obstacle that fully blocks the goal, it will just stop short of it.
  For actual re-routing, fall back to the full O-VLA trajectory optimizer.

## Relationship to VLA-Optimization (O-VLA)

| | O-VLA V4 / V6 | VLA-Optim (this repo) |
|---|---|---|
| Solves for | full trajectory (10-50 waypoints) | single-step action correction |
| Solvers | ProxQP + CHOMP + TOPP-RA, or CasADi/IPOPT | ProxQP only |
| Reported latency | 30ms-440ms per call | target: sub-18ms, benchmark to confirm |
| Collision model | full FCL mesh distance | link/obstacle spheres |
| Best for | offline repair, re-routing around a blocking obstacle | online per-step reactive safety |

## License

MIT — see [LICENSE](LICENSE).
