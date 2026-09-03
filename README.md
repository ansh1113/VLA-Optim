# VLA-Optim

A lightweight, real-time safety filter for VLA / imitation-learning policies running on
real robot arms. It sits between the policy and the controller, taking whatever action the
policy wants to execute this step and returning the closest action that is guaranteed to
respect joint limits, velocity limits, and live obstacle clearance.

## Why this exists

VLA and imitation-learning policies output raw joint or Cartesian commands with no notion
of "is this safe." Somewhere between the policy and the motors, something has to check that.
The two obvious options both have a problem for a policy that's making a new decision every
80-125ms:

- **A full motion planner (MoveIt / OMPL-style sampling planners)** is built to answer "find
  me a whole path from A to B," which is a bigger question than the one you're actually
  asking every control step. That generality costs latency.
- **A full trajectory optimizer** (this project's sibling repo,
  [VLA-Optimization](https://github.com/ansh1113/VLA-Optimization), which does ProxQP + CHOMP
  + TOPP-RA or CasADi + IPOPT) is the right tool when you need to *repair a whole planned
  trajectory* around obstacles, but it optimizes `n_waypoints * n_joints` variables at once,
  which is why it costs 30ms-440ms per call there. That's fine offline; it's a lot to spend
  every single policy tick.

VLA-Optim asks a narrower question instead:

> Given the policy's action for *this single step*, what is the smallest correction that
> keeps the arm safe?

That question turns into a QP with as many variables as the robot has joints (7 for a
Panda), not `n_waypoints * n_joints`. Smaller problem, much less time to solve it — which is
what makes it viable to run inside the control loop itself rather than as a background
planner.

## How it works

Every robot link you care about is approximated as a sphere centered at a configured URDF
frame. Every obstacle — wherever it comes from — is also a sphere. For every (link,
obstacle) pair, Pinocchio's forward kinematics and frame Jacobian give you an exact,
analytic distance and *rate of change of distance* with respect to the joint command, for
free, without needing to walk a mesh-to-mesh collision checker for every candidate action.
That turns "stay clear of this obstacle" into one linear inequality per pair — the whole
safety constraint set is linear in the commanded joint velocity, so the optimization stays a
small, structured QP instead of a search.

```
minimize      || q̇ − q̇_policy ||²
subject to    joint position limits
              joint velocity limits
              clearance(link, obstacle) ≥ 0   for every tracked (link, obstacle) pair
```

The clearance constraint is a control-barrier-style condition: it doesn't require an
obstacle to be perfectly static, only that it not close distance faster than the filter (run
at 50-200Hz) can react to and brake for.

### End-to-end workflow

```
                     perception source (any depth sensor / point cloud)
                                    |
                                    v
                          ┌───────────────────┐
                          │  LiveObstacleFeed │   background thread
                          │   (perception.py) │   crop -> cluster -> spheres
                          └─────────┬─────────┘   + self-filter vs current arm pose
                                    |
                                    v   latest obstacle list (thread-safe)
                                    |
policy (PI / lerobot / any VLA) ---+---> RealtimeSafetyFilter ---> safe action ---> controller
     runs at ~8-10Hz                        runs at 50-200Hz, always reads the
     own thread/process                     latest q, latest policy action,
                                             and latest obstacles on every call
```

The policy, the perception feed, and the filter are three independent loops running at three
independent rates. The filter never blocks waiting on the policy or the camera — it always
consumes whatever the *latest* values are. That decoupling is what gives you obstacle
reactivity *between* policy decisions, not just once per policy tick.

### Why the perception layer is sensor-agnostic

`LiveObstacleFeed` doesn't know or care what camera you have. You hand it one function —
`point_cloud_fn() -> (N, 3) array in the camera's own frame` — and everything downstream
(workspace cropping, clustering into spheres, transforming into the robot base frame,
filtering the arm's own geometry back out) is identical no matter what's on the other end of
that callback: a stereo camera, a single depth sensor, an existing perception node's output,
or a simulator feeding synthetic points during testing. Swapping cameras, or running the
exact same filter in sim with fabricated obstacles, is a one-function change, not a rewrite.

## Design principles

- **Correct a single action, don't replan a trajectory.** The whole point is a QP small
  enough to run every control step.
- **Morphology-agnostic.** Loads any URDF via Pinocchio; there's no per-robot tuning beyond
  picking which frames get a bounding sphere.
- **Sensor-agnostic perception.** See above — the filter's math has no idea what camera fed
  it obstacles.
- **Deliberately simple collision model.** Bounding spheres instead of full mesh distance,
  traded for speed and for constraints that are exactly linear in the decision variable. See
  Limitations below for what that trade costs.
- **Never makes the policy's action less safe, never makes it smarter.** This filter can only
  slow or redirect an unsafe action toward safety; it can't route around an obstacle that
  fully blocks the goal. That's what the full trajectory optimizer in VLA-Optimization is
  for.

## Install

```bash
conda create -n vla-optim python=3.9
conda activate vla-optim
conda install -c conda-forge pinocchio=3.9.0
pip install -r requirements.txt
```

No depth-sensor SDK is a dependency of this repo — install whatever your camera needs
separately and wrap it in a `point_cloud_fn` (see Usage below).

## Configure

1. **Link spheres** (`config/robot_links.yaml`): pick a handful of frames along your URDF
   (e.g. each link origin, plus the end-effector) and give each a bounding radius that
   comfortably contains the real geometry at that point. Fewer, larger spheres = faster and
   more conservative. This is the main thing you'll want to tune per-robot.

2. **Camera extrinsics** (`config/camera_extrinsics.yaml`): whatever depth sensor you use
   reports points in its own frame. You need the sensor-to-robot-base transform from your
   own hand-eye calibration; there's no way around doing that calibration once per rig. The
   file is a placeholder identity transform — replace it before trusting any live obstacle
   position.

## Usage

Sanity-check the math with no hardware at all. `tests/fixtures/` has a tiny 3-joint
synthetic URDF + link config if you don't have a real one handy yet:

```bash
python examples/run_safety_filter_demo.py \
  --urdf tests/fixtures/test_robot.urdf --links-config tests/fixtures/test_links.yaml
```

Wire it into a live policy loop:

```python
from vla_optim.safety_filter import RealtimeSafetyFilter
from vla_optim.perception import LiveObstacleFeed

def get_point_cloud():
    # wrap whatever depth sensor/SDK you have; return an (N, 3) array of
    # points in the camera's own frame, or None if no new frame is ready
    ...

filt = RealtimeSafetyFilter("path/to/robot.urdf", "config/robot_links.yaml", dt=0.02)
feed = LiveObstacleFeed("config/camera_extrinsics.yaml", point_cloud_fn=get_point_cloud)
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

## Benchmark it before trusting it

```bash
python benchmark/benchmark_latency.py --urdf path/to/robot.urdf --n-obstacles 1 3 5 10
```

This reports per-stage timing (distance+Jacobian eval, QP build, ProxQP solve) so you can
see directly whether it beats whatever your current collision-checking latency is, rather
than assuming it does.

| stage | what it measures |
|---|---|
| distance/jacobian | FK + per-(link,obstacle) distance and Jacobian, analytic |
| qp build | stacking constraint rows into `C, l, u` |
| qp solve | ProxQP dense solve |

## Relationship to VLA-Optimization (O-VLA)

| | O-VLA V4 / V6 | VLA-Optim (this repo) |
|---|---|---|
| Solves for | full trajectory (10-50 waypoints) | single-step action correction |
| Solvers | ProxQP + CHOMP + TOPP-RA, or CasADi/IPOPT | ProxQP only |
| Reported latency | 30ms-440ms per call | designed for well under one policy control step; benchmark to confirm on your hardware |
| Collision model | full FCL mesh distance | link/obstacle bounding spheres |
| Best for | offline repair, re-routing around a blocking obstacle | online per-step reactive safety |

They're complementary: this repo answers "is the policy's next single action safe," the
other answers "find me a whole safe trajectory." A system could reasonably use both — this
filter on every step, the full optimizer invoked only when the filter detects it can't find
a feasible correction (i.e. the policy is driving straight at something it can't get around
one step at a time).

## Anticipated questions

**Why not just use a full planner's built-in real-time servo mode?**
Servo-style real-time control in a full planning stack usually means "check the current
command against the scene and slow/stop near collisions" — which is close to what this repo
does, just without the planning stack's dependency weight (no full workspace/graph
infrastructure required to run it) and with an explicit, inspectable QP instead of an
opaque scaling heuristic. If you already have a mature servo pipeline in place and it hits
your latency target, there may be little reason to switch — benchmark both and compare.

**Why bounding spheres instead of real mesh collision?**
Speed and linearity: a sphere-vs-sphere distance and its Jacobian are closed-form, so the
entire constraint set is exactly linear in `q̇` and the QP is tiny. Real mesh distance
queries are more accurate but nonlinear and more expensive per pair — worth it for offline
trajectory repair, likely not worth it for a filter meant to run every control step. If your
links are far from spherical, this is the first place accuracy gets traded for speed —
tune sphere placement and radius accordingly.

**What happens if the QP is infeasible (e.g. an obstacle is pressing against a joint limit)?**
The filter returns zero velocity rather than passing through an unfiltered command. It fails
safe, not silent.

## Limitations (read before deploying)

- **Sphere approximation only.** Links are treated as spheres at a single frame origin, not
  their real mesh. This is conservative in the wrong direction if the sphere doesn't fully
  contain the link, and needlessly restrictive if the sphere is much bigger than the link.
- **No self-collision.** Only robot-vs-obstacle spheres are checked, not link-vs-link.
- **Obstacles assumed quasi-static within one control step.** The constraint doesn't account
  for obstacle velocity; fine at 50-100Hz filter rates for slow-moving obstacles, not fine
  for fast-moving ones.
- **Self-filtering is heuristic.** Points near the arm's own current-pose spheres are
  dropped from the obstacle set; a poorly calibrated extrinsic or a fast-moving arm can leak
  some of the arm itself back in as a false obstacle.
- **Sensor extrinsics must be calibrated by you.** The shipped config is an identity
  placeholder.
- **Corrects, doesn't route.** This filter only ever makes the policy's action safer or
  equal, never smarter — it won't find a way around an obstacle that fully blocks the goal,
  it will just stop short of it.

## License

MIT — see [LICENSE](LICENSE).
