# VLA-Optim

Real-time safety filter for VLA policies on real arms. Sits between the policy and the
controller: takes the policy's action for this step, returns the closest action that
respects joint/velocity/acceleration/torque limits, a Cartesian workspace region, and
clearance to obstacles - live, static, and the robot's own other links.

## Why

Full motion planners and full trajectory optimizers solve a bigger problem than needed
every control step - a whole path, or a whole multi-waypoint trajectory. This repo only
asks: *is the policy's next single action safe, and if not, what's the smallest fix?*
That's a QP with `n_joints` variables, not `n_waypoints * n_joints`, so it's cheap enough
to run every step instead of in the background.

## How it works

The decision variable is joint *acceleration* `q̈` (torque is a function of acceleration,
not velocity, so this is what lets torque limits sit in the same QP as everything else).
Each tracked link and each obstacle is a sphere; getting from a 3D obstacle position to a
corrected command is 4 steps:

1. **Position → clearance + direction.** FK gives the link's position `p_link(q)`:
   `clearance = ||p_link − p_obstacle|| − r_link − r_obstacle`, `direction = (p_link − p_obstacle) / ||p_link − p_obstacle||`

2. **Kinematics → constraint on acceleration.** The link's Jacobian `J_v(q)` (exact, from the
   URDF via Pinocchio) maps joint motion to that point's Cartesian motion, so the predicted
   clearance after one step is linear in `q̈` (`q̇_next = q̇ + q̈ dt`):
   `d(clearance)/dt ≈ direction · J_v(q) · q̇_next`

3. **One safety inequality per pair** (link↔obstacle, or link↔link for self-collision):
   `direction · J_v(q) · q̇_next ≥ −α (clearance − d_safe)` — don't let clearance shrink
   faster than the filter can brake for. Joint/velocity/position limits, torque limits
   (`τ = M(q) q̈ + h(q, q̇)`, from Pinocchio's rigid-body dynamics), and Cartesian workspace
   bounds are each their own linear row in the same `q̈`.

4. **QP = projection.** Minimize `‖q̈ − q̈_desired‖²` subject to every row above. The result
   is the closest acceleration to what the policy's action implied that still satisfies
   every constraint — ProxQP solves this directly, nothing is searched. Singularity
   avoidance isn't a hard row here — it's a soft nudge added to the cost, since a hard
   manipulability constraint would be nonconvex.

```
perception source (any depth sensor)          policy (PI / lerobot / any VLA)
        |  crop -> cluster -> spheres              |  ~8-10Hz, own thread
        v  + self-filter vs arm pose               v
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

- `config/robot_links.yaml`: link spheres, self-collision pairs, static/environment
  obstacles, acceleration limits, Cartesian workspace bounds, singularity thresholds. Every
  section is optional except `links`; see the comments in the file itself for the schema.
  Joint position/velocity/torque limits come straight from the URDF, no config needed.
- `config/camera_extrinsics.yaml`: sensor-to-base transform from your own hand-eye
  calibration. Shipped as identity, replace before trusting any obstacle position.

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

Reports per-stage timing (distance/Jacobian, QP build, ProxQP solve), check this against
your own collision-checking latency before assuming it's faster.

## Limitations

- Bounding spheres, not real mesh, accurate enough to be conservative, not tight, for
  odd-shaped or elongated links.
- Obstacles (and self-collision pairs) assumed quasi-static within one control step.
- Self-filtering (dropping the arm's own geometry from detected obstacles) is heuristic.
- Singularity avoidance is a soft cost bias, not a guarantee.
- Corrects unsafe actions, doesn't route around a fully blocked goal, that needs a full
  replanner, not a per-step filter.
- Fails safe: an infeasible or non-converged QP returns zero velocity, never an unfiltered
  command.

## License

MIT — see [LICENSE](LICENSE).
