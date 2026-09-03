import numpy as np
import pinocchio as pin
import proxsuite
import yaml

from vla_optim.obstacles import Obstacle

LARGE = 1e20
ROW_EPS = 1e-9
MAX_QP_ITER = 100  # a real-time filter needs a bounded worst-case solve time
                    # far more than it needs ProxQP's default 10000-iteration
                    # budget to squeeze out extra precision on a hard/near-
                    # infeasible problem; fail fast into the zero-velocity
                    # fallback instead.
REF_FRAME = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED


def _symmetrize(M: np.ndarray) -> np.ndarray:
    lower = np.tril_indices(M.shape[0], -1)
    M = M.copy()
    M[lower] = M.T[lower]
    return M


def _append_row(C, l, u, row, lo, hi):
    """Skip (near-)zero coefficient rows -- they carry no real constraint
    information (hold for any q̈) and can happen exactly at kinematically
    degenerate poses, e.g. a tracked point sitting right on a joint's own
    rotation axis. Otherwise normalize to a unit-norm row before appending:
    box/torque/collision rows naturally come in wildly different scales
    (a joint-limit row has coefficient ~1, a torque row's can be ~0.01,
    a collision row's smaller still), and several such rows constraining
    the same direction at very different scales is enough to make ProxQP
    misreport infeasibility on an otherwise perfectly solvable problem."""
    norm = np.linalg.norm(row)
    if norm <= ROW_EPS:
        return
    C.append(row / norm)
    l.append(lo / norm if abs(lo) < LARGE else lo)
    u.append(hi / norm if abs(hi) < LARGE else hi)


class RealtimeSafetyFilter:
    """Per-step safety filter: minimally corrects a policy's action so it
    respects joint position/velocity/acceleration/torque limits, a
    Cartesian workspace region, and clearance to both live and static
    obstacles (including the robot's own other links).

    The decision variable is joint *acceleration* q̈ for this control step
    (velocity and position are then obtained by integrating it), because
    torque is a function of acceleration, not velocity -- everything else
    (limits, collision, workspace) is re-derived as a linear constraint on
    q̈ so it can all sit in the same QP:

        minimize    || q̈ - q̈_desired ||²
        subject to  acceleration limits
                    velocity limits        (on the predicted next velocity)
                    position limits        (on the predicted next position)
                    torque limits           (tau = M(q) q̈ + h(q, q̇))
                    clearance(link, obstacle) >= 0     for every tracked pair,
                    clearance(link, link)     >= 0     for configured self-collision pairs
                    workspace / Cartesian half-space constraints

    Obstacle and self-collision clearance use a control-barrier-style
    condition evaluated against the *predicted* next velocity
    (q̇_prev + q̈ dt): clearance shouldn't shrink faster than the filter can
    brake for. Singularity avoidance is a soft bias in the cost (nudging q̈
    away from directions that reduce manipulability) rather than a hard
    constraint, since manipulability is nonconvex and a hard constraint on
    it could make the QP infeasible.
    """

    def __init__(self, urdf_path: str, links_config_path: str, dt: float):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self._fd_data = self.model.createData()
        self.dt = dt
        self.nq = self.model.nq

        with open(links_config_path) as f:
            cfg = yaml.safe_load(f)

        self.d_safe = float(cfg["d_safe"])
        self.alpha = float(cfg["alpha"])

        self.link_frames = []
        self.link_radii = []
        self._frame_radius_by_name = {}
        for entry in cfg["links"]:
            frame_id = self.model.getFrameId(entry["frame"])
            if frame_id >= len(self.model.frames):
                raise ValueError(f"frame '{entry['frame']}' not found in URDF")
            radius = float(entry["radius"])
            self.link_frames.append(frame_id)
            self.link_radii.append(radius)
            self._frame_radius_by_name[entry["frame"]] = (frame_id, radius)

        self.self_collision_pairs = []
        for a, b in cfg.get("self_collision_pairs", []):
            self.self_collision_pairs.append((
                self._frame_radius_by_name[a], self._frame_radius_by_name[b]))

        self.static_obstacles = [
            Obstacle(center=np.array(o["center"], dtype=float), radius=float(o["radius"]))
            for o in cfg.get("static_obstacles", [])
        ]

        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit
        self.v_max = self.model.velocityLimit
        self.effort_max = self.model.effortLimit

        accel_cfg = cfg.get("accel_limits", 10.0)
        self.a_max = (np.full(self.nq, float(accel_cfg)) if np.isscalar(accel_cfg)
                       else np.array(accel_cfg, dtype=float))

        self.cartesian_frame = None
        self.workspace_bounds = None
        self.cartesian_planes = []
        cart_cfg = cfg.get("cartesian")
        if cart_cfg:
            self.cartesian_frame = self.model.getFrameId(cart_cfg["frame"])
            self.workspace_bounds = cart_cfg.get("workspace_box")
            self.cartesian_planes = [
                (np.array(p["normal"], dtype=float), float(p["offset"]))
                for p in cart_cfg.get("planes", [])
            ]

        self.singularity_frame = None
        sing_cfg = cfg.get("singularity")
        if sing_cfg:
            self.singularity_frame = self.model.getFrameId(sing_cfg["frame"])
            self.w_min = float(sing_cfg["min_manipulability"])
            self.sing_gain = float(sing_cfg["gain"])

        self._qdot_prev = np.zeros(self.nq)
        self._qp = None
        self._qp_n_in = None

    def link_spheres(self, q: np.ndarray) -> list:
        """Current (center, radius) for every configured link sphere, for
        use in obstacle self-filtering."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return [
            (self.data.oMf[frame_id].translation.copy(), radius)
            for frame_id, radius in zip(self.link_frames, self.link_radii)
        ]

    def _clearance_row(self, p_a, J_a, p_b, J_b, radius_sum, d_safe, v0):
        """One CBF constraint row for a pair of spheres (the second sphere
        may be static, in which case J_b is None)."""
        diff = p_a - p_b
        dist = np.linalg.norm(diff)
        clearance = dist - radius_sum
        unit = diff / dist if dist > 1e-6 else np.zeros(3)
        J_rel = J_a if J_b is None else (J_a - J_b)
        row = (unit @ J_rel) * self.dt          # coefficient on q̈
        rhs = -self.alpha * (clearance - d_safe) - unit @ J_rel @ v0
        return row, rhs

    def _box_rows(self, q, v0):
        C, l, u = [], [], []
        dt, dt2 = self.dt, self.dt ** 2
        for i in range(self.nq):
            row = np.zeros(self.nq)
            row[i] = 1.0
            a_lo, a_hi = -self.a_max[i], self.a_max[i]
            v_lo = (-self.v_max[i] - v0[i]) / dt
            v_hi = (self.v_max[i] - v0[i]) / dt
            p_lo = 2 * (self.q_min[i] - q[i] - v0[i] * dt) / dt2
            p_hi = 2 * (self.q_max[i] - q[i] - v0[i] * dt) / dt2
            _append_row(C, l, u, row, max(a_lo, v_lo, p_lo), min(a_hi, v_hi, p_hi))
        return C, l, u

    def _torque_rows(self, q, v0):
        M = _symmetrize(pin.crba(self.model, self.data, q))
        h = pin.nonLinearEffects(self.model, self.data, q, v0)
        C, l, u = [], [], []
        for row, lo, hi in zip(M, -self.effort_max - h, self.effort_max - h):
            _append_row(C, l, u, row, lo, hi)
        return C, l, u

    def _obstacle_rows(self, v0, obstacles):
        C, l, u = [], [], []
        for frame_id, radius in zip(self.link_frames, self.link_radii):
            p_link = self.data.oMf[frame_id].translation
            J_link = pin.getFrameJacobian(self.model, self.data, frame_id, REF_FRAME)[:3, :]
            for obs in obstacles:
                row, rhs = self._clearance_row(p_link, J_link, obs.center, None,
                                                radius + obs.radius, self.d_safe, v0)
                _append_row(C, l, u, row, rhs, LARGE)
        return C, l, u

    def _self_collision_rows(self, v0):
        C, l, u = [], [], []
        for (frame_a, radius_a), (frame_b, radius_b) in self.self_collision_pairs:
            p_a = self.data.oMf[frame_a].translation
            p_b = self.data.oMf[frame_b].translation
            J_a = pin.getFrameJacobian(self.model, self.data, frame_a, REF_FRAME)[:3, :]
            J_b = pin.getFrameJacobian(self.model, self.data, frame_b, REF_FRAME)[:3, :]
            row, rhs = self._clearance_row(p_a, J_a, p_b, J_b, radius_a + radius_b,
                                            self.d_safe, v0)
            _append_row(C, l, u, row, rhs, LARGE)
        return C, l, u

    def _cartesian_rows(self, v0):
        if self.cartesian_frame is None:
            return [], [], []
        C, l, u = [], [], []
        dt2 = self.dt ** 2
        p = self.data.oMf[self.cartesian_frame].translation
        J = pin.getFrameJacobian(self.model, self.data, self.cartesian_frame, REF_FRAME)[:3, :]
        drift = J @ v0 * self.dt  # position change already implied by current velocity

        if self.workspace_bounds:
            for axis, key in enumerate("xyz"):
                bounds = self.workspace_bounds.get(key)
                if not bounds:
                    continue
                lo, hi = bounds
                row = J[axis, :] * dt2
                _append_row(C, l, u, row, lo - p[axis] - drift[axis], hi - p[axis] - drift[axis])

        for normal, offset in self.cartesian_planes:
            row = (normal @ J) * dt2
            _append_row(C, l, u, row, -LARGE, offset - normal @ p - normal @ drift)

        return C, l, u

    def _manipulability(self, data, q, frame_id):
        pin.forwardKinematics(self.model, data, q)
        pin.updateFramePlacements(self.model, data)
        pin.computeJointJacobians(self.model, data, q)
        J = pin.getFrameJacobian(self.model, data, frame_id, REF_FRAME)[:3, :]
        return np.sqrt(max(np.linalg.det(J @ J.T), 0.0))

    def _singularity_bias(self, q):
        """Soft cost nudge, not a constraint: pushes q̈ toward directions
        that increase manipulability once it drops below w_min. Returns a
        (nq,) vector to subtract from the QP's linear cost term."""
        if self.singularity_frame is None:
            return np.zeros(self.nq)
        w = self._manipulability(self.data, q, self.singularity_frame)
        if w >= self.w_min:
            return np.zeros(self.nq)

        eps = 1e-4
        grad = np.zeros(self.nq)
        for i in range(self.nq):
            q_plus, q_minus = q.copy(), q.copy()
            q_plus[i] += eps
            q_minus[i] -= eps
            w_plus = self._manipulability(self._fd_data, q_plus, self.singularity_frame)
            w_minus = self._manipulability(self._fd_data, q_minus, self.singularity_frame)
            grad[i] = (w_plus - w_minus) / (2 * eps)

        return self.sing_gain * (self.w_min - w) * grad * self.dt

    def filter(self, q: np.ndarray, qdot_pi: np.ndarray, obstacles: list,
               qdot_current: np.ndarray = None) -> np.ndarray:
        v0 = self._qdot_prev if qdot_current is None else np.asarray(qdot_current)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q)

        all_obstacles = self.static_obstacles + list(obstacles)

        C_rows, l_rows, u_rows = self._box_rows(q, v0)
        for rows in (self._torque_rows(q, v0),
                     self._obstacle_rows(v0, all_obstacles),
                     self._self_collision_rows(v0),
                     self._cartesian_rows(v0)):
            C_rows += rows[0]
            l_rows += rows[1]
            u_rows += rows[2]

        C = np.vstack(C_rows)
        l = np.array(l_rows)
        u = np.array(u_rows)

        n_in = C.shape[0]
        if self._qp is None or n_in != self._qp_n_in:
            self._qp = proxsuite.proxqp.dense.QP(self.nq, 0, n_in)
            self._qp.settings.max_iter = MAX_QP_ITER
            self._qp_n_in = n_in

        qddot_desired = (qdot_pi - v0) / self.dt
        H = np.eye(self.nq)
        g = -qddot_desired - self._singularity_bias(q)

        self._qp.init(H, g, None, None, C, l, u)
        self._qp.solve()

        solved = self._qp.results.info.status == proxsuite.proxqp.PROXQP_SOLVED
        if not solved:
            # infeasible constraint set (e.g. an obstacle pressing a joint/
            # torque limit at once) -- fail safe: command zero velocity
            # rather than pass through an unfiltered or garbage command.
            # (ProxQP's results.x is not reliably None on infeasibility, so
            # the solve status has to be checked explicitly.)
            qdot_safe = np.zeros(self.nq)
        else:
            qdot_safe = v0 + np.asarray(self._qp.results.x) * self.dt

        self._qdot_prev = qdot_safe
        return qdot_safe
