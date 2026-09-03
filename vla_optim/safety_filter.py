import numpy as np
import pinocchio as pin
import proxsuite
import yaml

from vla_optim.obstacles import Obstacle

LARGE = 1e20


class RealtimeSafetyFilter:
    """Per-step safety filter: minimally corrects a policy's joint-velocity
    command so it respects joint limits, velocity limits, and clearance to
    live obstacles.

    Each configured link is modeled as a sphere at a frame origin; each
    obstacle is a sphere. For every (link, obstacle) pair we get an analytic
    clearance and its Jacobian from Pinocchio, and enforce a control-barrier
    -style constraint:

        d(link, obs)/dt >= -alpha * (d(link, obs) - d_safe)

    which keeps clearance from shrinking faster than it can be braked for,
    without requiring the policy or the filter to know anything about
    obstacle motion beyond "static within one control step".
    """

    def __init__(self, urdf_path: str, links_config_path: str, dt: float):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.dt = dt
        self.nq = self.model.nq

        with open(links_config_path) as f:
            cfg = yaml.safe_load(f)

        self.d_safe = float(cfg["d_safe"])
        self.alpha = float(cfg["alpha"])
        self.link_frames = []
        self.link_radii = []
        for entry in cfg["links"]:
            frame_id = self.model.getFrameId(entry["frame"])
            if frame_id >= len(self.model.frames):
                raise ValueError(f"frame '{entry['frame']}' not found in URDF")
            self.link_frames.append(frame_id)
            self.link_radii.append(float(entry["radius"]))

        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit
        self.v_max = self.model.velocityLimit

        n_box = self.nq
        n_cbf_max = len(self.link_frames) * 8  # upper bound; resized per call if needed
        self._qp = proxsuite.proxqp.dense.QP(self.nq, 0, n_box + n_cbf_max)
        self._qp_n_in = n_box + n_cbf_max

    def link_spheres(self, q: np.ndarray) -> list:
        """Current (center, radius) for every configured link sphere, for
        use in obstacle self-filtering."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return [
            (self.data.oMf[frame_id].translation.copy(), radius)
            for frame_id, radius in zip(self.link_frames, self.link_radii)
        ]

    def filter(self, q: np.ndarray, qdot_pi: np.ndarray, obstacles: list) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q)

        C_rows = []
        l_rows = []
        u_rows = []

        # joint velocity + joint position box constraints, combined per-joint
        for i in range(self.nq):
            row = np.zeros(self.nq)
            row[i] = 1.0
            v_lo, v_hi = -self.v_max[i], self.v_max[i]
            p_lo = (self.q_min[i] - q[i]) / self.dt
            p_hi = (self.q_max[i] - q[i]) / self.dt
            C_rows.append(row)
            l_rows.append(max(v_lo, p_lo))
            u_rows.append(min(v_hi, p_hi))

        # CBF clearance constraints, one per (link, obstacle) pair
        for frame_id, link_radius in zip(self.link_frames, self.link_radii):
            p_link = self.data.oMf[frame_id].translation
            J = pin.getFrameJacobian(self.model, self.data, frame_id,
                                      pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_v = J[:3, :]

            for obs in obstacles:
                diff = p_link - obs.center
                dist = np.linalg.norm(diff)
                clearance = dist - link_radius - obs.radius
                if dist < 1e-6:
                    unit = np.zeros(3)
                else:
                    unit = diff / dist
                # d(clearance)/dt = unit^T J_v qdot  ;  require >= -alpha*(clearance - d_safe)
                row = unit @ J_v
                C_rows.append(row)
                l_rows.append(-self.alpha * (clearance - self.d_safe))
                u_rows.append(LARGE)

        C = np.vstack(C_rows)
        l = np.array(l_rows)
        u = np.array(u_rows)

        n_in = C.shape[0]
        if n_in != self._qp_n_in:
            self._qp = proxsuite.proxqp.dense.QP(self.nq, 0, n_in)
            self._qp_n_in = n_in

        H = np.eye(self.nq)
        g = -qdot_pi

        self._qp.init(H, g, None, None, C, l, u)
        self._qp.solve()

        x = self._qp.results.x
        if x is None:
            # infeasible box+CBF set (shouldn't happen with a static q_min/q_max
            # box, but obstacles pressing a joint limit can do it) -- brake to
            # zero rather than pass through an unfiltered command.
            return np.zeros(self.nq)
        return np.asarray(x)
