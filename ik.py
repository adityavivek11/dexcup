"""Inverse kinematics for the G1 right arm: hand pose in the world -> 7 arm joint angles.

WHY IK?
  Your recordings give *hand poses* (where your wrist is, which way it faces). The robot takes *joint
  angles*. IK converts one to the other. This is also why the policy can work in "hand pose" space:
  a pose is the same quantity for a human and a robot, while joint angles are not.

HOW (damped least squares, the workhorse method):
  1. Measure the error e between the current and target palm pose (3 position + 3 rotation numbers).
  2. The Jacobian J (6x7) says how much the palm pose moves per small change of each joint.
  3. Solve J dq = e for the joint step dq. With 7 joints and 6 constraints there are many solutions,
     and near "straight-arm" poses J is nearly singular, so add damping:
         dq = J^T (J J^T + lambda^2 I)^-1 e
     The damping trades a little accuracy for never producing huge, jerky joint jumps.
  4. Clamp to joint limits and repeat until the error is small.
"""
import mujoco
import numpy as np
from build_g1_scene import RIGHT_ARM


class ArmIK:
    def __init__(self, model: mujoco.MjModel, site="palm_center", body="right_wrist_yaw_link"):
        self.m = model
        self.d = mujoco.MjData(model)           # private scratch copy: IK never disturbs the sim
        self.site = model.site(site).id
        self.body = model.body(body).id
        self.jids = [model.joint(j).id for j in RIGHT_ARM]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.jids])
        self.lo = model.jnt_range[self.jids, 0]
        self.hi = model.jnt_range[self.jids, 1]
        self.q_nominal = None       # set to a preferred arm posture (7,) to enable the null-space term
        self.posture_gain = 0.1

    def solve(self, qpos_now, target_pos, target_quat=None, iters=200, tol=1e-3, damping=0.05,
              rot_weight=0.5):
        """Return (arm_q (7,), pos_err_m, rot_err_rad). target_quat None -> position only."""
        d = self.d
        d.qpos[:] = qpos_now
        jacp = np.zeros((3, self.m.nv)); jacr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, d); mujoco.mj_comPos(self.m, d)
            e_pos = target_pos - d.site_xpos[self.site]
            if target_quat is not None:
                cur = np.zeros(4); mujoco.mju_mat2Quat(cur, d.xmat[self.body])
                neg = np.zeros(4); mujoco.mju_negQuat(neg, cur)
                dq = np.zeros(4); mujoco.mju_mulQuat(dq, target_quat, neg)
                e_rot = np.zeros(3); mujoco.mju_quat2Vel(e_rot, dq, 1.0)   # axis*angle in world frame
            else:
                e_rot = np.zeros(3)
            if np.linalg.norm(e_pos) < tol and np.linalg.norm(e_rot) < 10 * tol:
                break
            mujoco.mj_jacSite(self.m, d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dadr], rot_weight * jacr[:, self.dadr]]) if target_quat is not None \
                else jacp[:, self.dadr]
            e = np.concatenate([e_pos, rot_weight * e_rot]) if target_quat is not None else e_pos
            Jpinv = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(len(e)))
            step = Jpinv @ e
            if self.q_nominal is not None:
                # Null-space posture term: (I - J+ J) projects a "move toward the nominal posture" step
                # onto directions that don't change the hand pose. 7 joints - 6 constraints = 1 spare
                # direction (the elbow swivel), which this uses to keep the arm natural and on one branch.
                N = np.eye(len(self.qadr)) - Jpinv @ J
                step += N @ (self.posture_gain * (self.q_nominal - d.qpos[self.qadr]))
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + step, self.lo, self.hi)
        mujoco.mj_kinematics(self.m, d)
        pe = np.linalg.norm(target_pos - d.site_xpos[self.site])
        re = np.linalg.norm(e_rot) if target_quat is not None else 0.0
        return d.qpos[self.qadr].copy(), pe, re


def reset_ready(m, d, stand_qpos, ik: "ArmIK"):
    """Reset to standing, then put the right arm in the 'ready to reach' pose (hand above the table).
    Episodes start here, as on the real robot: the arm is raised before the task begins."""
    from build_g1_scene import reset, ARM_NOMINAL, READY_PALM
    reset(m, d, stand_qpos)
    ik.q_nominal = np.array(ARM_NOMINAL)
    q = d.qpos.copy(); q[ik.qadr] = ARM_NOMINAL
    arm, _, _ = ik.solve(q, np.array(READY_PALM), np.array([1.0, 0, 0, 0]), iters=300)
    d.qpos[ik.qadr] = arm
    for jname_idx, adr in enumerate(ik.qadr):
        d.ctrl[m.actuator(m.joint(ik.jids[jname_idx]).name).id] = arm[jname_idx]
    mujoco.mj_forward(m, d)
