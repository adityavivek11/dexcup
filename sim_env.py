"""Closed-loop rollouts: the policy only sees make_obs(...) and outputs (relative palm target, grip).

Same controller as the demos (50 Hz palm target -> IK -> servos, 500 Hz physics), so a policy is judged
in exactly the loop its training data came from. Every policy call is timed: that's where the latency
numbers in the results table come from.
"""
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent
T_MAX = 1000                  # ticks = 20 s (longest successful demo: 19.4 s)
HOLD_OK = 50                  # success once the tube sits on the coaster, upright, released, for 1 s


class Env:
    def __init__(self):
        import warnings; warnings.filterwarnings("ignore")
        import test_g1_grasp as T
        self.T, self.m, self.d, self.ik, self.gc = T, T.m, T.d, T.ik, T.gc
        m = self.m
        self.cup = m.joint("cup_free").qposadr[0]
        self.fq = [m.joint(j).qposadr[0] for j in T.FINGER_JOINTS]
        self.close = np.array([T.FINGERS_CLOSE[j] for j in T.FINGER_JOINTS])
        self.open = np.array([T.FINGERS_OPEN[j] for j in T.FINGER_JOINTS])
        self.f_act = [m.actuator(j).id for j in T.FINGER_JOINTS]
        self.arm_dof = [m.joint(j).dofadr[0] for j in T.RIGHT_ARM]

    def reset(self, cup_xy, coaster_xy):
        import mujoco
        from ik import reset_ready
        m, d = self.m, self.d
        m.geom_pos[m.geom("coaster").id][:2] = coaster_xy
        m.site_pos[m.site("coaster_center").id][:2] = coaster_xy
        reset_ready(m, d, self.T.STAND, self.ik)
        d.qpos[self.cup:self.cup + 7] = [cup_xy[0], cup_xy[1], self.T.CUP_START[2], 1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        self.coaster = np.r_[coaster_xy, m.geom_pos[m.geom("coaster").id][2]]
        self.ok_ticks = 0

    def obs(self):
        from policy import make_obs
        d = self.d
        # always the full 38-d observation (velocities at the end); each agent keeps what its recipe uses
        return make_obs(d.site("palm_center").xpos, d.qpos[self.cup:self.cup + 3],
                        d.qpos[self.cup + 3:self.cup + 7], self.coaster, d.qpos[self.fq], d.qpos[self.ik.qadr],
                        d.qvel[self.arm_dof])[0]

    def palm(self):
        return self.d.site("palm_center").xpos.copy()

    def step(self, target, grip_closed):
        import mujoco
        q, _, _ = self.ik.solve(self.d.qpos, target, self.T.HAND_QUAT, iters=30)
        self.d.ctrl[self.T.ARM_ACT] = q
        self.d.ctrl[self.f_act] = self.close if grip_closed else self.open
        mujoco.mj_step(self.m, self.d, nstep=self.T.SUBSTEPS)

    def status(self):
        """'success' | 'fail' | None (still running). Uses privileged sim state: evaluation only."""
        d = self.d
        cup = d.qpos[self.cup:self.cup + 3]
        up = d.body("cup").xmat.reshape(3, 3)[2, 2]
        if up < 0.5 or cup[2] < self.T.CUP_START[2] - 0.05:
            return "fail"                                       # fallen over or off the table
        grasped, f = self.gc.is_grasped(d)
        on = np.linalg.norm(cup[:2] - self.coaster[:2]) < 0.03 and up > 0.95 and cup[2] > self.T.CUP_START[2] + 0.01
        # "Released" = NOT grasped by the project's grasp definition (contacts.py: thumb AND index/middle
        # > 0.5 N). Changed 3 Nov after looking at teacher v3 results (disclosed in CLAUDE.md): the first
        # version demanded zero finger contact, stricter than both the grasp definition and the replay
        # filter that built the training data, and failed a tube released 0.8 cm from centre whose open
        # fingers were still brushing it.
        released = not grasped
        self.ok_ticks = self.ok_ticks + 1 if (on and released) else 0
        return "success" if self.ok_ticks >= HOLD_OK else None


class TeacherAgent:
    """Calls the teacher every tick and blends overlapping chunks (ACT temporal ensembling): the action
    for tick t averages every past chunk's prediction for t, newer chunks weighted more (exp(-0.1*age)).
    Done in ABSOLUTE palm targets, since each chunk was predicted relative to the anchor object (tube or
    coaster, policy.anchor) at its own time."""
    def __init__(self, model, m=0.1):
        from policy import DEFAULT_CFG, obs_dim
        self.model, self.m = model.eval(), m
        self.cfg = getattr(model, "cfg", DEFAULT_CFG)
        self.dim = obs_dim(self.cfg)
        self.reset()                                            # usable without an explicit reset()

    def reset(self):
        self.buf = []                                         # (issue tick, abs targets (H,3), grip prob (H,))

    def __call__(self, obs, palm, t):
        from policy import ref_point
        with torch.no_grad():
            out = self.model(torch.from_numpy(obs[:self.dim])[None])[0].numpy()
        ref = ref_point(obs[:31], palm, self.cfg["frame"])[0]
        self.buf.append((t, ref + out[:, :3], 1 / (1 + np.exp(-out[:, 3]))))
        self.last_chunk = self.buf[-1][1:]                      # this tick's raw chunk (abs targets, grip prob)
        self.buf = [b for b in self.buf if t - b[0] < len(b[1])]
        w, tgt, g = [], [], []
        for t0, a, p in self.buf:
            k = t - t0
            w.append(np.exp(-self.m * k)); tgt.append(a[k]); g.append(p[k])
        w = np.array(w) / np.sum(w)
        return w @ np.array(tgt), float(w @ np.array(g))


class StudentAgent:
    def __init__(self, model):
        from policy import DEFAULT_CFG, obs_dim
        self.model = model.eval()
        self.cfg = getattr(model, "cfg", DEFAULT_CFG)
        self.dim = obs_dim(self.cfg)
        # chunked student: run exactly like the teacher (same temporal ensembling), see policy.Student
        self.ens = TeacherAgent(model) if getattr(model, "chunk", False) else None

    def reset(self):
        if self.ens:
            self.ens.reset()

    def __call__(self, obs, palm, t):
        if self.ens:
            return self.ens(obs, palm, t)
        from policy import ref_point
        with torch.no_grad():
            out = self.model(torch.from_numpy(obs[:self.dim])[None])[0].numpy()
        return ref_point(obs[:31], palm, self.cfg["frame"])[0] + out[:3], float(1 / (1 + np.exp(-out[3])))


def rollout(env, agent, cup_xy, coaster_xy, record=False):
    """Run one episode. Returns dict(success, ticks, call_ms list, [obs, phase-relevant logs])."""
    env.reset(cup_xy, coaster_xy)
    agent.reset()
    ms, log = [], []
    status = None
    for t in range(T_MAX):
        o, palm = env.obs(), env.palm()
        t0 = time.perf_counter()
        target, grip = agent(o, palm, t)
        ms.append((time.perf_counter() - t0) * 1000)
        if record:
            log.append((o, target, grip))
        env.step(target, grip > 0.5)
        status = env.status()
        if status:
            break
    return dict(success=status == "success", ticks=t + 1, call_ms=ms, log=log)
