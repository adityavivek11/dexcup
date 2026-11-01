"""Stage 5 stand-in: generate randomized scripted demos + automatic phase labels.

WHY scripted demos before your recordings?
  Every downstream stage (teacher, student, router, evaluation) can be built and debugged on clean data
  now. When your iPhone demos arrive, they enter at exactly this point (as replayed, physics-checked
  trajectories in the same file format), so only tracking + retargeting are new. If something breaks
  then, you know where to look.

WHY randomize?
  A policy trained on one cup position memorizes one trajectory. We vary cup start, coaster position,
  grasp offset (+-4 mm) and speed (+-15%) so the policy must actually look at where things are.

WHY an automatic phase labeler (instead of using the script's phase names)?
  Real demos have no script. The labeler below uses only physics signals available after replay
  (contacts, cup height, distances), so it will work unchanged on your recordings. Here we can check
  it against the script's ground truth, which is the only chance we get to validate it.

Output: data/scripted/ep_XXXX.npz, one per episode (failures kept, flagged), plus a summary.
Run:    python gen_demos.py --n 200 --workers 2
"""
import argparse
import os
from pathlib import Path
import numpy as np

PHASES = ["approach", "grasp", "transport", "place", "retreat"]
TEACHER_PHASES = {1, 3}            # grasp, place -> the big model; everything else -> the student
SCRIPT_TO_PHASE = {"approach_above": 0, "approach_down": 0, "slide_in": 1, "grasp": 1, "lift": 1,
                   "transport": 2, "place": 3, "release": 3, "back_off": 4, "retreat": 4}
OUT = Path(__file__).parent / "data" / "scripted"


def run_episode(seed: int):
    import mujoco
    import test_g1_grasp as T                       # model, IK, grasp checker, finger poses
    from ik import reset_ready
    m, d, ik, gc = T.m, T.d, T.ik, T.gc
    rng = np.random.default_rng(seed)

    # ---- randomize the scene -------------------------------------------------------------
    z = T.CUP_START[2]
    cup0 = np.array([rng.uniform(0.28, 0.36), rng.uniform(-0.24, -0.16), z])
    while True:
        coaster = np.array([rng.uniform(0.28, 0.36), rng.uniform(-0.03, 0.05), T.COASTER[2]])
        if np.linalg.norm(coaster[:2] - cup0[:2]) > 0.12:
            break
    m.geom_pos[m.geom("coaster").id] = coaster
    m.site_pos[m.site("coaster_center").id] = coaster + [0, 0, 0.002]
    reset_ready(m, d, T.STAND, ik)
    cup_adr = m.joint("cup_free").qposadr[0]
    d.qpos[cup_adr:cup_adr + 7] = [*cup0, 1, 0, 0, 0]
    mujoco.mj_forward(m, d)

    off = T.OFF + rng.uniform(-0.004, 0.004, 3)     # imperfect grasp placement, like a human's
    speed = rng.uniform(0.85, 1.15)
    on_coaster = np.array([coaster[0], coaster[1], z])
    C = T.CLEAR
    wps = [("approach_above", cup0 + off + C + [0, 0, 0.12], False, 1.2),
           ("approach_down",  cup0 + off + C,                False, 1.0),
           ("slide_in",       cup0 + off,                    False, 0.8),
           ("grasp",          cup0 + off,                    True,  0.8),
           ("lift",           cup0 + off + [0, 0, 0.09],     True,  0.8),
           ("transport",      on_coaster + off + [0, 0, 0.09], True, 1.5),
           ("place",          on_coaster + off + [0, 0, 0.005], True, 1.0),
           ("release",        on_coaster + off + [0, 0, 0.005], False, 0.6),
           ("back_off",       on_coaster + off + C + [0, 0, 0.005], False, 0.6),
           ("retreat",        on_coaster + off + C + [0, 0, 0.12], False, 0.8)]

    rec = {k: [] for k in ["arm_q", "arm_dq", "finger_q", "palm_pos", "palm_quat", "cup_pos",
                           "cup_quat", "coaster_pos", "action", "phase_script", "grasped",
                           "contact_any"]}
    cur = d.site("palm_center").xpos.copy()
    fq_adr = [m.joint(j).qposadr[0] for j in T.FINGER_JOINTS]
    arm_dof = [m.joint(j).dofadr[0] for j in T.RIGHT_ARM]
    for phase, target, closed, secs in wps:
        start, ticks = cur.copy(), max(1, int(secs * speed / T.CTRL_DT))
        fingers = T.FINGERS_CLOSE if closed else T.FINGERS_OPEN
        for k in range(ticks):
            a = min(1.0, k / (0.7 * ticks)); a = a * a * (3 - 2 * a)
            cur = start + a * (np.asarray(target) - start)
            # ---- observation BEFORE acting (what a policy would see) ----
            ok, f = gc.is_grasped(d)
            pq = np.zeros(4); mujoco.mju_mat2Quat(pq, d.site("palm_center").xmat)
            rec["arm_q"].append(d.qpos[ik.qadr].copy()); rec["arm_dq"].append(d.qvel[arm_dof].copy())
            rec["finger_q"].append(d.qpos[fq_adr].copy())
            rec["palm_pos"].append(d.site("palm_center").xpos.copy()); rec["palm_quat"].append(pq)
            rec["cup_pos"].append(d.qpos[cup_adr:cup_adr + 3].copy())
            rec["cup_quat"].append(d.qpos[cup_adr + 3:cup_adr + 7].copy())
            rec["coaster_pos"].append(coaster.copy())
            rec["grasped"].append(ok); rec["contact_any"].append(sum(f.values()) > 0.1)
            rec["phase_script"].append(SCRIPT_TO_PHASE[phase])
            # ---- action: palm target pose + finger targets (hand-agnostic pose space) ----
            finger_targets = np.array([fingers[j] for j in T.FINGER_JOINTS])
            rec["action"].append(np.concatenate([cur, T.HAND_QUAT, finger_targets]))
            arm_q, _, _ = ik.solve(d.qpos, cur, T.HAND_QUAT, iters=30)
            d.ctrl[T.ARM_ACT] = arm_q
            for j, q in fingers.items():
                d.ctrl[m.actuator(j).id] = q
            mujoco.mj_step(m, d, nstep=T.SUBSTEPS)

    ep = {k: np.array(v) for k, v in rec.items()}
    final_cup = d.qpos[cup_adr:cup_adr + 3]
    upright = d.body("cup").xmat.reshape(3, 3)[2, 2] > 0.95
    ep["success"] = np.array(bool(np.linalg.norm(final_cup[:2] - coaster[:2]) < 0.03 and upright))
    ep["phase_auto"] = label_phases(ep)
    ep["seed"] = np.array(seed)
    return ep


def label_phases(ep) -> np.ndarray:
    """Physics-only phase labels (works on replayed human demos too).
    approach  : until the palm is within 2 cm of the distance it will hold the cup at
    grasp     : until the cup has been lifted 2 cm
    transport : until the cup is within 3 cm (xy) of the coaster
    place     : until the hand has let go (no finger contact)
    retreat   : the rest
    """
    T = len(ep["grasped"])
    lab = np.zeros(T, dtype=int)
    g = np.flatnonzero(ep["grasped"])
    if len(g) == 0:
        return lab                                              # never grasped: all approach
    d_palm_cup = np.linalg.norm(ep["palm_pos"] - ep["cup_pos"], axis=1)
    d_hold = d_palm_cup[g[0]]
    t_grasp = np.flatnonzero(d_palm_cup < d_hold + 0.02)[0]
    z0 = ep["cup_pos"][0, 2]
    lifted = np.flatnonzero(ep["cup_pos"][:, 2] > z0 + 0.02)
    t_lift = lifted[0] if len(lifted) else T
    near = np.flatnonzero((np.linalg.norm(ep["cup_pos"][:, :2] - ep["coaster_pos"][:, :2], axis=1) < 0.03)
                          & (np.arange(T) > t_lift))
    t_place = near[0] if len(near) else T
    after = np.flatnonzero(~ep["contact_any"] & (np.arange(T) > t_place))
    t_ret = after[0] if len(after) else T
    for t0, ph in [(t_grasp, 1), (t_lift, 2), (t_place, 3), (t_ret, 4)]:
        lab[t0:] = ph
    return lab


def _worker(seed):
    if __import__("sys").platform.startswith("linux"):  # headless servers; macOS uses its native GL
        os.environ.setdefault("MUJOCO_GL", "egl")
    import warnings; warnings.filterwarnings("ignore")
    ep = run_episode(seed)
    np.savez_compressed(OUT / f"ep_{seed:04d}.npz", **ep)
    return seed, bool(ep["success"]), float((ep["phase_auto"] == ep["phase_script"]).mean()), len(ep["action"])


if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(_worker, range(args.n))
    succ = np.array([r[1] for r in res]); agree = np.array([r[2] for r in res])
    print(f"episodes={len(res)}  success={succ.mean():.1%}  mean_len={np.mean([r[3] for r in res]):.0f} ticks")
    print(f"auto-vs-script phase agreement (successful eps): mean={agree[succ].mean():.1%}  "
          f"worst={agree[succ].min():.1%}")
    # Where each boundary lands, auto minus script, in ms (negative = auto switches earlier)
    offs = {p: [] for p in PHASES[1:]}
    tf = []
    for s in np.array([r[0] for r in res])[succ]:
        e = np.load(OUT / f"ep_{s:04d}.npz")
        for i, p in enumerate(PHASES[1:], start=1):
            offs[p].append((np.argmax(e["phase_auto"] >= i) - np.argmax(e["phase_script"] >= i)) * 20)
        tf.append(np.isin(e["phase_auto"], list(TEACHER_PHASES)).mean())
    print("boundary offset auto-script (ms, median [p10,p90]): " + "  ".join(
        f"{p}:{np.median(v):+.0f}[{np.percentile(v,10):+.0f},{np.percentile(v,90):+.0f}]" for p, v in offs.items()))
    print(f"fraction of ticks in teacher phases (grasp+place): {np.mean(tf):.1%}  <- upper bound on savings")
