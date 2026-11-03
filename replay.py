"""Stage 5: replay retargeted human demos on the simulated G1, keep the ones that physically work.

WHY replay in physics instead of training on the retargeted plans directly?
  A plan is only a wish: the arm may not reach, the tube may slip, the place may miss the roll. Replaying
  with the same controller the policies will use (50 Hz palm target -> IK -> servos, 500 Hz physics) gives
  (a) a success/failure filter and (b) observations that are physically consistent with the actions,
  which is what the teacher and student are trained on. Same record format as gen_demos.py, so every later
  stage reads scripted and human-derived episodes alike.

Output: data/replayed/<clip>.npz (failures kept, flagged) and a summary.
Run:    python replay.py [clips...] [--S 0.5] [--workers 6]
"""
import argparse
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "replayed"
DART_OUT = ROOT / "data" / "dart"
# Noise injection (DART), used when the teacher cut corners on the approach: an Ornstein-Uhlenbeck random
# walk is added to the EXECUTED palm target while the LABEL stays the clean plan target, so the data
# contains states off the demo path labelled with the way back. Smooth (tau 0.5 s), ~1 cm: per-tick white
# noise mostly cancels in the servo and creates no real deviation. Zero while closing on / releasing the
# tube: the grasp tolerates only ~+-5 mm, so noise there mainly knocks the tube over.
NOISE_SIGMA, NOISE_TAU = 0.01, 0.5
PHASE_ID = {"approach": 0, "grasp": 1, "transport": 2, "place": 3, "retreat": 4}
# Table box (build_g1_scene): centre (0.42, -0.05), half-size (0.20, 0.35). Keep objects 3 cm inside.
TABLE_X, TABLE_Y = (0.22 + 0.08, 0.62 - 0.05), (-0.40 + 0.05, 0.30 - 0.05)


def run(clip, S, shift=None, noise_seed=None):
    import warnings; warnings.filterwarnings("ignore")
    import mujoco
    import retarget
    import test_g1_grasp as T
    from ik import reset_ready
    from gen_demos import label_phases
    P, why = retarget.plan(clip, S, shift)
    if P is None:
        return clip, dict(ok=False, why=why)
    for name, xy in (("cup", P["cup0"]), ("coaster", P["coaster"])):
        if not (TABLE_X[0] < xy[0] < TABLE_X[1] and TABLE_Y[0] < xy[1] < TABLE_Y[1]):
            return clip, dict(ok=False, why=f"{name} off the robot's table at {np.round(xy[:2], 2)}")
    m, d, ik, gc = T.m, T.d, T.ik, T.gc
    coaster = P["coaster"]
    m.geom_pos[m.geom("coaster").id][:2] = coaster[:2]
    m.site_pos[m.site("coaster_center").id][:2] = coaster[:2]
    reset_ready(m, d, T.STAND, ik)
    cup_adr = m.joint("cup_free").qposadr[0]
    d.qpos[cup_adr:cup_adr + 7] = [*P["cup0"], 1, 0, 0, 0]
    mujoco.mj_forward(m, d)

    # expand the plan into one palm target per 20 ms tick
    ticks, cur = [], d.site("palm_center").xpos.copy()
    for seg in P["segs"]:
        if seg[0] == "poly":                                # continuous multi-waypoint motion
            _, pts, secs, closed, phases = seg
            path, idx = P["poly"]([cur if q is None else q for q in pts], secs)
            ticks += [(phases[i], x, closed) for x, i in zip(path, idx)]
        else:
            phase, a, b, secs, closed = seg
            path = a if secs is None else P["lin"](cur if a is None else a, b, secs)   # None: human carry
            ticks += [(phase, x, closed) for x in path]
        cur = path[-1]

    rec = {k: [] for k in ["arm_q", "arm_dq", "finger_q", "palm_pos", "palm_quat", "cup_pos", "cup_quat",
                           "coaster_pos", "action", "phase_script", "grasped", "contact_any"]}
    fq_adr = [m.joint(j).qposadr[0] for j in T.FINGER_JOINTS]
    arm_dof = [m.joint(j).dofadr[0] for j in T.RIGHT_ARM]
    coaster_world = np.r_[coaster[:2], m.geom_pos[m.geom("coaster").id][2]]
    track_err = []
    rng = np.random.default_rng(noise_seed) if noise_seed is not None else None
    n = np.zeros(3)
    a_ou = 0.02 / NOISE_TAU
    for phase, target, closed in ticks:
        fingers = T.FINGERS_CLOSE if closed else T.FINGERS_OPEN
        ok, f = gc.is_grasped(d)                            # observation BEFORE acting
        pq = np.zeros(4); mujoco.mju_mat2Quat(pq, d.site("palm_center").xmat)
        rec["arm_q"].append(d.qpos[ik.qadr].copy()); rec["arm_dq"].append(d.qvel[arm_dof].copy())
        rec["finger_q"].append(d.qpos[fq_adr].copy())
        rec["palm_pos"].append(d.site("palm_center").xpos.copy()); rec["palm_quat"].append(pq)
        rec["cup_pos"].append(d.qpos[cup_adr:cup_adr + 3].copy())
        rec["cup_quat"].append(d.qpos[cup_adr + 3:cup_adr + 7].copy())
        rec["coaster_pos"].append(coaster_world.copy())
        rec["grasped"].append(ok); rec["contact_any"].append(sum(f.values()) > 0.1)
        rec["phase_script"].append(PHASE_ID[phase])
        finger_targets = np.array([fingers[j] for j in T.FINGER_JOINTS])
        rec["action"].append(np.concatenate([target, T.HAND_QUAT, finger_targets]))
        exec_target = target
        if rng is not None:
            n = n * (1 - a_ou) + NOISE_SIGMA * np.sqrt(2 * a_ou) * rng.standard_normal(3)
            if phase not in ("grasp", "place"):
                exec_target = target + n                     # label (recorded above) stays clean
        arm_q, _, _ = ik.solve(d.qpos, exec_target, T.HAND_QUAT, iters=30)
        d.ctrl[T.ARM_ACT] = arm_q
        for j, q in fingers.items():
            d.ctrl[m.actuator(j).id] = q
        mujoco.mj_step(m, d, nstep=T.SUBSTEPS)
        track_err.append(np.linalg.norm(d.site("palm_center").xpos - target))
    ep = {k: np.array(v) for k, v in rec.items()}
    final = d.qpos[cup_adr:cup_adr + 3]
    upright = d.body("cup").xmat.reshape(3, 3)[2, 2] > 0.95
    dist = float(np.linalg.norm(final[:2] - coaster[:2]))
    ep["success"] = np.array(bool(dist < 0.03 and upright))
    ep["phase_auto"] = label_phases(ep)
    ep["clip"] = np.array(clip); ep["S"] = np.array(S)
    out_dir, name = (OUT, clip) if noise_seed is None else (DART_OUT, f"{clip}_n{noise_seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{name}.npz", **ep)
    clip = name
    held = bool(ep["grasped"][ep["phase_script"] == 2].mean() > 0.8) if (ep["phase_script"] == 2).any() else False
    why = "" if ep["success"] else (f"dropped during carry" if not held else
                                     f"tube {dist*100:.1f} cm from coaster, upright={upright}")
    return clip, dict(ok=bool(ep["success"]), why=why, dist=dist, held=held, ticks=len(ticks),
                      track_p95=float(np.percentile(track_err, 95)), cup0=P["cup0"][:2].round(3).tolist(),
                      coaster=coaster[:2].round(3).tolist())


def _geom(clip):
    import warnings; warnings.filterwarnings("ignore")
    import retarget
    g, _ = retarget.human_geometry(clip)
    return clip, g


def _worker(args):
    try:
        return run(*args)
    except Exception as e:
        return args[0], dict(ok=False, why=f"ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--S", type=float, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--noise", type=int, default=0, help="K noise-injected replays per SUCCESSFUL clip (DART)")
    a = ap.parse_args()
    import retarget
    S = a.S if a.S is not None else retarget.S_DEFAULT
    clips = a.clips or sorted(p.stem for p in (ROOT / "data" / "tracks").glob("*.npz"))
    import json
    if a.noise:
        ok_clips = a.clips or sorted(p.stem for p in OUT.glob("*.npz") if bool(np.load(p)["success"]))
        shift = np.array(json.loads(retarget.SHIFT_FILE.read_text())[str(S)])
        jobs = [(c, S, shift, k) for c in ok_clips for k in range(1, a.noise + 1)]
        with mp.get_context("spawn").Pool(a.workers) as pool:
            res = dict(pool.map(_worker, jobs))
        n_ok = sum(r["ok"] for r in res.values())
        fails = [c for c, r in res.items() if not r["ok"]]
        print(f"DART: {n_ok}/{len(res)} noisy replays still succeed (kept for training); failed: {fails}")
        raise SystemExit
    with mp.get_context("spawn").Pool(a.workers) as pool:
        # The shared workspace shift uses ALL tracked clips, so it doesn't depend on which clips are
        # replayed in this run. Geometry is cached per clip (data/human_geom).
        allc = sorted(p.stem for p in (ROOT / "data" / "tracks").glob("*.npz"))
        G = {c: g for c, g in pool.map(_geom, allc) if g is not None}
        reach = retarget.Reach()
        pts = [x for g in G.values() for x in retarget.mapped(g, S)[:2]]
        shift = retarget.workspace_shift(pts, reach.center)
        old = json.loads(retarget.SHIFT_FILE.read_text()) if retarget.SHIFT_FILE.exists() else {}
        old[str(S)] = shift.tolist(); retarget.SHIFT_FILE.write_text(json.dumps(old, indent=1))
        fit = sum(reach(retarget.mapped(g, S, shift)[0]) and reach(retarget.mapped(g, S, shift)[1]) for g in G.values())
        print(f"S={S}: shift {shift.round(3).tolist()} m; {fit}/{len(G)} clips inside the measured workspace")
        res = dict(pool.map(_worker, [(c, S, shift) for c in clips]))
    for c, r in sorted(res.items()):
        extra = f" cup0={r['cup0']} coaster={r['coaster']} track_p95={r['track_p95']*1000:.0f}mm" if "cup0" in r else ""
        print(f"{c}: {'SUCCESS' if r['ok'] else 'fail: ' + r['why']}{extra}")
    n = sum(r["ok"] for r in res.values())
    print(f"\nS={S}: success {n}/{len(res)}")
