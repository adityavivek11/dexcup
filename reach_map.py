"""Map where the G1 can grasp and lift the tube (scripted approach + grasp + 9 cm lift + hold).

WHY: retargeting must put the human demo's tube and coaster positions inside the region the robot can
actually work in. Guessing that region cost a round of failed replays; this measures it.
Output: data/reach_map.npz (grid x, y, success, palm tracking error). Run: python reach_map.py
"""
import itertools
import numpy as np
from pathlib import Path

XS = np.round(np.arange(0.26, 0.50, 0.03), 3)
YS = np.round(np.arange(-0.34, 0.12, 0.04), 3)


def trial(xy):
    import warnings; warnings.filterwarnings("ignore")
    import mujoco
    import test_g1_grasp as T
    from ik import reset_ready
    m, d, ik, gc = T.m, T.d, T.ik, T.gc
    reset_ready(m, d, T.STAND, ik)
    adr = m.joint("cup_free").qposadr[0]
    cup0 = np.array([xy[0], xy[1], T.CUP_START[2]])
    d.qpos[adr:adr + 7] = [*cup0, 1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    off, clr = T.OFF, T.CLEAR
    wps = [(cup0 + off + clr + [0, 0, T.UP], False, 1.2), (cup0 + off + clr, False, 1.0), (cup0 + off, False, .8),
           (cup0 + off, True, .8), (cup0 + off + [0, 0, .09], True, .8), (cup0 + off + [0, 0, .09], True, .6)]
    cur = d.site("palm_center").xpos.copy(); err = []
    for tgt, closed, secs in wps:
        start, n = cur.copy(), int(secs / T.CTRL_DT)
        for k in range(n):
            a = min(1, k / (0.7 * n)); a = a * a * (3 - 2 * a); cur = start + a * (tgt - start)
            q, _, _ = ik.solve(d.qpos, cur, T.HAND_QUAT, iters=30); d.ctrl[T.ARM_ACT] = q
            for j in T.FINGER_ACTS:
                d.ctrl[m.actuator(j).id] = (T.FINGERS_CLOSE if closed else T.FINGERS_OPEN)[j]
            mujoco.mj_step(m, d, nstep=T.SUBSTEPS)
            err.append(np.linalg.norm(d.site("palm_center").xpos - cur))
    ok, _ = gc.is_grasped(d)
    lifted = d.qpos[adr + 2] > cup0[2] + 0.06
    return xy, bool(ok and lifted), float(np.percentile(err, 95))


if __name__ == "__main__":
    import multiprocessing as mp
    grid = list(itertools.product(XS, YS))
    with mp.get_context("spawn").Pool(6) as p:
        res = p.map(trial, grid)
    S = {xy: (ok, e) for xy, ok, e in res}
    print("grasp+lift success (#), x down / y across; palm error p95 in mm where it failed")
    print("   y: " + " ".join(f"{y:+.2f}" for y in YS))
    for x in XS:
        print(f"x={x:.2f} " + " ".join(("  ##  " if S[(x, y)][0] else f"{S[(x, y)][1]*1000:5.0f} ") for y in YS))
    from paths import DATA                  # per-hand output (paths.py)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(DATA / "reach_map.npz", xs=XS, ys=YS, ok=np.array([[S[(x, y)][0] for y in YS] for x in XS]),
             err=np.array([[S[(x, y)][1] for y in YS] for x in XS]))
    print(f"\n{sum(v[0] for v in S.values())}/{len(S)} positions work")
