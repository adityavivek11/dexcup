"""Stage 4: turn one tracked human demo into a robot episode plan (palm targets + grip, at 50 Hz).

WHY object-centric retargeting (not "put the robot palm where the human wrist was")?
  The human hand and the G1's 3-finger hand have different shapes. The G1 only holds the 9.4 cm tube if
  its palm lands within ~+-5 mm of one specific offset (swept in test_g1_grasp.py). Copying the human
  wrist pose would put the palm where a HUMAN hand grips, i.e. usually a miss. So we take from the demo
  what describes the TASK, and let the robot use its own proven grasp:
    - where the tube started          (recovered by "running the carry backwards", see cup_path)
    - where the coaster is            (per clip, find_coaster.py)
    - the path the tube travelled     (wrist motion between grasp and place, tube rigidly attached)
    - the human's timing              (approach / carry / retreat durations)
  Approach and retreat use the safe scripted motions (descend beside the tube, slide in), retimed to the
  human durations: the human approach ends where the human hand grips, which knocks the tube for the robot.

WHY scale horizontal positions?
  Human tube-coaster distances are 24-75 cm; the G1's arm is shorter. Horizontal positions are scaled by S
  about the head (egocentric: what was 30 cm in front of the person ends up S*30 cm in front of the robot),
  then one shared shift centres all clips on the MEASURED robot workspace (reach_map.py). Clips whose tube
  or coaster still land outside it are skipped. Heights are NOT scaled: clearance is a physical distance.

Frames: track poses are in the coaster-tag frame (z = table normal). The human's facing direction is the
camera's optical axis projected on the table; it is rotated onto the robot's +x.
"""
from pathlib import Path
import json
import numpy as np
import cv2
from scipy.signal import savgol_filter

ROOT = Path(__file__).parent
CTRL_DT = 0.02
S_DEFAULT = 0.3                 # horizontal scale: 48/55 clips fit the measured workspace (0.35: 45, 0.5: 22;
                                # 0.25 fits all but the tube nearly touches the roll). See CLAUDE.md.
GRASP_PAUSE, RELEASE_PAUSE = 0.6, 0.5      # robot fingers need time to close/open (humans do it moving)
MIN_PHASE = 0.8                 # seconds: floor for retimed scripted segments (servo/IK can't go faster)
TIME_SCALE = 1.0                # >1 slows the human carry down (robot slower than a human)
ROBOT_HEAD_XY = np.array([0.096, 0.0])          # build_g1_scene prints head=[0.096 0 1.237]
SHIFT_FILE = ROOT / "data" / "retarget_shift.json"
GEO_DIR = ROOT / "data" / "human_geom"


# ---------------------------------------------------------------- human-side processing
def fill_and_smooth(t, T):
    """Fill tracking gaps and smooth. Positions: linear interpolation then Savitzky-Golay (keeps the timing
    of stops better than a moving average). Rotations: nearest valid frame, then the same filter on the
    matrix entries, re-orthonormalised (fine for the small per-frame changes measured: median 0.8 deg)."""
    v = ~np.isnan(T[:, 0, 0])
    vi = np.flatnonzero(v)
    p = np.stack([np.interp(t, t[v], T[v, k, 3]) for k in range(3)], 1)
    R = T[vi[np.abs(np.arange(len(t))[:, None] - vi[None]).argmin(1)], :3, :3]
    p = savgol_filter(p, 15, 2, axis=0)
    Rs = savgol_filter(R.reshape(len(t), 9), 15, 2, axis=0).reshape(-1, 3, 3)
    U, _, Vt = np.linalg.svd(Rs)
    return p, U @ Vt


def segment(t, p, coaster_xy):
    """Grasp and place = the two moments the hand nearly stops: place where it is near the coaster,
    grasp before that and well away from it. Indices into t."""
    sp = np.linalg.norm(np.gradient(p, t, axis=0), axis=1)
    dxy = np.linalg.norm(p[:, :2] - coaster_xy, axis=1)
    mid = (t > 0.15 * t[-1]) & (t < 0.95 * t[-1])
    near = mid & (dxy < dxy[mid].min() + 0.05)
    ip = np.flatnonzero(near)[np.argmin(sp[near])]
    cand = np.flatnonzero(mid & (t < t[ip] - 0.8) & (dxy > dxy[ip] + 0.08))
    if len(cand) == 0:
        return None
    return cand[np.argmin(sp[cand])], ip


def cup_path(p, R, ig, ip, cup_place):
    """Tube centre over [ig, ip], assuming it moves rigidly with the wrist between grasp and place.
    Anchored at the place (tube on the roll, centred), then carried backwards: at ig this gives the
    start position. Rotation is used, not just translation: the wrist really turns ~55 deg (median)
    while carrying, and per-frame tag rotation noise is only ~0.8 deg."""
    off_wrist = R[ip].T @ (cup_place - p[ip])          # tube centre in the wrist frame at place
    return np.einsum("nij,j->ni", R[ig:ip + 1], off_wrist) + p[ig:ip + 1]


def facing(clip):
    """Human head position and facing direction (unit xy), in the coaster-tag frame, from early frames."""
    from tags import detect, pose, COASTER_ID
    from calibrate import load
    cap = cv2.VideoCapture(str(ROOT / "recordings" / f"{clip}.MOV"))
    K, dist, _ = load(None, (int(cap.get(3)), int(cap.get(4))))
    pos, fwd = [], []
    for i in range(0, 45, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            break
        tg = detect(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        if COASTER_ID in tg:
            Tc = np.linalg.inv(pose(tg[COASTER_ID], COASTER_ID, K, dist))   # camera in tag frame
            pos.append(Tc[:3, 3])
            fwd.append(Tc[:2, 2] / np.linalg.norm(Tc[:2, 2]))               # optical axis on the table
    if not pos:
        return None
    f = np.mean(fwd, 0)
    return np.median(pos, 0), f / np.linalg.norm(f)


def human_geometry(clip):
    """Everything taken from the human demo, in the coaster-tag frame. Cached (it needs a video read).
    Returns (dict, None) or (None, reason)."""
    from build_g1_scene import CUP_HALF_H, COASTER_HALF_H
    f = GEO_DIR / f"{clip}.npz"
    if f.exists():
        g = dict(np.load(f))
        return (g, None) if "why" not in g else (None, str(g["why"]))
    GEO_DIR.mkdir(parents=True, exist_ok=True)

    def fail(why):
        np.savez(f, why=why)
        return None, why
    coast = json.loads((ROOT / "data" / "coaster.json").read_text())
    if clip not in coast:
        return fail("no coaster")
    c_xy = np.array(coast[clip]["xy_mm"]) / 1000
    d = np.load(ROOT / "data" / "tracks" / f"{clip}.npz")
    t = d["t"]
    p, R = fill_and_smooth(t, d["T_cw"])
    seg = segment(t, p, c_xy)
    if seg is None:
        return fail("no grasp/place segmentation")
    ig, ip = seg
    C = cup_path(p, R, ig, ip, np.r_[c_xy, 2 * COASTER_HALF_H + CUP_HALF_H])
    z_err = C[0, 2] - CUP_HALF_H                                  # start should be standing on the table
    if abs(z_err) > 0.04:
        return fail(f"tube start height off by {z_err*100:+.1f} cm (bad rotation or segmentation)")
    hf = facing(clip)
    if hf is None:
        return fail("no head pose")
    g = dict(t=t, ig=ig, ip=ip, C=C, c_xy=c_xy, head=hf[0], fwd=hf[1], z_err=z_err)
    np.savez(f, **g)
    return g, None


# ---------------------------------------------------------------- mapping to the robot
def mapped(g, S, shift=np.zeros(2)):
    """(cup0_xy, coaster_xy, tube path Nx3) in robot world, for scale S and workspace shift."""
    from build_g1_scene import TABLE_TOP
    c, s_ = g["fwd"]
    Rz = np.array([[c, s_], [-s_, c]])                          # rotates the facing direction onto +x
    def f(P):
        P = np.atleast_2d(P)
        xy = ROBOT_HEAD_XY + shift + S * (P[:, :2] - g["head"][:2]) @ Rz.T
        return np.column_stack([xy, TABLE_TOP + P[:, 2]])
    Cr = f(g["C"])
    return Cr[0, :2], f(np.r_[g["c_xy"], 0])[0, :2], Cr


def workspace_shift(points, center):
    """One constant xy shift for ALL clips, putting the median of all tube-start and coaster positions at
    the centre of the measured robot workspace.
    WHY: head-relative positions scaled by S put objects too close to the robot's body (the person sat
    closer to the table than suits the G1's arm). A single shared shift keeps every clip's head-relative
    variation; it just moves the robot back from the table."""
    return center - np.median(np.array(points), axis=0)


class Reach:
    """Measured grasp-and-lift feasibility (data/reach_map.npz from reach_map.py), nearest grid cell."""
    def __init__(self):
        r = np.load(ROOT / "data" / "reach_map.npz")
        self.xs, self.ys, self.ok = r["xs"], r["ys"], r["ok"]
        cells = np.array([(x, y) for i, x in enumerate(self.xs) for j, y in enumerate(self.ys) if self.ok[i, j]])
        self.center = np.median(cells, axis=0)

    def __call__(self, xy):
        if not (self.xs[0] - 0.015 <= xy[0] <= self.xs[-1] + 0.015 and
                self.ys[0] - 0.02 <= xy[1] <= self.ys[-1] + 0.02):
            return False
        i, j = np.abs(self.xs - xy[0]).argmin(), np.abs(self.ys - xy[1]).argmin()
        # A failed cell still counts if >= 3 of its 4 neighbours work: isolated failures inside the region
        # had small palm errors (grasp-tolerance noise), unlike reach-limit failures at the edges.
        nb = [(i + a, j + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        good = sum(self.ok[a, b] for a, b in nb if 0 <= a < len(self.xs) and 0 <= b < len(self.ys))
        return bool(self.ok[i, j] or good >= 3)


def lin(a, b, secs):
    """Smoothstep interpolation from a to b over secs at 50 Hz (no velocity jumps at segment ends)."""
    n = max(1, int(round(secs / CTRL_DT)))
    u = np.linspace(0, 1, n)[:, None]
    u = u * u * (3 - 2 * u)
    return a + u * (b - a)


def poly(points, secs):
    """Move along a polyline with ONE ease-in/ease-out over its whole arc length: speed is zero only at the
    first and last point, never at the corners. Returns (positions (n,3), segment index per tick).
    WHY (teacher v2 stalled at waypoints): the first version chained segments that each eased to a FULL
    STOP. At such a stop the demo's next action is ~0 ("start moving very slowly"), so a state-based policy,
    which sees no velocity or clock, has a fixed point there and hovers forever (val 3: stuck 9.6 cm above
    the tube). Same path corners as before, so the same tube clearance."""
    P = np.asarray(points, float)
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    n = max(2, int(round(secs / CTRL_DT)))
    u = np.linspace(0, 1, n)
    s = (u * u * (3 - 2 * u)) * L[-1]                     # smoothstep on arc length
    idx = np.clip(np.searchsorted(L, s, side="right") - 1, 0, len(P) - 2)
    frac = np.where(L[idx + 1] > L[idx], (s - L[idx]) / np.maximum(L[idx + 1] - L[idx], 1e-9), 0)
    return P[idx] + frac[:, None] * (P[idx + 1] - P[idx]), idx


def plan(clip, S=S_DEFAULT, shift=None):
    """Robot episode plan for one clip, or (None, reason). shift: see workspace_shift (loaded if None)."""
    if shift is None:
        shift = np.array(json.loads(SHIFT_FILE.read_text())[str(S)])
    from build_g1_scene import CUP_HALF_H, COASTER_HALF_H, TABLE_TOP
    import test_g1_grasp as G
    g, why = human_geometry(clip)
    if g is None:
        return None, why
    t, ig, ip = g["t"], int(g["ig"]), int(g["ip"])
    _, coaster, Cr = mapped(g, S, shift)
    reach = Reach()
    if not reach(Cr[0, :2]):
        return None, f"tube start outside robot workspace {np.round(Cr[0, :2], 2)}"
    if not reach(coaster):
        return None, f"coaster outside robot workspace {np.round(coaster, 2)}"
    table_z = TABLE_TOP + CUP_HALF_H
    place_z = TABLE_TOP + 2 * COASTER_HALF_H + CUP_HALF_H
    cup0 = np.r_[Cr[0, :2], table_z]
    target = np.r_[coaster, place_z]
    # Pin both ends exactly: the reconstructed start can be up to 4 cm off in height, so add an offset that
    # is (start error) at the grasp and fades to 0 by the place -> no jump at either end.
    u = np.linspace(0, 1, len(Cr))[:, None]
    Cr = Cr + (1 - u) * (cup0 - Cr[0]) + u * (target - Cr[-1])
    # Clearance: tracking noise must not drag the tube through the roll or the table. Away from the start
    # the tube bottom stays >= 2 cm above the table; over the roll it stays above the roll top, with 1.5 cm
    # margin until it is within 3 cm of the centre (then it lowers straight down onto the roll).
    d_end = np.linalg.norm(Cr[:, :2] - target[:2], axis=1)
    d_start = np.linalg.norm(Cr[:, :2] - cup0[:2], axis=1)
    zmin = np.where(d_start > 0.02, table_z + 0.02, table_z)
    over_roll = d_end < 0.092                                  # tube radius + roll radius
    zmin = np.where(over_roll, place_z + np.where(d_end > 0.03, 0.015, 0.0), zmin)
    Cr[:, 2] = np.maximum(Cr[:, 2], zmin)
    if len(Cr) > 15:
        Cr[1:-1, 2] = np.maximum(savgol_filter(Cr[:, 2], 15, 2)[1:-1], zmin[1:-1])
    Cr[0], Cr[-1] = cup0, target
    # ---- timeline at 50 Hz -------------------------------------------------------------------------
    t_appr = max(MIN_PHASE * 3, t[ig])                 # human approach duration (scripted path retimed)
    t_ret = max(MIN_PHASE * 2, t[-1] - t[ip])
    off, clear = G.OFF, G.CLEAR
    above = cup0 + off + clear + [0, 0, 0.12]
    # Segments: (phase, start | path, end, secs, fingers closed?), or ("poly", waypoints, secs, closed,
    # phase per polyline segment) for continuous motions (None waypoint = the palm wherever it is).
    segs = [
        # approach: ready -> above-beside -> down-beside -> slide in, one continuous motion (see poly)
        ("poly", [None, above, cup0 + off + clear, cup0 + off], t_appr, False, ["approach", "approach", "grasp"]),
        ("grasp", cup0 + off, cup0 + off, GRASP_PAUSE, True),   # still palm, but the fingers visibly close
    ]
    # carry: the human tube path, resampled to 50 Hz, palm = tube + robot grasp offset
    tc = (t[ig:ip + 1] - t[ig]) * TIME_SCALE
    tt = np.arange(0, tc[-1], CTRL_DT)
    carry = np.stack([np.interp(tt, tc, Cr[:, k]) for k in range(3)], 1) + off
    place = Cr[-1] + off
    segs += [("transport", carry, None, None, True),
             ("place", place, place, RELEASE_PAUSE, False),          # still palm, fingers visibly open
             ("poly", [place, place + clear, place + clear + [0, 0, 0.12]], t_ret, False, ["retreat", "retreat"])]
    return dict(clip=clip, S=S, cup0=cup0, coaster=np.r_[coaster, 0.0], segs=segs, lin=lin, poly=poly,
                t_grasp=float(t[ig]), t_place=float(t[ip]), z_err=float(g["z_err"]),
                carry_len=float(np.sum(np.linalg.norm(np.diff(Cr[:, :2], axis=0), axis=1)))), None
