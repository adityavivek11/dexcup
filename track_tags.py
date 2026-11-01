"""Stage 3a: per-frame wrist-tag pose expressed in the COASTER frame, for every clip.

WHY the coaster frame?
  The head (camera) moves during a demo, so camera-frame poses are meaningless on their own. The coaster
  is static and lies flat on the table, so "wrist relative to coaster" is a fixed, table-aligned frame:
  its z axis is the table normal and its origin is (tag + measured offset) the place target. The robot
  scene has a coaster too, so retargeting is just "put the palm at the same pose relative to the coaster".

WHY interpolate the coaster pose across gaps?
  The arm covers the coaster tag during the place. The coaster itself doesn't move, only the camera,
  and head motion over a fraction of a second is small and smooth, so interpolating T_cam_coaster
  between the frames where it was seen is a good estimate. Long gaps get flagged, not trusted.

Output: data/tracks/<clip>.npz with
  t (N,) seconds | T_cw (N,4,4) wrist tag in coaster-tag frame (NaN where unknown)
  wrist_seen, coaster_seen (N,) bool | coaster_interp (N,) bool (coaster pose interpolated, not seen)
  reproj_px (N,2) PnP reprojection error for [coaster, wrist] (big = bad corners or wrong tag size)
Run:  python track_tags.py recordings/ --workers 4
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from tags import detect, pose, TAG_SIZE, COASTER_ID, WRIST_ID

OUT = Path(__file__).parent / "data" / "tracks"
MAX_COASTER_GAP_S = 1.5     # longer gaps: don't trust interpolation (head may have moved a lot)
REPROJ_GATE_X = 4.0         # coaster detection rejected if reprojection error > 4x the clip median...
REPROJ_MIN_GATE_PX = 0.5    # ...and above this floor (so near-perfect clips don't reject normal noise)


def reproj_err(T, corners, tag_id, K, dist):
    s = TAG_SIZE[tag_id] / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
    p, _ = cv2.projectPoints(obj, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)
    return float(np.sqrt(np.mean(np.sum((p.reshape(4, 2) - corners) ** 2, axis=1))))


def slerp_T(Ta, Tb, a):
    """Interpolate two rigid transforms: rotation by slerp (via rotation vector), translation linearly."""
    Ra, Rb = Ta[:3, :3], Tb[:3, :3]
    rv = cv2.Rodrigues(Ra.T @ Rb)[0]
    T = np.eye(4)
    T[:3, :3] = Ra @ cv2.Rodrigues(rv * a)[0]
    T[:3, 3] = (1 - a) * Ta[:3, 3] + a * Tb[:3, 3]
    return T


def fill_coaster(T_cc: list, t: np.ndarray):
    """Fill missing camera-from-coaster poses by interpolation (holding at the clip ends)."""
    seen = np.array([T is not None for T in T_cc])
    idx = np.flatnonzero(seen)
    out, interp = list(T_cc), np.zeros(len(T_cc), bool)
    if len(idx) == 0:
        return out, interp
    for i in range(len(T_cc)):
        if seen[i]:
            continue
        before, after = idx[idx < i], idx[idx > i]
        if len(before) and len(after):
            a, b = before[-1], after[0]
            if t[b] - t[a] > MAX_COASTER_GAP_S:
                continue
            out[i] = slerp_T(T_cc[a], T_cc[b], (t[i] - t[a]) / (t[b] - t[a]))
        else:
            j = before[-1] if len(before) else after[0]
            if abs(t[i] - t[j]) > MAX_COASTER_GAP_S:
                continue
            out[i] = T_cc[j]
        interp[i] = True
    return out, interp


def track(path: Path, calib: str | None = None) -> Path:
    from calibrate import load
    K, dist, (w, h) = load(calib) if calib else load()
    cap = cv2.VideoCapture(str(path))
    T_cc, T_cw_cam, ts, err = [], [], [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if (fr.shape[1], fr.shape[0]) != (w, h):
            raise ValueError(f"{path.name}: frame {fr.shape[1]}x{fr.shape[0]} != calibration {w}x{h}. "
                             "Record in the same video mode you calibrated in.")
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        found = detect(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        row = [np.nan, np.nan]
        Ts = {}
        for k, tid in enumerate((COASTER_ID, WRIST_ID)):
            if tid in found:
                T = pose(found[tid], tid, K, dist)
                if T is not None:
                    Ts[tid] = T
                    row[k] = reproj_err(T, found[tid], tid, K, dist)
        T_cc.append(Ts.get(COASTER_ID))
        T_cw_cam.append(Ts.get(WRIST_ID))
        err.append(row)
    t = np.array(ts) - ts[0]
    err = np.array(err)
    # Reject partially occluded coaster detections. When the hand covers one corner, the detector can
    # still "find" the tag with that corner misplaced; the pose is then wrong by ~1 cm at the wrist
    # (found in the synthetic test). Such detections fit the square badly, so gate on reprojection
    # error relative to this clip's typical error and treat them as unseen (-> interpolated).
    ce = err[:, 0]
    if np.isfinite(ce).any():
        bad = ce > max(REPROJ_MIN_GATE_PX, REPROJ_GATE_X * np.nanmedian(ce))
        for i in np.flatnonzero(bad):
            T_cc[i] = None
    coaster_seen = np.array([T is not None for T in T_cc])
    wrist_seen = np.array([T is not None for T in T_cw_cam])
    T_cc_f, interp = fill_coaster(T_cc, t)
    T_cw = np.full((len(t), 4, 4), np.nan)
    for i, (Tc, Tw) in enumerate(zip(T_cc_f, T_cw_cam)):
        if Tc is not None and Tw is not None:
            T_cw[i] = np.linalg.inv(Tc) @ Tw          # wrist tag expressed in the coaster-tag frame
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / (path.stem + ".npz")
    np.savez(out, t=t, T_cw=T_cw, wrist_seen=wrist_seen, coaster_seen=coaster_seen,
             coaster_interp=interp, reproj_px=np.array(err))
    valid = ~np.isnan(T_cw[:, 0, 0])
    print(f"{path.name}: {len(t)} frames, coaster seen {coaster_seen.mean():.0%} (+{interp.mean():.0%} interp), "
          f"wrist seen {wrist_seen.mean():.0%}, wrist-in-coaster valid {valid.mean():.0%}, "
          f"median reproj {np.nanmedian(err, axis=0).round(2).tolist()} px")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--calib", default=None)
    a = ap.parse_args()
    files = []
    for p in map(Path, a.paths):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in (".mov", ".mp4")) if p.is_dir() else [p]
    if a.workers > 1 and len(files) > 1:
        import multiprocessing as mp
        # spawn: same behaviour on Windows (laptop) and Linux (cloud)
        with mp.get_context("spawn").Pool(a.workers) as pool:
            pool.starmap(track, [(f, a.calib) for f in files])
    else:
        for f in files:
            track(f, a.calib)
