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

WHY also use the checkerboard as a second anchor?
  On the first 1080p demos the arm and tube hid the coaster tag for 1.6-2.5 s during the PLACE, the
  phase that matters most, too long to interpolate. The checkerboard lies flat and static on the table
  far from the arm. From frames where both are seen we estimate the board's pose in the coaster-tag frame
  (per clip, so the board may move between clips but not during one); when the tag is hidden, the
  board's pose gives the camera pose instead. Only remaining short gaps are interpolated.

Output: data/tracks/<clip>.npz with
  t (N,) seconds | T_cw (N,4,4) wrist tag in coaster-tag frame (NaN where unknown)
  wrist_seen, coaster_seen (N,) bool | coaster_from_board (N,) bool | coaster_interp (N,) bool
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
BOARD_PATTERN, BOARD_SQUARE = (9, 6), 0.024                  # printed 25 mm x 0.96 printer scale
_BOARD_OBJ = np.zeros((54, 3), np.float64)
_BOARD_OBJ[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * BOARD_SQUARE


def board_pose(gray, K, dist):
    """T_cam_board or None. Detected at half resolution (50 vs 114 ms/frame, ~same hit rate), corners
    then refined on the full image so accuracy isn't lost."""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5)
    ok, c = cv2.findChessboardCornersSB(small, BOARD_PATTERN, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    c = cv2.cornerSubPix(gray, (c * 2).astype(np.float32), (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    ok, rv, tv = cv2.solvePnP(_BOARD_OBJ, c.reshape(-1, 2).astype(np.float64), K, dist)
    if not ok:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rv)[0]
    T[:3, 3] = tv.ravel()
    return T


def mean_T(Ts):
    """Robust average of rigid transforms: median translation, rotation = SVD projection of the mean
    rotation matrix (the 'chordal mean'; fine for tightly clustered rotations like these)."""
    U, _, Vt = np.linalg.svd(np.mean([T[:3, :3] for T in Ts], axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.median([T_[:3, 3] for T_ in Ts], axis=0)
    return T


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
    cap = cv2.VideoCapture(str(path))
    # load() scales K if the clip's resolution differs from the calibration (same aspect ratio only)
    K, dist, _ = load(calib, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    T_cc, T_cw_cam, T_cb, ts, err = [], [], [], [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        found = detect(gray)
        T_cb.append(board_pose(gray, K, dist))
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
    # Second anchor: board pose in the coaster-tag frame, from frames where both are seen
    both = [np.linalg.inv(Tc) @ Tb for Tc, Tb in zip(T_cc, T_cb) if Tc is not None and Tb is not None]
    from_board = np.zeros(len(t), bool)
    board_spread_mm = np.nan
    if len(both) >= 5:
        T_tb = mean_T(both)
        # Consistency: how much the per-frame estimates scatter. Large = board moved or bad detections.
        board_spread_mm = float(np.median([np.linalg.norm(T[:3, 3] - T_tb[:3, 3]) for T in both]) * 1000)
        T_bt = np.linalg.inv(T_tb)
        for i in range(len(t)):
            if T_cc[i] is None and T_cb[i] is not None:
                T_cc[i] = T_cb[i] @ T_bt
                from_board[i] = True
    T_cc_f, interp = fill_coaster(T_cc, t)
    T_cw = np.full((len(t), 4, 4), np.nan)
    for i, (Tc, Tw) in enumerate(zip(T_cc_f, T_cw_cam)):
        if Tc is not None and Tw is not None:
            T_cw[i] = np.linalg.inv(Tc) @ Tw          # wrist tag expressed in the coaster-tag frame
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / (path.stem + ".npz")
    np.savez(out, t=t, T_cw=T_cw, wrist_seen=wrist_seen, coaster_seen=coaster_seen,
             coaster_from_board=from_board, coaster_interp=interp, reproj_px=np.array(err),
             board_spread_mm=board_spread_mm)
    valid = ~np.isnan(T_cw[:, 0, 0])
    print(f"{path.name}: {len(t)} frames, coaster seen {coaster_seen.mean():.0%} (+{from_board.mean():.0%} via board, "
          f"spread {board_spread_mm:.1f} mm; +{interp.mean():.0%} interp), "
          f"wrist seen {wrist_seen.mean():.0%}, wrist-in-coaster valid {valid.mean():.0%}, "
          f"median reproj {np.nanmedian(err, axis=0).round(2).tolist()} px")
    return out


def _safe_track(args):
    """Returns None on success, or 'clip: error' so the batch reports it instead of dying."""
    try:
        track(*args)
        return None
    except Exception as e:
        print(f"{Path(args[0]).name}: ERROR {type(e).__name__}: {e}", flush=True)
        return f"{Path(args[0]).name}: {e}"


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
        # spawn: same behaviour on Windows (laptop) and Linux (cloud). imap_unordered + per-clip error
        # catching: with starmap, one failing clip aborted the pool and silently dropped 3 other clips.
        with mp.get_context("spawn").Pool(a.workers) as pool:
            failed = [r for r in pool.imap_unordered(_safe_track, [(f, a.calib) for f in files]) if r]
    else:
        failed = [r for r in map(_safe_track, [(f, a.calib) for f in files]) if r]
    print(f"\n{len(files) - len(failed)}/{len(files)} clips tracked" + (f"; FAILED: {failed}" if failed else ""))
