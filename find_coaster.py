"""Stage 3b: find the coaster (tape roll) centre in each clip, in the coaster-tag frame.

WHY per clip?
  The user moved the tape roll relative to the tag between clips (for more diverse data), so the fixed
  COASTER_OFFSET measured on the first setup is wrong for many clips. The place target must be measured.

HOW
  The coaster tag gives the table plane exactly. For a plane at height h above the table, the map from
  plane coordinates (mm) to image pixels is a homography H = K [r1 r2 (h*r3 + t)]. Warping the image with
  it gives a straight-down, metric view (1 px = 1 mm) in which a circle on that plane IS a circle.
  We rectify the plane of the roll's TOP RIM (h = roll height, 2 cm): the rim is the clearest edge, and
  rectifying the table plane instead would shift and stretch it by parallax (up to ~2 cm at oblique views).
  A circle of the roll's radius is then found with a Hough transform, on early frames (hand far away,
  tube not yet on the roll), and the median over frames is kept.

Output: data/coaster.json  {clip: {"xy_mm": [x, y], "n_frames": k, "spread_mm": s, "radius_mm": r}}
Run:    python find_coaster.py recordings/ [--debug]   (--debug writes overlay images to data/coaster_debug/)
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from tags import detect, pose, COASTER_ID, TAG_SIZE
from build_g1_scene import COASTER_RADIUS, COASTER_HALF_H

OUT = Path(__file__).parent / "data" / "coaster.json"
DEBUG = Path(__file__).parent / "data" / "coaster_debug"
HALF = 260                     # rectified window: +-26 cm around the tag (the roll is placed near the tag)
EARLY_S = 1.5                  # only frames from the first 1.5 s
TAG_MASK = TAG_SIZE[COASTER_ID] * 1000 / 2 * 1.6   # hide the tag + its white margin (high-contrast edges)
R_MM = COASTER_RADIUS * 1000   # 45 mm outer radius


def rectify(frame, T, K, dist, h):
    """Top-down metric view of the plane z=h (metres) in the tag frame. Pixel (u,v) <-> (x,y) mm with
    x = u - HALF, y = HALF - v (image up = tag +y)."""
    und = cv2.undistort(frame, K, dist)
    R, t = T[:3, :3], T[:3, 3]
    H_plane = K @ np.column_stack([R[:, 0] / 1000, R[:, 1] / 1000, R[:, 2] * h + t])   # mm -> pixels
    S = np.array([[1, 0, -HALF], [0, -1, HALF], [0, 0, 1.0]])                          # rect px -> mm
    return cv2.warpPerspective(und, H_plane @ S, (2 * HALF, 2 * HALF), flags=cv2.WARP_INVERSE_MAP)


def find_circle(top):
    g = cv2.GaussianBlur(cv2.cvtColor(top, cv2.COLOR_BGR2GRAY), (5, 5), 1.5)
    cv2.circle(g, (HALF, HALF), int(TAG_MASK), int(np.median(g)), -1)      # hide the tag
    # HOUGH_GRADIENT_ALT: more accurate centres; param2 = circle "perfectness" (0..1)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=60, param1=80, param2=0.6,
                         minRadius=int(R_MM * 0.85), maxRadius=int(R_MM * 1.15))
    if c is None:
        return None
    x, y, r = c[0][0]               # strongest first
    return float(x - HALF), float(HALF - y), float(r)


def process(path: Path, debug=False):
    from calibrate import load
    cap = cv2.VideoCapture(str(path))
    K, dist, _ = load(None, (int(cap.get(3)), int(cap.get(4))))
    fps = cap.get(cv2.CAP_PROP_FPS)
    hits, last = [], None
    for i in range(int(EARLY_S * fps)):
        ok, fr = cap.read()
        if not ok:
            break
        if i % 3:
            continue
        found = detect(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        if COASTER_ID not in found:
            continue
        T = pose(found[COASTER_ID], COASTER_ID, K, dist)
        top = rectify(fr, T, K, dist, 2 * COASTER_HALF_H)
        c = find_circle(top)
        if c:
            hits.append(c)
            last = (top, c)
    if not hits:
        return path.stem, None
    h = np.array(hits)
    xy = np.median(h[:, :2], axis=0)
    spread = float(np.median(np.linalg.norm(h[:, :2] - xy, axis=1)))
    if debug and last:
        DEBUG.mkdir(parents=True, exist_ok=True)
        top = last[0].copy()
        cv2.circle(top, (int(xy[0] + HALF), int(HALF - xy[1])), int(R_MM), (0, 0, 255), 2)
        cv2.drawMarker(top, (int(xy[0] + HALF), int(HALF - xy[1])), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(top, f"{path.stem} ({xy[0]:.0f},{xy[1]:.0f})mm n={len(h)} spread={spread:.1f}",
                    (5, 20), 0, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(DEBUG / f"{path.stem}.jpg"), top)
    return path.stem, {"xy_mm": xy.round(1).tolist(), "n_frames": len(h), "spread_mm": round(spread, 1),
                       "radius_mm": round(float(np.median(h[:, 2])), 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    files = []
    for p in map(Path, a.paths):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in (".mov", ".mp4")) if p.is_dir() else [p]
    files = [f for f in files if f.stem != "IMG_8849"]          # calibration video, not a demo
    import multiprocessing as mp
    from functools import partial
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = dict(pool.map(partial(process, debug=a.debug), files))
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    old.update({k: v for k, v in res.items() if v})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(old, indent=1))
    for k, v in sorted(res.items()):
        print(f"{k}: " + (f"centre {v['xy_mm']} mm, r={v['radius_mm']} mm, n={v['n_frames']}, spread {v['spread_mm']} mm"
                          if v else "NOT FOUND"))
    print(f"\nfound {sum(1 for v in res.values() if v)}/{len(res)}")
