"""Camera calibration from a short video of the printed checkerboard (printables/checkerboard.pdf).

WHY calibrate at all?
  Turning a tag's 4 image corners into a 3D pose needs the camera's focal length, principal point and
  lens distortion (the "intrinsics" K, dist). A LiDAR app would have given these from ARKit; the stock
  Camera app doesn't, so we measure them once. They are only valid for the SAME video mode (4K, 1x lens,
  stabilization off, focus locked), so film the board in exactly the mode you record demos in.

How to film: board flat on a stiff surface; 20-30 s, move the phone slowly so the board appears at
different distances, tilts (up to ~45 deg) and in all parts of the image, including the corners.

Run:  python calibrate.py calib/board.MOV --square-mm 25.0   ->  calib/camera.json
      (--square-mm = your ruler-measured square size)
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

PATTERN = (9, 6)       # inner corners of the 10x7-square board
MAX_VIEWS = 40         # more views add little accuracy but slow calibrateCamera a lot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "calib" / "camera.json"))
    a = ap.parse_args()

    obj = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * a.square_mm / 1000.0

    cap = cv2.VideoCapture(a.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Sample ~3x more frames than we keep: some will be blurred or miss corners.
    idx = set(np.linspace(0, n - 1, min(n, MAX_VIEWS * 3)).astype(int).tolist())
    img_pts, size, last = [], None, None
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        if i not in idx:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        size = g.shape[::-1]
        # findChessboardCornersSB: the newer, more accurate and blur-tolerant detector
        found, c = cv2.findChessboardCornersSB(g, PATTERN, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            continue
        c = c.reshape(-1, 2)
        # Skip near-duplicate views (phone barely moved): they add no information
        if last is not None and np.mean(np.linalg.norm(c - last, axis=1)) < 0.02 * size[0]:
            continue
        img_pts.append(c.astype(np.float32))
        last = c
    if len(img_pts) > MAX_VIEWS:
        img_pts = [img_pts[i] for i in np.linspace(0, len(img_pts) - 1, MAX_VIEWS).astype(int)]
    print(f"usable views: {len(img_pts)}")
    if len(img_pts) < 10:
        raise SystemExit("Too few board views. Film longer, slower, with the whole board in frame.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj] * len(img_pts), img_pts, size, None, None)
    # Per-view error: one bad (blurred) view can drag the whole fit, so drop outliers and refit once
    errs = []
    for p, r, t in zip(img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, r, t, K, dist)
        errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - p) ** 2, axis=1)))))
    keep = [i for i, e in enumerate(errs) if e < 3 * np.median(errs)]
    if len(keep) < len(img_pts):
        img_pts = [img_pts[i] for i in keep]
        rms, K, dist, _, _ = cv2.calibrateCamera([obj] * len(img_pts), img_pts, size, None, None)
    print(f"views used: {len(img_pts)}  RMS reprojection error: {rms:.2f} px  (good: < 1 px)")
    print(f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}  image {size[0]}x{size[1]}")

    out = Path(a.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"width": size[0], "height": size[1], "K": K.tolist(),
                               "dist": dist.ravel().tolist(), "rms_px": rms,
                               "square_mm": a.square_mm, "views": len(img_pts)}, indent=2))
    print("wrote", out)


def load(path=Path(__file__).parent / "calib" / "camera.json"):
    """(K, dist, (width, height)) for track_tags.py."""
    c = json.loads(Path(path).read_text())
    return np.array(c["K"]), np.array(c["dist"]), (c["width"], c["height"])


if __name__ == "__main__":
    main()
