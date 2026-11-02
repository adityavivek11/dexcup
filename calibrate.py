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
    ap.add_argument("--free", action="store_true", help="also fit tangential distortion and k3")
    ap.add_argument("--landscape", action="store_true",
                    help="board video was filmed portrait but demos are landscape: save the transposed camera")
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

    # Coverage check. A low RMS does NOT mean a good calibration: the first real board video had the board
    # small (~15% of image width) and always in the upper third, gave RMS 0.16 px, and a focal length off by
    # ~40%. Few, similar views let many wrong camera models fit equally well.
    span = np.array([p.max(0) - p.min(0) for p in img_pts])[:, 0] / size[0]
    cen = np.array([p.mean(0) for p in img_pts]) / np.array(size)
    print(f"board width: median {np.median(span):.0%} of image (want > 30%) | centre covers "
          f"x {cen[:,0].min():.0%}-{cen[:,0].max():.0%}, y {cen[:,1].min():.0%}-{cen[:,1].max():.0%} (want ~15-85%)")

    # Default: fx, fy and principal point free; tangential distortion and k3 fixed at 0 (iPhone video is
    # lens-corrected by the phone; freeing them changed nothing on real footage). An earlier default also
    # forced square pixels (fx = fy) and a centred principal point. That was WRONG for this setup: on the
    # 1080p board video it raised RMS 0.43 -> 0.68 px and moved fx ~9%. The free fit finds fx/fy ~1.006,
    # most likely the printer scaling the sheet slightly differently along vs across the paper (only one
    # direction was measured). --free also frees tangential distortion and k3.
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | (0 if a.free else (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3))
    K0 = np.array([[0.8 * size[0], 0, size[0] / 2], [0, 0.8 * size[0], size[1] / 2], [0, 0, 1.0]])

    def fit(pts):
        return cv2.calibrateCameraExtended([obj] * len(pts), pts, size, K0.copy(), np.zeros(5), flags=flags)

    rms, K, dist, rvecs, tvecs, sd, _, _ = fit(img_pts)
    # Per-view error: one bad (blurred) view can drag the whole fit, so drop outliers and refit once
    errs = []
    for p, r, t in zip(img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, r, t, K, dist)
        errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - p) ** 2, axis=1)))))
    keep = [i for i, e in enumerate(errs) if e < 3 * np.median(errs)]
    if len(keep) < len(img_pts):
        img_pts = [img_pts[i] for i in keep]
        rms, K, dist, _, _, sd, _, _ = fit(img_pts)
    long_f = K[0, 0] if size[0] >= size[1] else K[1, 1]
    fov = np.degrees(2 * np.arctan(max(size) / 2 / long_f))
    print(f"views used: {len(img_pts)}  RMS reprojection error: {rms:.2f} px  (good: < 1 px)")
    print(f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}  "
          f"image {size[0]}x{size[1]}  FOV across long side {fov:.0f} deg")

    dist = dist.ravel()
    if a.landscape and size[1] > size[0]:
        # iPhone records the sensor landscape and stores a rotation flag; a portrait video is the same
        # camera with image axes swapped. Cross-check: 720p portrait and 1080p portrait board videos gave
        # the same focal lengths (scaled) within ~1%.
        K = np.array([[K[1, 1], 0, K[1, 2]], [0, K[0, 0], K[0, 2]], [0, 0, 1.0]])
        dist = dist.copy(); dist[[2, 3]] = dist[[3, 2]]          # tangential p1 <-> p2 swap with the axes
        size = (size[1], size[0])
        print(f"transposed to landscape {size[0]}x{size[1]}")
    out = Path(a.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"width": size[0], "height": size[1], "K": K.tolist(),
                               "dist": dist.tolist(), "rms_px": rms, "source": Path(a.video).name,
                               "square_mm": a.square_mm, "views": len(img_pts)}, indent=2))
    print("wrote", out)


def load(path=None, frame_size=None):
    """(K, dist, (width, height)) for track_tags.py.

    If frame_size differs from the calibration but has the same aspect ratio (e.g. a 720p clip with a
    1080p calibration), K is scaled. That assumes the video modes share one field of view, which held
    on this iPhone: 720p and 1080p board videos gave the same focal length within ~1% after scaling.
    """
    c = json.loads(Path(path or Path(__file__).parent / "calib" / "camera.json").read_text())
    K, dist, size = np.array(c["K"]), np.array(c["dist"]), (c["width"], c["height"])
    if frame_size and tuple(frame_size) != size and frame_size[0] < frame_size[1] and size[0] > size[1]:
        # Portrait clip, landscape calibration: same camera with image axes swapped. (A 90 deg rotation is
        # transpose + flip; the flip moves the principal point by ~2x its offset from centre, ~12 px here:
        # negligible next to the tag noise.) IMG_8857 (portrait) gave the best wrist visibility of batch 2.
        K = np.array([[K[1, 1], 0, K[1, 2]], [0, K[0, 0], K[0, 2]], [0, 0, 1.0]])
        dist = dist.copy(); dist[[2, 3]] = dist[[3, 2]]
        size = (size[1], size[0])
    if frame_size and tuple(frame_size) != size:
        s = frame_size[0] / size[0]
        if abs(frame_size[1] / size[1] - s) > 1e-3:
            raise ValueError(f"frame {frame_size} vs calibration {size}: different aspect/orientation. "
                             "Record demos in the same orientation as the calibration (landscape).")
        K = K.copy(); K[:2] *= s
        size = tuple(frame_size)
    return K, dist, size


if __name__ == "__main__":
    main()
