"""AprilTag detection and pose, shared by check_clips.py and track_tags.py.

WHY pupil-apriltags (the reference AprilTag library), not OpenCV's aruco module?
  aruco was used first (no extra dependency). On the 1080p demos the wrist tag wraps slightly around the
  wrist; aruco found its outline but failed to decode the bits, missing clearly visible tags
  (IMG_8852: aruco 50% vs apriltag 63% of frames). apriltag was also 3x faster with quad_decimate=2
  (27 vs ~85 ms per 1080p frame). On border-less tags (first test clips) both failed equally: that was
  a printing problem, not a detector problem.
"""
import cv2
import numpy as np

COASTER_ID, WRIST_ID = 1, 0
# Black-square edge in metres. UPDATE with the ruler-measured values after printing (printers rescale).
# Ruler-checked 2 Nov: printer scaled everything by 48/50 = 0.96 (scale bar 48 mm) -> 70->67.2, 35->33.6 mm.
TAG_SIZE = {COASTER_ID: 0.0672, WRIST_ID: 0.0336}
# Coaster (tape roll) centre in the coaster-tag frame, metres. User measured 44 mm from the tag's paper edge;
# direction (+y) found by projecting candidates onto IMG_8846 frame 0. 33.6 half-square + 16.8 white margin
# + ~2.5 cut allowance + 44 = ~97 mm.
COASTER_OFFSET = np.array([0.0, 0.097, 0.0])

MIN_MARGIN = 20      # decision margin: real detections here are ~60-70; random junk is < 10
_detector = None


def _get():
    # Created lazily: the C detector object can't be pickled, and each spawn worker needs its own.
    global _detector
    if _detector is None:
        from pupil_apriltags import Detector
        # quad_decimate=2: find quads on a half-size image (faster, and slightly MORE robust on these clips);
        # refine_edges then fits the edges on the full-resolution image, so corner accuracy is kept.
        _detector = Detector(families="tag36h11", quad_decimate=2.0, refine_edges=1)
    return _detector


def detect(gray: np.ndarray) -> dict[int, np.ndarray]:
    """Return {tag_id: 4x2 corner array (px)} for our two tags, in aruco order (TL, TR, BR, BL).
    Unknown IDs and low-confidence decodes are ignored."""
    out = {}
    for r in _get().detect(gray):
        if r.tag_id in TAG_SIZE and r.decision_margin >= MIN_MARGIN:
            # pupil's corner order differs from aruco's. [1,0,3,2] was MEASURED (matches aruco to <1 px on
            # IMG_8855); the documented-looking guess [3,2,1,0] was ~125 px off, i.e. a rotated pose.
            out[r.tag_id] = r.corners[[1, 0, 3, 2]].astype(np.float64)
    return out


def pose(corners: np.ndarray, tag_id: int, K: np.ndarray, dist: np.ndarray):
    """4x4 transform T_cam_tag (tag frame -> camera frame, metres), or None if PnP fails.

    WHY IPPE_SQUARE: it is the PnP solver made for exactly this case (4 coplanar corners of a square)
    and is far more stable than the generic iterative solver on small tags.
    """
    s = TAG_SIZE[tag_id] / 2
    # corner order (see detect): top-left, top-right, bottom-right, bottom-left; tag frame z points out of the tag
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T
