"""AprilTag detection and pose, shared by check_clips.py and track_tags.py.

WHY OpenCV's aruco module (not pupil-apriltags)?
  It ships inside opencv-python (one less dependency on both machines) and supports tag36h11 directly.
  The first test clips also showed both detectors fail the same way on border-less tags, so the
  detector choice was never the problem; the printing was.
"""
import cv2
import numpy as np

COASTER_ID, WRIST_ID = 1, 0
# Black-square edge in metres. UPDATE with the ruler-measured values after printing (printers rescale).
TAG_SIZE = {COASTER_ID: 0.070, WRIST_ID: 0.035}

_params = cv2.aruco.DetectorParameters()
# Sub-pixel corner refinement: cheap, and corner accuracy is what sets pose accuracy.
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
_detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                    _params)


def detect(gray: np.ndarray) -> dict[int, np.ndarray]:
    """Return {tag_id: 4x2 corner array (px)} for our two tags. Unknown IDs are ignored."""
    corners, ids, _ = _detector.detectMarkers(gray)
    if ids is None:
        return {}
    out = {}
    for c, i in zip(corners, np.ravel(ids)):
        if int(i) in TAG_SIZE:
            out[int(i)] = np.reshape(c, (4, 2)).astype(np.float64)
    return out


def pose(corners: np.ndarray, tag_id: int, K: np.ndarray, dist: np.ndarray):
    """4x4 transform T_cam_tag (tag frame -> camera frame, metres), or None if PnP fails.

    WHY IPPE_SQUARE: it is the PnP solver made for exactly this case (4 coplanar corners of a square)
    and is far more stable than the generic iterative solver on small tags.
    """
    s = TAG_SIZE[tag_id] / 2
    # aruco corner order: top-left, top-right, bottom-right, bottom-left; tag frame z points out of the tag
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T
