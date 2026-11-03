"""Where each experiment reads and writes, so a second hand can't overwrite the first one's results.

WHY: the 3-finger (Dex3) results on main are the publishable fallback. The 5-finger (Inspire) run repeats
the same pipeline, so every per-robot OUTPUT gets its own folder. Inputs that don't depend on the robot
(recordings, tag tracks, coaster positions, cached human geometry) stay shared in data/.

  DEXCUP_HAND=inspire (default on this branch)  -> data/inspire/, models/inspire/, results/inspire/
  DEXCUP_HAND=dex3                              -> data/, models/, results/  (the original locations)
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent
HAND = os.environ.get("DEXCUP_HAND", "inspire")
assert HAND in ("dex3", "inspire"), HAND

SHARED = ROOT / "data"                                   # tracks/, coaster.json, human_geom/
if HAND == "dex3":
    DATA, MODELS, RESULTS = ROOT / "data", ROOT / "models", ROOT / "results"
else:
    DATA, MODELS, RESULTS = ROOT / "data" / HAND, ROOT / "models" / HAND, ROOT / "results" / HAND
