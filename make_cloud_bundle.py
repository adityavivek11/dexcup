"""Pack what cloud training needs into dexcup_cloud.zip: code + the successful replayed episodes.

WHY so little: training needs only PyTorch, numpy and the episodes, not MuJoCo or the robot model.
The trained model comes back to the laptop, where the sim evaluation runs (CPU either way).
Run: python make_cloud_bundle.py   ->  dexcup_cloud.zip  (upload this + cloud_train.ipynb)
"""
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
CODE = ["policy.py", "train_teacher.py"]

if __name__ == "__main__":
    eps = sorted(p for p in (ROOT / "data" / "replayed").glob("*.npz") if bool(np.load(p)["success"]))
    # noise-injected replays (DART); policy.split(dart=True) keeps only those of TRAINING clips
    dart = sorted(p for p in (ROOT / "data" / "dart").glob("*.npz") if bool(np.load(p)["success"]))
    out = ROOT / "dexcup_cloud.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in CODE:
            z.write(ROOT / f, f"dexcup/{f}")
        # frozen validation clips: without this file the cloud run would fall back to "every 5th"
        z.write(ROOT / "data" / "split.json", "dexcup/data/split.json")
        for p in eps:
            z.write(p, f"dexcup/data/replayed/{p.name}")
        for p in dart:
            z.write(p, f"dexcup/data/dart/{p.name}")
    print(f"wrote {out.name}: {len(CODE)} code files + {len(eps)} episodes + {len(dart)} DART episodes, "
          f"{out.stat().st_size/1e6:.1f} MB")
