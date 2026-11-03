"""Recompute phase_auto (gen_demos.label_phases) in already-replayed episode files, without re-running physics.

WHY: the labeller's grasp-start and place-end rules didn't fit the Inspire's top-down grasp (3 Nov): auto
labels agreed with the planned phases only 61% and marked 47% of ticks as teacher phases (Dex3: 84%, 25%).
The physics replay itself was fine, so only the labels are rewritten. Uses paths.DATA (per hand).
Run:  python relabel.py   (rewrites DATA/replayed/*.npz and DATA/dart/*.npz in place)
"""
import numpy as np
from paths import DATA
from gen_demos import label_phases

if __name__ == "__main__":
    for sub in ["replayed", "dart"]:
        n = 0
        for p in sorted((DATA / sub).glob("*.npz")):
            e = dict(np.load(p, allow_pickle=True))
            e["phase_auto"] = label_phases(e)
            np.savez_compressed(p, **e); n += 1
        print(f"relabelled {n} files in {DATA / sub}")
