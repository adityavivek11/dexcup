"""Quick quality gate for recorded clips: run on every batch BEFORE spending time on tracking.

WHY: a bad clip (tags not detected, no still pause, low resolution) costs seconds to re-record now and
hours to debug later. Checks per clip:
  - resolution/fps: 4K needed so the tags are big enough in the image
  - coaster tag (ID 1) and wrist tag (ID 0) detection rate, and the longest wrist-tag gap
    (short gaps get filled by smoothing; long ones mean the wrist pose is unknown mid-grasp)
  - still time at start/end: the pauses mark where a demo begins and ends

Run:  python check_clips.py recordings/            (or a list of files)
      python check_clips.py recordings/ --step 1   (every frame; default checks every 3rd for speed)
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from tags import detect, COASTER_ID, WRIST_ID

MIN_HEIGHT = 2160            # 4K (shorter side, after rotation)
MIN_WRIST, MIN_COASTER = 0.80, 0.50   # coaster gets occluded by the arm during place; it's static, so fine
MAX_WRIST_GAP_S = 0.5
MIN_STILL_S = 0.7            # asked for 1 s; allow some slack
STILL_THR = 1.5              # mean abs grey-level change per frame on a 160x90 thumbnail


def still_runs(diffs: np.ndarray, fps: float) -> tuple[float, float]:
    """Seconds of stillness at the start and at the end (consecutive low-motion frames)."""
    moving = diffs > STILL_THR
    start = np.argmax(moving) if moving.any() else len(diffs)
    end = np.argmax(moving[::-1]) if moving.any() else len(diffs)
    return start / fps, end / fps


def check(path: Path, step: int) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seen = {COASTER_ID: [], WRIST_ID: []}
    diffs, prev, shape = [], None, None
    for i in range(n):
        ok, fr = cap.read()                      # read every frame (needed for motion), detect on every `step`
        if not ok:
            break
        shape = fr.shape
        thumb = cv2.cvtColor(cv2.resize(fr, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diffs.append(float(np.mean(cv2.absdiff(thumb, prev))))
        prev = thumb
        if i % step == 0:
            found = detect(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
            for k in seen:
                seen[k].append(k in found)
    w = np.array(seen[WRIST_ID], bool)
    gap, run = 0, 0
    for v in w:                                  # longest run of consecutive misses
        run = 0 if v else run + 1
        gap = max(gap, run)
    s0, s1 = still_runs(np.array(diffs), fps)
    r = dict(clip=path.name, res=f"{shape[1]}x{shape[0]}", fps=fps, dur=n / fps,
             coaster=float(np.mean(seen[COASTER_ID])), wrist=float(w.mean()),
             gap=gap * step / fps, still0=s0, still1=s1)
    problems = []
    if min(shape[:2]) < MIN_HEIGHT: problems.append("not 4K")
    if r["coaster"] < MIN_COASTER: problems.append("coaster tag rarely seen")
    if r["wrist"] < MIN_WRIST: problems.append("wrist tag rarely seen")
    if r["gap"] > MAX_WRIST_GAP_S: problems.append(f"wrist gap {r['gap']:.1f}s")
    if s0 < MIN_STILL_S: problems.append("no still start")
    if s1 < MIN_STILL_S: problems.append("no still end")
    r["verdict"] = "OK" if not problems else "; ".join(problems)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--step", type=int, default=3)
    a = ap.parse_args()
    files = []
    for p in map(Path, a.paths):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in (".mov", ".mp4")) if p.is_dir() else [p]
    print(f"{'clip':14} {'res':10} {'fps':>5} {'dur':>5} {'coast':>6} {'wrist':>6} {'gap':>5} "
          f"{'still0':>6} {'still1':>6}  verdict")
    n_ok = 0
    for f in files:
        r = check(f, a.step)
        n_ok += r["verdict"] == "OK"
        print(f"{r['clip']:14} {r['res']:10} {r['fps']:5.1f} {r['dur']:5.1f} {r['coaster']:6.0%} "
              f"{r['wrist']:6.0%} {r['gap']:5.1f} {r['still0']:6.1f} {r['still1']:6.1f}  {r['verdict']}")
    print(f"\n{n_ok}/{len(files)} clips OK")
    sys.exit(0 if n_ok == len(files) else 1)
