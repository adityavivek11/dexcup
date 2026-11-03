# 🤖 dexcup: Cheaper Humanoid Manipulation by Calling a Big Policy Only When It Matters

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Simulator: MuJoCo 3.17](https://img.shields.io/badge/Simulator-MuJoCo%203.17-green.svg)](https://mujoco.org/)

> A Unitree G1 humanoid learns to pick up a tube and place it on a coaster from **our own egocentric iPhone
> recordings**. A large **teacher** policy is called only during the delicate **grasp** and **place** phases;
> a 250x smaller **student** drives the rest. The router keeps 88% of the teacher's success at **3x less
> compute per control tick**, and a random router with the same budget does far worse (76% vs 43%, p = 2.5e-7).

---

## 📖 Table of Contents
- [Overview](#-overview)
- [How this differs from the template's RL-finetuned VLA](#-how-this-differs-from-the-templates-rl-finetuned-vla)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Usage Guide](#-usage-guide)
  - [1. Pre-training (Imitation Learning)](#1-pre-training-imitation-learning)
  - [2. Student distillation and router](#2-student-distillation-and-router-in-place-of-rl-fine-tuning)
  - [3. Evaluation](#3-evaluation)
- [Configuration](#-configuration)
- [Results & Benchmarks](#-results--benchmarks)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Appendix: our journey](#-appendix-our-journey)

---

## 📖 Overview

Large manipulation policies are expensive to run at control rate, but most of a pick-and-place is easy:
moving an empty hand toward an object, carrying it, retreating. Only a few moments need precision, namely
closing the fingers on the object and letting go at the right spot.

**dexcup** tests a simple idea: run a big **teacher** policy only in those moments and a tiny **student**
everywhere else, with a small **router** deciding online, from the robot's own observation, which one to call.

- **Robot:** Unitree G1 humanoid (fixed pelvis) with the 3-finger **Dex3-1** hand, in MuJoCo.
- **Task:** pick up a cardboard tube (9.4 cm x 17.8 cm) and place it on a tape-roll "coaster".
- **Demos:** 62 usable egocentric iPhone videos of a person doing the task, tracked with two AprilTags,
  retargeted to the robot and **replayed in physics** (only demos that physically succeed are kept).
- **Headline:** the router reaches **76%** success with the teacher called on **32%** of ticks
  (teacher-only 86%), at ~3x less compute per tick. A **random router with the same teacher budget**
  reaches only 43%, so *when* the teacher is called matters, not just how often.
- **Case study:** the identical pipeline on a 5-finger **Inspire** hand (branch `inspire-hand`) shows that the
  same task became substantially harder to execute with the more complex hand (see [Results](#-results--benchmarks)).

---

## 🧭 How this differs from the template's RL-finetuned VLA

The template describes RL fine-tuning of a pre-trained Vision-Language-Action model. **We did not do that**,
and we want to be upfront about it:

| Template assumption | What we actually built | Why |
|---|---|---|
| Pre-trained VLA (OpenVLA, RT-2) | Teacher trained from scratch: ACT-style transformer (7.4M params) | Our question is the *cost* of running a big policy; a 4-day deadline and no local GPU ruled out VLA fine-tuning |
| Vision + language input | **State-based** observation (palm, tube, coaster, finger and arm joint state from the sim) | Scope cut for the deadline; image-based teacher is future work |
| RL fine-tuning (PPO/SAC) | **Imitation learning + DART + DAgger**, then a learned **router** | No reward engineering needed; DAgger fixes compounding errors, which is what RL fine-tuning is usually used for |
| Generic datasets | **Our own** recordings with a phone on the head | The project's point is learning from cheap, self-collected human demos |

---

## ✨ Key Features

* **Self-collected human demos:** stock iPhone camera (no depth, no LiDAR), two printed AprilTags, a printed
  checkerboard. All metric information comes from calibration and tag poses.
* **Physics-validated retargeting:** every human demo is replayed on the G1 in MuJoCo; only demos where the
  robot physically succeeds are used for training (41 of 62).
* **Teacher / student / router:** ACT-style chunked transformer teacher (7.4M params), 28.9k-param chunked MLP
  student, 6.5k-param phase classifier as the router.
* **Router-in-the-loop DAgger:** the student is trained on exactly the states it meets when deployed inside
  the router.
* **Honest evaluation:** 100 unseen random layouts, a **random-router control with the same teacher budget**,
  paired McNemar tests, latency mean + p95.
* **Cheap compute:** simulation and evaluation on a laptop CPU; teacher trained on a free Kaggle T4 GPU.

---

## 🏗️ System Architecture

```text
iPhone video ──► AprilTags + calibration ──► wrist & coaster poses ──► retarget ──► MuJoCo replay ──► kept demos
                                                                     (object-centric)  (physics filter)   + DART noise

At run time, every 20 ms (50 Hz):
  observation (38-d state) ──► router (phase net) ──┬─ grasp / place ──► TEACHER (7.4M, ~10 ms) ─┐
                                                    └─ everything else ─► STUDENT (29k, ~0.08 ms) ─┤
                                                                                                   ▼
                                       palm target + grip ──► arm IK ──► joint servos (500 Hz physics)
```

1. **Observation (state):** palm position, tube position and orientation, coaster position, 7 finger joints,
   7 arm joints, relative vectors, and 7 arm joint velocities (38 numbers).
2. **Teacher:** transformer over semantic observation tokens, predicting a 16-step chunk of palm targets plus
   grip; overlapping chunks are blended (ACT temporal ensembling).
3. **Student:** small MLP with the same chunked output and blending, but without velocity inputs (see Appendix).
4. **Router:** phase classifier (approach / grasp / transport / place / retreat); the teacher is called for
   grasp and place. It sees only the observation, never ground-truth labels or contacts.
5. **Control:** palm target -> damped-least-squares arm IK with a null-space posture term -> position servos.

---

## 📁 Repository Structure

```text
├── make_printables.py        # print-ready AprilTags + checkerboard (printables/*.pdf)
├── calibrate.py              # camera intrinsics from a checkerboard video -> calib/camera.json
├── check_clips.py            # quality gate for every recording batch (tag visibility, gaps)
├── tags.py, track_tags.py    # AprilTag detection and wrist/coaster pose tracking per clip
├── find_coaster.py           # finds the tape roll per clip (top-view Hough circle)
├── build_g1_scene.py         # G1 + Dex3-1, fixed pelvis, table, tube, coaster, head camera
├── ik.py, contacts.py        # arm IK; grasp check (thumb + opposing finger > 0.5 N)
├── test_g1_grasp.py          # scripted pick-and-place (physics sanity check)
├── reach_map.py              # measures where the robot can grasp and lift
├── retarget.py, replay.py    # human demo -> robot plan -> physics replay (+ DART noise)
├── gen_demos.py              # scripted demos + the physics-only phase labeller
├── policy.py                 # observation/action recipes, Teacher, Student, PhaseNet
├── train_teacher.py          # teacher training (GPU or CPU)
├── train_student.py          # DAgger and router-in-the-loop DAgger
├── router.py                 # phase net training; router and random-router agents
├── sim_env.py, eval_policy.py # closed-loop environment with per-call timing; evaluation
├── view_replay.py            # render or live-view replays
├── make_cloud_bundle.py, kaggle/, cloud/, modal_train.py, cloud_train.ipynb   # cloud training
├── CLAUDE.md                 # full lab notebook: every decision, failure and number
└── requirements.txt
```

Data (`data/`), trained models (`models/`), results (`results/`) and the raw videos are not in git (size and
privacy).

---

## 💻 Installation

Tested on Windows 11, Python 3.13, MuJoCo 3.17, PyTorch 2.14 (CPU). Linux works too (the scripts set
`MUJOCO_GL=egl` there).

**1. Clone the repository:**
```bash
git clone https://github.com/<your-username>/dexcup.git
cd dexcup
```

**2. Robot model** (sparse clone of only the G1 from MuJoCo Menagerie; the code looks in `~/mujoco_menagerie`):
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie
git -C ~/mujoco_menagerie sparse-checkout set unitree_g1
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Check the simulation:**
```bash
python test_g1_grasp.py
```
It should end with `SUCCESS=True` (scripted pick-and-place, tube ~1 cm from the coaster centre).

---

## 📊 Dataset Preparation

We record our own data; there is nothing to download.

1. **Print** `python make_printables.py` -> two AprilTags (tag36h11: ID 1 beside the coaster, ID 0 on the back
   of the right wrist) and a checkerboard. Measure the printed sizes; printers rescale.
2. **Calibrate** the phone camera with a checkerboard video in the same camera mode as the demos:
   `python calibrate.py <calibration video> --square-mm <measured size>`.
3. **Record** head-mounted, looking down: 1080p, 30 fps, 0.5x lens, HDR off, stabilisation off, AE/AF lock,
   portrait, slow movements, one demo per clip. Vary the tube and coaster positions.
4. **Check** every batch: `python check_clips.py recordings/`.
5. **Track** tags and find the coaster: `python track_tags.py recordings/ --workers 4` then
   `python find_coaster.py recordings/`.
6. **Retarget and replay** on the robot: `python reach_map.py`, then `python replay.py` (physics filter, phase
   labels) and `python replay.py --noise 4 <training clips>` (DART noise-injected copies).
   Watch any replay with `python view_replay.py <clip> --live`.

---

## 🚀 Usage Guide

### 1. Pre-training (Imitation Learning)
The teacher is trained by behaviour cloning on the replayed demos plus their DART copies. The recipe that
worked is palm-relative actions **with arm velocities** in the observation:
```bash
python train_teacher.py --dart --epochs 80 --frame palm --vel --name teacher_pv.pt
```
A T4 GPU takes ~20 s per epoch (~30 min); a laptop CPU ~7.5 min per epoch. For free GPU time we used Kaggle:
`python make_cloud_bundle.py`, upload the zip as a private dataset, and push the kernel from `kaggle/`.

### 2. Student distillation and router (in place of RL fine-tuning)
```bash
python router.py                                                   # phase net -> models/phase.pt
python train_student.py --teacher models/teacher.pt                # behaviour cloning + DAgger
python train_student.py --router --rounds 4 --rollouts 36          # DAgger inside the router
```
Router-DAgger keeps a checkpoint every round and resumes after a crash. We picked the round with the best
router success on 40 fresh layouts that are neither training nor test layouts.

### 3. Evaluation
```bash
python eval_policy.py teacher --layouts random --n 100
python eval_policy.py router  --layouts random --n 100
python eval_policy.py random  --layouts random --n 100   # same teacher fraction and segment length as the router
python eval_policy.py student --layouts random --n 100
```
Each run writes `results/<agent>_random.json` with per-episode success, ticks and per-call latency.

---

## ⚙️ Configuration

There is no config system; settings are module constants with comments explaining each value. The important ones:

| Setting | Value | Where |
|---|---|---|
| Control rate / physics rate | 50 Hz / 500 Hz | `test_g1_grasp.py` |
| Grasp offset (palm vs tube) | (-0.13, -0.0625, 0) m, swept, ~1 cm wide window | `test_g1_grasp.py` |
| Retargeting scale / workspace shift | S = 0.3 about the head; shift onto the measured reach map | `retarget.py` |
| DART noise | OU process, sigma 1 cm, tau 0.5 s, none during grasp/place | `replay.py` |
| Teacher | d=256, 4 layers, 8 heads, chunk H=16, ensembling exp(-0.1 age) | `policy.py`, `sim_env.py` |
| Teacher phases | grasp, place | `router.py` |
| Success | tube within 3 cm of the coaster centre, upright, released, for 1 s | `sim_env.py` |
| Test layouts | 100 uniform layouts, seed 0, >= 10 cm apart, inside the reach map | `eval_policy.py` |

---

## 📈 Results & Benchmarks

**Main result: G1 + Dex3-1 (3 fingers), 100 unseen random layouts.**

| Policy | Success | Teacher calls | Compute per tick (mean) | Teacher calls per success |
|---|:---:|:---:|:---:|:---:|
| Teacher only | 86% | 100% | 10.3 ms | 558 |
| **Router (ours)** | **76%** | **32%** | **~3.4 ms (3x less)** | **226 (2.5x fewer)** |
| Random router, same budget | 43% | 31% | ~3.2 ms | 545 |
| Student only | 16% | 0% | 0.08 ms | - |

* **Router vs random router** on the same 100 layouts: router wins 38, random wins 5, **p = 2.5e-7** (exact McNemar).
  Calling the teacher at the right moments matters.
* **Router vs teacher:** 12 vs 2, p = 0.013. The router costs ~10 points of success for 3x less compute.
* **Latency** (laptop CPU, one process, one thread, timed back to back): teacher 10.3 ms mean / 12.4 ms p95,
  student 0.08 / 0.15 ms, phase net 0.06 / 0.07 ms. Router compute per tick is estimated from these and the
  measured teacher-call fraction (latencies measured inside the parallel evaluation are inflated).

**Case study: same pipeline on a 5-finger Inspire hand** (branch `inspire-hand`; 82 clips including 20 new
5-finger demos; 100 layouts from that hand's own reach map, so the layouts differ from the table above).

| Policy | Success | Teacher calls | Compute per tick |
|---|:---:|:---:|:---:|
| Teacher only | 59% | 100% | 10.6 ms |
| Router | 59% | 44% | ~4.8 ms (2.2x less) |
| Random router, same budget | 51% | 44% | ~4.7 ms |
| Student only | 1% | 0% | 0.07 ms |

The router matched the teacher (5 vs 5 discordant layouts) at 2.2x less compute, but the router-vs-random
gap was not significant (p = 0.23) and the teacher itself was much weaker. **In our case study the identical
task became substantially harder to execute with the more complex hand** (details in the Appendix). This is an
observation from one task and one run per hand, not a general law: the hand models, grasp strategies, data
and test layouts also differ between the two runs.

**Not done:** the planned generic-dataset ablation (same pipeline on a generic hand-object dataset) and any
real-robot test. All results are in simulation.

*Video: `g1_pick_place.mp4` shows an early scripted pick-and-place in the sim. Policy rollouts can be
rendered with `view_replay.py`.*

---

## 🤝 Contributing

Issues and pull requests are welcome. The lab notebook `CLAUDE.md` records why each design choice was made;
please read the relevant part before changing a constant.

---

## 📜 License

No license has been chosen yet. Until a `LICENSE` file is added, all rights are reserved. Third-party assets
keep their own licenses: the G1 model from MuJoCo Menagerie (BSD-3-Clause) and the Inspire hand model from
Unitree's `unitree_ros` repository.

---

## 🙏 Acknowledgements

* [MuJoCo](https://mujoco.org/) and [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  for the physics and the Unitree G1 model.
* [Unitree `unitree_ros`](https://github.com/unitreerobotics/unitree_ros) for the G1 + Inspire hand description.
* [ACT / Action Chunking with Transformers](https://tonyzhaozh.github.io/aloha/) (Zhao et al., 2023) for the
  teacher design and temporal ensembling.
* DART (Laskey et al., 2017) and DAgger (Ross et al., 2011) for noise injection and interactive distillation.
* [pupil-apriltags](https://github.com/pupil-labs/apriltags) for tag detection; OpenCV for calibration.
* Kaggle and Google Colab for free GPU time.

---

## 📎 Appendix: our journey

### Timeline (deadline 4 Nov 2026)
1. **1 Nov:** simulation, IK, grasp check, scripted pick-and-place; printed tags, calibration, tag tracking.
2. **2 Nov:** recorded demos in four batches (62 usable), retargeted and replayed them on the robot.
3. **3 Nov:** teacher versions v1 to v5 failed for different reasons until the velocity recipe worked; then the
   student, router and evaluation; in the evening the 5-finger Inspire case study with 20 new recordings.

### What worked
* **Physics replay as a filter.** Retargeted human motion is noisy; replaying every demo in MuJoCo and keeping
  only physical successes gave clean training data (41/62 for Dex3, 61/82 for Inspire).
* **Object-centric retargeting.** The robot uses its own measured grasp; the human demo supplies where the
  tube and coaster are, the carry path and the timing. Copying human finger poses onto a different hand did
  not need to be solved.
* **Measuring instead of guessing.** Grasp offsets, the reachable workspace and the coaster position were all
  swept or measured. Every guess we made was wrong at least once.
* **Arm velocities in the observation.** Teachers v1 to v5 reached at most 2/9 on held-out layouts. All of them
  failed where a state-only policy cannot tell "arriving at a point" from "leaving it". Adding 7 joint
  velocities took the teacher to 7/9, then 86% on 100 layouts.
* **Chunked student.** One-step students were unstable (see below). Giving the student the teacher's chunked
  output and temporal ensembling made it trainable.
* **DAgger inside the router.** Training the student on the states it meets when deployed raised the router
  from 42% to 76%, and turned "router vs random" from not significant (p = 0.53) into p = 2.5e-7.

### What didn't work (and what we learned)
* **First tag prints:** 0% detection, because the tags were trimmed to the black edge. Tags need a white margin.
* **cv2.aruco on a curved wrist tag:** missed detections; pupil-apriltags fixed it.
* **Chunk-target bug (teacher v1):** chunks were relative to the wrong reference, so the hand crawled. 0/9.
* **Behaviour-cloning drift:** fixed with DART noise-injected replays.
* **Demos that eased to full stops:** created fixed points where a state-based policy hovered forever; fixed by
  one continuous motion along the path.
* **Grip-label timing and object-relative actions (v4, v5):** both made things worse; reverted.
* **One-step student with velocity inputs:** learned "next move = current velocity" (the copycat problem) and
  its targets ran away to metres within 10 ticks.
* **Router before router-DAgger:** 42% vs random 37%, not significant. The student was the bottleneck.
* **Infrastructure:** a cross-validation run was lost when the laptop slept; a Colab GPU quota ran out. We
  moved to Kaggle (outputs persist) and added per-round checkpoints with resume.

### The 5-finger case study: same task, harder to execute
With the identical pipeline, moving from the 3-finger Dex3 to the 5-finger Inspire needed:
1. **A new grasp.** The Dex3's side grasp failed 0/144 in a sweep: the Inspire's thumb base sticks out of the
   palm where the tube would rest. The working grasp is pre-shaped and comes straight down over the tube.
2. **Physics fixes to the manufacturer's model:** no joint damping (fingers snapped shut in 54 ms and threw the
   tube) and no torsional friction (the tube rotated out of the grip).
3. **De-weighting two fingers.** With all five squeezing, the tube tipped ~30 degrees in the hand within 4 s;
   ring and pinky had to close only lightly (uprightness 0.86 -> 0.994). The working grip is effectively three
   fingers.
4. **Less noise tolerance:** 0/192 noisy replays survived the Dex3's 1 cm DART noise; we used 0.5 cm.
5. **A grip-decision threshold of 0.7** (chosen on training layouts): the teacher hesitated to release.

Even with ~2x the training data, the teacher reached 59% (Dex3: 86%). We read this as an observation, not
a law; see the caveats in [Results](#-results--benchmarks).

### Assumptions we made
* **Fixed pelvis.** The G1 does not balance; whole-body control is out of scope.
* **State-based policies.** The policy sees simulator state, not images, so a real robot would need state
  estimation for the tube and coaster.
* **Binary grip.** The fingers switch between an open and a closed pose; human finger motion is not retargeted.
* **Tube mass 0.15 kg (guessed, not weighed);** sizes measured from the real objects.
* **Rigid tube in the hand** when reconstructing the tube's start position from the wrist at the place moment.
* **Retargeting scale S = 0.3** about the head plus one shared shift onto the robot's measured workspace.
* **One evaluation-criterion change made after seeing results** (disclosed): "released" was redefined as "not
  grasped" by the grasp definition, instead of "no finger contact at all", which was stricter than the replay
  filter that built the training data.
* **Latency measured on the development laptop's CPU**, not on robot hardware.

### Future work
* Image-based teacher; real-robot test.
* The generic-dataset ablation (planned, not done).
* A controlled hand comparison: the same Inspire hand with two fingers disabled vs all five, everything else
  identical.
* Fix the Inspire teacher's grip uncertainty with more varied release positions in the data, instead of a
  threshold.
