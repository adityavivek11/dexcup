# 🤖 dexcup: Call the Big Brain Only When It Matters

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Simulator: MuJoCo 3.17](https://img.shields.io/badge/Simulator-MuJoCo%203.17-green.svg)](https://mujoco.org/)

> A simulated Unitree G1 learns tube pick-and-place from our own iPhone recordings. Calling a large teacher
> policy with the help of a router on only 32% of control steps, and a 250x smaller student otherwise, gives
> 76% success at about 3x less estimated compute, compared with 86% success when the teacher runs at every
> step. A random schedule with the same number of teacher calls reaches only 43%.

📄 **Paper:** [`paper/dexcup_paper.tex`](paper/dexcup_paper.tex) (PDF: [ADD DRIVE LINK]) · 🎥 **Recordings:** [ADD DRIVE LINK] · 🎬 **Demo:** [`demo_router.mp4`](demo_router.mp4)

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#️-system-architecture)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Usage Guide](#-usage-guide)
- [Configuration](#️-configuration)
- [Results & Benchmarks](#-results--benchmarks)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Appendix: our journey](#-appendix-our-journey)

---

## 📖 Overview

Large manipulation policies are accurate but expensive to run 50 times a second, which is a problem for a
robot's limited onboard compute. Yet most of a pick-and-place is easy: reaching and carrying don't need much
precision. Only grasping and placing are delicate.

Our goal was to make a humanoid manipulation policy cheaper to run without giving up much success, by calling
the large policy only when it matters. The test task: a simulated Unitree G1 (fixed pelvis, 3-finger Dex3-1
hand, in MuJoCo) picks up a cardboard tube (9.4 cm x 17.8 cm) and places it on a tape-roll "coaster",
learning from our own recordings rather than a generic dataset.

We recorded 62 usable egocentric iPhone videos of a person doing the task, tracked them with two AprilTags,
retargeted them to the robot and replayed them in physics, keeping the 41 that physically succeed. From those
we trained, by imitation learning (DART + DAgger) on state-based observations, a large teacher (7.4M
parameters), a 250x smaller student, and a small router that calls the teacher only during grasp and
place.

Across 100 unseen layouts, the router succeeds 76% of the time while calling the teacher on 32% of
control steps, at about 3x less estimated compute per step. The teacher running at every step succeeds
86%. A random router with the same number of teacher calls reaches only 43%, so when the teacher is
called matters.

As an observation, we noticed something interesting. Repeating the pipeline on a 5-finger hand, with 20
additional recordings (82 in total), showed the flip side: the identical task became substantially harder to
execute with the more complex hand. Compared with the 3-finger Dex3-1, the teacher's success fell from 86% to
59% and the router's from 76% to 59%, while the router's teacher calls rose from 32% to 44% of control steps
and its compute saving shrank from about 3x to about 2.2x. Each hand was tested on 100 layouts from its own
workspace.

---

## ✨ Key Features

* **3x less compute per step:** the router needs about 2.6 ms per control step on average, vs 8.0 ms for the
  teacher alone (estimated from measured model timings and the teacher-call rate), while reaching 76% success
  vs the teacher's 86%.
* **Self-collected human demos:** stock iPhone camera (no depth, no LiDAR), two printed AprilTags and a printed
  checkerboard. All metric information comes from calibration and tag poses.
* **Router-in-the-loop DAgger:** the student is trained on exactly the states it meets when deployed inside the
  router. This raised the router from 42% to 76% success.
* **Cheap compute:** simulation and evaluation on a laptop CPU; teacher trained on a free Kaggle T4 GPU.

---

## 🏗️ System Architecture

> 📝 *I know this description runs long, but I think leaving out any detail would be unfair to the project.*

The pipeline is built around our own egocentric recordings. Every training episode originates from a video we
recorded; no scripted or generic demonstrations are mixed in. Each demo supplies the tube's start position,
the coaster's position, the carry path and the timing. The robot supplies what a human video can't: its own
measured grasp and scripted approach and retreat paths, retimed to the human's durations.

```text
iPhone video ─► calibration + AprilTags ─► wrist, tube & coaster poses ─► retarget ─► physics replay ─► kept demos + DART copies
                                                                                                              │
                                        phase labels ─► ROUTER      demos ─► TEACHER ─► DAgger ─► STUDENT  ◄──┘

At run time, every 20 ms (50 Hz):
  state ─► router (phase net) ─┬─ grasp / place ─► TEACHER (7.4M, ~8 ms) ─────┐
                               └─ everything else ─► STUDENT (29k, ~0.05 ms) ─┤
                                                                              ▼
                              palm target + grip ─► arm IK ─► joint servos (500 Hz physics)
```

**Data**
1. **Recording:** head-mounted iPhone, stock camera, 1080p, 30 fps, 0.5x lens, exposure and focus locked. One
   demo per clip, varying the tube and coaster positions.
2. **Calibration:** checkerboard video -> camera intrinsics (fx, fy, cx, cy fitted separately; ~95° field of view).
3. **Tracking:** AprilTags (tag36h11, pupil-apriltags) on the back of the right wrist and beside the coaster.
   Poses are solved per frame and bad detections rejected by reprojection error. A checkerboard acts as a second
   fixed reference when the hand hides the coaster tag. Result: the wrist's pose in the coaster's frame.
4. **Coaster location:** the tape roll is found per clip by a circle fit in a rectified top view.
5. **Quality gate:** each clip must pass tag-visibility and tracking-gap checks: 62 usable clips.

**Demonstrations**

6. **Object-centric retargeting:** the tube's start is reconstructed by running the carry backwards from the
   place, assuming the tube is rigid in the hand. Positions are scaled (S = 0.3) about the head and shifted
   onto the robot's measured workspace (`reach_map.py`). The carry follows the human's tube path; approach and
   retreat are one continuous scripted motion each. Hand orientation is fixed and the grip is binary: human
   finger motion is not copied.
7. **Physics validation:** every demo is replayed on the G1 in MuJoCo; only physical successes are kept (41 of 62).
8. **Phase labels:** approach, grasp, transport, place, retreat, from physics-only rules (palm pose, tube lift,
   distance to the coaster, finger contact).
9. **DART:** 4 noisy replays per clip (slowly varying noise, sigma 1 cm, none during grasp and release), with the
   clean target as the label, teaching recovery from small errors. Successful noisy replays of training clips
   only: 88 training episodes (32 clean + 56 noisy) and 9 held-out clips, frozen before training.

**Policies**

10. **Observation:** 38 state numbers: palm, tube position and tilt, coaster, 7 finger joints, 7 arm joints,
    relative vectors, and 7 arm joint velocities. The student and router use the first 31 (no velocities).
11. **Action:** palm target relative to the current palm position, plus grip.
12. **Teacher (7.4M parameters):** ACT-style transformer (256 dims, 4 layers, 8 heads) over grouped observation
    tokens, predicting the next 16 steps. Overlapping predictions are blended, newer ones weighted more (ACT
    temporal ensembling). Trained by behaviour cloning for 80 epochs on a Kaggle T4 GPU.
13. **Student (28.9k parameters):** MLP (two hidden layers of 128) with the same 16-step output and blending.
    Trained by behaviour cloning, then DAgger (the teacher labels states the student visits), then
    **router-in-the-loop DAgger** (4 rounds of 36 episodes inside the real router). The best round was
    chosen on 40 separate layouts.
14. **Router (6.5k parameters):** small classifier predicting the phase from the observation alone, never from
    ground-truth labels or contacts. It calls the teacher for grasp and place and the student otherwise.

**Control**

15. **Palm target -> arm joints:** damped-least-squares IK with a posture term to keep the arm on one branch,
    running at 50 Hz, with position servos and 500 Hz physics.

**Evaluation**

16. **Test layouts:** 100 unseen random layouts within the training area, at least 10 cm apart and inside the
    measured workspace, with 20 s allowed per episode.
17. **Success:** the tube stays within 3 cm of the coaster centre, upright and released, for 1 s.
18. **Rows:** teacher only, student only, router, and a **random router** with the same teacher-call rate and
    segment length (the key control).
19. **Statistics and latency:** paired exact McNemar tests on the same layouts; per-model latency timed in one
    process on one thread, combined with the measured call rate to estimate compute per step.

---

## 📁 Repository Structure

The scripts sit in one folder, one per pipeline stage, grouped here in pipeline order.

```text
# Reviewer tools (no training needed)
├── watch_policy.py           # watch a trained policy, live or as a video
├── summarize_results.py      # recompute every table number and p-value from stored results
├── eval_policy.py            # closed-loop evaluation (teacher, router, random, student)
├── bench_latency.py          # per-model latency
├── demo_router.mp4           # router on test layout 0
├── models/                   # trained teacher, student, router
├── results/                  # per-episode results + latency
├── data/                     # processed robot episodes, split, reach map (no video)
├── paper/dexcup_paper.tex    # the paper

# Data: phone video -> poses
├── make_printables.py        # print-ready AprilTags + checkerboard (printables/*.pdf)
├── calibrate.py              # camera intrinsics -> calib/camera.json
├── check_clips.py            # quality gate for every recording batch
├── tags.py, track_tags.py    # AprilTag detection and wrist/coaster tracking
├── find_coaster.py           # finds the tape roll per clip

# Simulation
├── build_g1_scene.py         # G1 + Dex3-1, fixed pelvis, table, tube, coaster, head camera
├── ik.py, contacts.py        # arm IK; grasp check (thumb + opposing finger > 0.5 N)
├── test_g1_grasp.py          # scripted pick-and-place (physics sanity check)
├── reach_map.py              # measures where the robot can grasp and lift

# Demonstrations
├── retarget.py, replay.py    # human demo -> robot plan -> physics replay (+ DART noise)
├── gen_demos.py              # scripted test demos + the physics-only phase labeller
├── view_replay.py            # render or live-view replays

# Policies
├── policy.py                 # observation/action recipes, Teacher, Student, PhaseNet
├── train_teacher.py          # teacher training (GPU or CPU)
├── train_student.py          # DAgger and router-in-the-loop DAgger
├── router.py                 # phase net training; router and random-router agents
├── sim_env.py                # closed-loop environment with per-call timing
├── make_cloud_bundle.py, kaggle/, cloud/, modal_train.py, cloud_train.ipynb   # cloud training

└── requirements.txt          # exact package versions
```

**Branches:** `main` = Dex3-1 (main result). `inspire-hand` = the 5-finger case study, with its own models, data
and results, and the same commands.

---

## 💻 Installation

Tested on Windows 11 with conda and Python 3.13; `pip` in a plain virtual environment also works. On Windows, run
these in the Miniconda Prompt.

```bash
git clone https://github.com/adityavivek11/dexcup.git
cd dexcup
```

Robot model (sparse clone of only the G1 from MuJoCo Menagerie; the code looks in `~/mujoco_menagerie`):
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie
git -C ~/mujoco_menagerie sparse-checkout set unitree_g1
```

Environment (`--override-channels` keeps it on conda-forge, so Anaconda's default-channel terms don't come up):
```bash
conda create -n dexcup --override-channels -c conda-forge python=3.13 pip -y
conda activate dexcup
pip install -r requirements.txt
```

---

## 📊 Dataset

* **Raw recordings (iPhone videos):** [ADD DRIVE LINK]
* **Processed episodes (robot joint and object data, no video):** in `data/`. These are all the evaluation needs.

To record your own data, see [Reproducing from scratch](#reproducing-from-scratch-optional).

---

## 🚀 Usage Guide

**No training is needed: the trained models are included.**

```bash
python test_g1_grasp.py                               # check the sim: ends with SUCCESS=True
python watch_policy.py router --layout 0 --live       # watch the policy (close the window to stop)
python watch_policy.py router --layout 0 --video r.mp4  # or save a video
python summarize_results.py                           # recompute every table number from stored results
python eval_policy.py router --layouts random --n 10  # re-run the first 10 test layouts (about 2 min)
```

Agents: `teacher`, `router`, `random`, `student`. `eval_policy.py` overwrites `results/<agent>_random.json`; restore
the stored results afterwards with `git checkout results`. Run `random` before `router` with a small `--n`, since
`random` reads its call rate from the router's results file.

**What to expect:** on clean copies in fresh environments, the first 10 layouts of all 4 agents reproduced the stored
results exactly (same success and tick count, 40 of 40, on both branches). Other machines may differ slightly.

The models are saved as whole Python objects, so they load with `weights_only=False`, which can run code. Only load
model files from a source you trust.

### Reproducing from scratch (optional)

The full 100-layout evaluation takes about 15 minutes per agent on a laptop CPU:
```bash
python eval_policy.py teacher --layouts random --n 100
python eval_policy.py router  --layouts random --n 100
python eval_policy.py random  --layouts random --n 100
python eval_policy.py student --layouts random --n 100
python bench_latency.py
```

Retraining (a T4 GPU takes about 30 min for the teacher, a laptop CPU about 10 h):
```bash
python train_teacher.py --dart --epochs 80 --frame palm --vel --name teacher_pv.pt
python router.py
python train_student.py --teacher models/teacher.pt
python train_student.py --router --rounds 4 --rollouts 36
```

From your own recordings: print `make_printables.py` (measure the printed sizes), calibrate with
`calibrate.py <video> --square-mm <size>`, record head-mounted at 1080p, 30 fps, 0.5x lens, HDR and stabilisation off,
exposure and focus locked, portrait, moving slowly, then:
```bash
python check_clips.py recordings/
python track_tags.py recordings/ --workers 4
python find_coaster.py recordings/
python reach_map.py
python replay.py
python replay.py --noise 4 <training clips>
```

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

**Main result: G1 + Dex3-1 (3 fingers), 100 unseen random layouts.** Every number below can be recomputed with
`python summarize_results.py`.

| Policy | Success | Teacher calls | Compute per step (est.) | Teacher calls per success |
|---|:---:|:---:|:---:|:---:|
| Teacher only | 86% | 100% | 8.0 ms | 558 |
| **Router (ours)** | **76%** | **32%** | **2.6 ms (3.0x less)** | **226 (2.5x fewer)** |
| Random router, same budget | 43% | 31% | 2.5 ms | 545 |
| Student only | 16% | 0% | 0.05 ms | - |

* **Router vs random router** on the same 100 layouts: router better on 38, random better on 5, **p = 2.5e-7** (exact
  McNemar). Calling the teacher at the right moments matters.
* **Router vs teacher:** 12 vs 2, p = 0.013. The router costs 10 points of success for 3x less compute.
* **Latency** (laptop CPU, one process, one thread, mean / p95): teacher 7.97 / 10.40 ms, student 0.055 / 0.065 ms,
  phase net 0.051 / 0.077 ms. Compute per step is an **estimate** from these and the measured teacher-call rate, not an
  end-to-end measurement. Absolute times vary with the machine's power state (the teacher measured 8-23 ms across
  sessions); the ratio stayed near 3 and is the number to compare.

**Case study: same pipeline on a 5-finger Inspire hand** (branch `inspire-hand`; 82 clips including 20 new 5-finger
demos; 100 layouts from that hand's own reach map, so the layouts differ from the table above).

| Policy | Success | Teacher calls | Compute per step (est.) |
|---|:---:|:---:|:---:|
| Teacher only | 59% | 100% | 8.1 ms |
| Router | 59% | 44% | 3.6 ms (2.2x less) |
| Random router, same budget | 51% | 44% | 3.6 ms |
| Student only | 1% | 0% | 0.06 ms |

The router matched the teacher (5 vs 5 discordant layouts) at 2.2x less compute, but the router-vs-random gap was not
significant (p = 0.23) and the teacher itself was much weaker. **In our case study the identical task became
substantially harder to execute with the more complex hand** (details in the Appendix and the paper). This is an
observation from one task and one run per hand, not a general law: the hand models, grasp strategies, data and test
layouts also differ between the two runs.

**Scope:** pick-and-place may look like a very simple experiment; that choice was due to time constraints. The same
method can be applied to other robotic tasks with identifiable delicate phases, though we have not yet tested any.

**Not done:** the planned generic-dataset ablation and any real-robot test. All results are in simulation.

---

## 🤝 Contributing

Issues and pull requests are welcome. Code comments record why each design choice was made; please read them
before changing a constant.

---

## 📜 License

No license has been chosen yet. Until a `LICENSE` file is added, all rights are reserved. Third-party assets
keep their own licenses: the G1 model from MuJoCo Menagerie (BSD-3-Clause) and the Inspire hand model on branch
`inspire-hand`, from Unitree's `unitree_ros` (BSD-3-Clause, copy in `assets/unitree_inspire/LICENSE`).

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
* AI assistance (Claude Code, Anthropic) for software development, analysis and drafting.

---

## 📎 Appendix: our journey

### Timeline (deadline 4 Nov 2026)
1. **1 Nov:** simulation, IK, grasp check, scripted pick-and-place; printed tags, calibration, tag tracking.
2. **2 Nov:** recorded demos in four batches (62 usable), retargeted and replayed them on the robot.
3. **3 Nov:** teacher versions v1 to v5 failed for different reasons until the velocity recipe worked; then the
   student, router and evaluation; in the evening the 5-finger Inspire case study with 20 new recordings.
4. **4 Nov:** clean-environment reproduction checks, this README and the paper.

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
1. **A new grasp.** The Dex3's side grasp completed no pick-and-place in two sweeps of 144 configurations each: the
   Inspire's thumb base sticks out of the palm where the tube would rest. The working grasp is pre-shaped and comes
   straight down over the tube.
2. **Physics fixes to the manufacturer's model:** no joint damping (fingers snapped shut in 54 ms and threw the
   tube) and no torsional friction (the tube rotated out of the grip).
3. **De-weighting two fingers.** With all five squeezing, the tube tipped ~30 degrees in the hand within 4 s;
   ring and pinky had to close only lightly (uprightness 0.86 -> 0.994). The working grip is effectively three
   fingers.
4. **Less noise tolerance:** 0/192 noisy replays survived the Dex3's 1 cm DART noise; we used 0.5 cm.
5. **A grip-decision threshold of 0.7** (chosen on training layouts): the teacher hesitated to release. On the test
   layouts, most of its failures are then the hand not closing, so one threshold can't fix both decisions.

Even with ~2x the training data, the teacher reached 59% (Dex3: 86%). We read this as an observation, not
a law; see the caveats in [Results](#-results--benchmarks).

### Assumptions we made
* **Fixed pelvis.** The G1 does not balance; whole-body control is out of scope.
* **State-based policies.** The policy sees simulator state, not images, so a real robot would need state
  estimation for the tube and coaster. There is no language input and no RL.
* **Binary grip.** The fingers switch between an open and a closed pose; human finger motion is not retargeted.
* **Tube mass 0.15 kg (guessed, not weighed);** sizes measured from the real objects.
* **Rigid tube in the hand** when reconstructing the tube's start position from the wrist at the place moment.
* **Retargeting scale S = 0.3** about the head plus one shared shift onto the robot's measured workspace.
* **One evaluation-criterion change made after seeing results** (disclosed): "released" was redefined as "not
  grasped" by the grasp definition, instead of "no finger contact at all", which was stricter than the replay
  filter that built the training data.
* **Latency measured on the development laptop's CPU**, not on robot hardware.

### Future work
* Image-based teacher; real-robot test; other tasks.
* The generic-dataset ablation (planned, not done).
* A controlled hand comparison: the same Inspire hand with two fingers disabled vs all five, everything else
  identical.
* Fix the Inspire teacher's grip uncertainty with more varied release positions in the data, instead of a
  threshold.
