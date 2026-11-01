# dexcup — G1 Dex3-1 cup-to-coaster with teacher/student routing

## Setup on Linux

```bash
# 1. Python env (3.10–3.12)
python3 -m venv ~/dexcup-env
source ~/dexcup-env/bin/activate

# 2. Robot models: must live at ~/mujoco_menagerie (build_g1_scene.py looks there)
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie

# 3. Dependencies (for an NVIDIA GPU, install the CUDA build of torch from pytorch.org first)
cd dexcup
pip install -r requirements.txt

# 4. Check it works
python build_g1_scene.py        # prints model size + head/palm/cup positions
python test_g1_grasp.py         # scripted pick-and-place, should end with SUCCESS=True
```

To watch the sim live: `python -m mujoco.viewer`, then drag a scene in.

Offscreen rendering (videos, head-camera images) uses `MUJOCO_GL=egl`, which the scripts set on
Linux. EGL works with NVIDIA drivers or Mesa. If rendering fails on a machine without a GPU,
run with `MUJOCO_GL=osmesa` (needs `sudo apt install libosmesa6`).

Check PyTorch sees the GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
nvidia-smi    # shows GPU model and memory
```

## Files

| File | What it does |
|---|---|
| `build_g1_scene.py` | Full G1 + Dex3-1 hands, fixed pelvis, table/cup/coaster, egocentric head camera |
| `ik.py` | Right-arm inverse kinematics (damped least squares + null-space posture) |
| `contacts.py` | Grasp confirmation: thumb + one opposing finger, >0.5 N each |
| `test_g1_grasp.py` | Scripted pick-and-place on the G1 (physics sanity check) |
| `gen_demos.py` | Randomized scripted demos + automatic phase labels (stand-in until real data) |
| `build_scene.py`, `test_scripted_grasp.py` | v1: floating-hand version, kept for reference |

## Pipeline status

1. Sim, IK, grasp check — **done**
2. iPhone 14 Pro recordings (Record3D, LiDAR, `.r3d`) — **recording**
3. Hand + AprilTag tracking → table frame
4. Retargeting (wrist → IK, fingers → vector retargeting)
5. Physics replay, filter, phase labels
6. Teacher policy (Diffusion Policy / ACT)
7. Student via DAgger
8. Router (online phase prediction)
9. Evaluation table + random-router control + generic-data ablation
