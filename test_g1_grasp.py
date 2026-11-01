"""Scripted pick-and-place on the full G1 — the physics sanity check, now with a real arm.

Control loop (this is the same structure the learned policies will plug into):
  every 20 ms (50 Hz):  desired palm pose  --IK-->  7 arm joint targets  -->  arm servos
                        desired finger angles  ------------------------->  finger servos
  physics runs at 500 Hz underneath (2 ms steps).
Why 50 Hz? It's a typical rate for manipulation policies, and the latency/teacher-call budget in your
headline table is measured per control tick, so we fix the tick now.

Each 20 ms tick, IK is warm-started from the previous solution, which keeps the 7-DoF arm on one
continuous branch instead of flipping elbow configurations.
"""
import os
import mujoco
import numpy as np
from build_g1_scene import make, CUP_START, COASTER, RIGHT_ARM
from contacts import GraspChecker
from ik import ArmIK, reset_ready

CTRL_DT = 0.02
m, d, STAND = make()
SUBSTEPS = int(round(CTRL_DT / m.opt.timestep))
ik = ArmIK(m)
gc = GraspChecker(m)
ARM_ACT = [m.actuator(j).id for j in RIGHT_ARM]
# ALL 7 finger joints are commanded explicitly in both states. Leaving any out means it inherits the
# reset pose: G1's 'stand' curls thumb_1 to -1.05, which parked the thumb inside the grasp space.
FINGER_JOINTS = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                 "right_hand_index_0_joint", "right_hand_index_1_joint",
                 "right_hand_middle_0_joint", "right_hand_middle_1_joint"]
FINGERS_OPEN = dict.fromkeys(FINGER_JOINTS, 0.0)
FINGERS_CLOSE = {**FINGERS_OPEN, "right_hand_index_0_joint": 1.2, "right_hand_index_1_joint": 1.3,
                 "right_hand_middle_0_joint": 1.2, "right_hand_middle_1_joint": 1.3,
                 "right_hand_thumb_2_joint": -1.3}
HAND_QUAT = np.array([1.0, 0, 0, 0])            # fingers forward (+x), hand closes toward +y
OFF = np.array([-0.104, -0.05, 0.0])            # palm relative to cup centre (worked in the v1 test)
CUP_ON_COASTER = np.array([COASTER[0], COASTER[1], CUP_START[2]])

CLEAR = np.array([0, -0.02, 0])                 # descend 2 cm away from the cup, then slide in
WAYPOINTS = [  # (phase, palm target in world, fingers closed?, seconds)
    # Come down BESIDE the cup with clearance, then slide in sideways (like a person does).
    # Descending straight onto the grasp pose put the open thumb on the rim and blocked the arm.
    ("approach_above", CUP_START + OFF + CLEAR + [0, 0, 0.12], False, 1.2),
    ("approach_down",  CUP_START + OFF + CLEAR,                False, 1.0),
    ("slide_in",       CUP_START + OFF,                        False, 0.8),
    ("grasp",          CUP_START + OFF,                        True,  0.8),
    ("lift",           CUP_START + OFF + [0, 0, 0.09],         True,  0.8),
    ("transport",      CUP_ON_COASTER + OFF + [0, 0, 0.09],    True,  1.5),
    ("place",          CUP_ON_COASTER + OFF + [0, 0, 0.005],   True,  1.0),
    ("release",        CUP_ON_COASTER + OFF + [0, 0, 0.005],   False, 0.6),
    ("back_off",       CUP_ON_COASTER + OFF + CLEAR + [0, 0, 0.005], False, 0.6),
    ("retreat",        CUP_ON_COASTER + OFF + CLEAR + [0, 0, 0.12],  False, 0.8),
]


def run(verbose=True, record=False):
    # Start in the 'ready' pose (hand raised above the table). Starting from the hanging-arm stand pose
    # swept the fingers through the table edge and the hip on the way up.
    reset_ready(m, d, STAND, ik)
    frames, log = [], []
    renderer = mujoco.Renderer(m, 240, 320) if record else None
    cur = d.site("palm_center").xpos.copy()
    for phase, target, closed, secs in WAYPOINTS:
        start, ticks = cur.copy(), int(secs / CTRL_DT)
        for k in range(ticks):
            a = min(1.0, k / (0.7 * ticks))
            a = a * a * (3 - 2 * a)                       # smoothstep: no velocity jumps at waypoints
            cur = start + a * (np.asarray(target) - start)
            arm_q, _, _ = ik.solve(d.qpos, cur, HAND_QUAT, iters=30)
            d.ctrl[ARM_ACT] = arm_q
            for j, q in (FINGERS_CLOSE if closed else FINGERS_OPEN).items():
                d.ctrl[m.actuator(j).id] = q
            mujoco.mj_step(m, d, nstep=SUBSTEPS)
            if record and k % 2 == 0:
                renderer.update_scene(d, "ego"); ego = renderer.render()
                cam = mujoco.MjvCamera(); cam.lookat[:] = [0.25, -0.1, 0.9]
                cam.distance, cam.azimuth, cam.elevation = 1.3, 140, -20
                renderer.update_scene(d, cam); frames.append(np.hstack([renderer.render(), ego]))
        ok, f = gc.is_grasped(d)
        track_err = np.linalg.norm(d.site("palm_center").xpos - np.asarray(target))
        log.append((phase, ok))
        if verbose:
            print(f"{phase:15s} grasp={ok!s:5s} cup={d.site('cup_center').xpos.round(3)} "
                  f"palm_track_err={track_err*1000:4.1f}mm  forces="
                  + " ".join(f"{k}:{v:.1f}" for k, v in f.items()))
    dist = np.linalg.norm(d.site("cup_center").xpos[:2] - d.site("coaster_center").xpos[:2])
    upright = d.body("cup").xmat.reshape(3, 3)[2, 2] > 0.95
    success = bool(dist < 0.03 and upright)
    if verbose:
        print(f"\ncup-to-coaster = {dist*100:.1f} cm, upright={upright}  ->  SUCCESS={success}")
    return success, log, frames


if __name__ == "__main__":
    if __import__("sys").platform.startswith("linux"):  # headless servers; macOS uses its native GL
        os.environ.setdefault("MUJOCO_GL", "egl")
    run()
