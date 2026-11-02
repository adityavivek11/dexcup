"""Stage 1 (v2): full Unitree G1 + Dex3-1 hands at a table, with an egocentric head camera.

WHY switch from the flying wrist to the full body?
  1. Egocentric data: your phone sees the world from your head. The G1 has a head camera in the same
     place, so "human head frame" maps onto "robot head frame" directly. That's the cleanest
     retargeting reference: we express your wrist pose *relative to your head* and ask the robot's
     arm to put its wrist at the same pose *relative to its head* (scaled to the robot's size).
  2. Realism: the arm's 7 joints have limits and finite reach, so poses the human can do but the
     robot can't get caught here instead of at deployment.

WHY a fixed pelvis?
  Balancing while manipulating is a separate, hard problem (whole-body control). Our question is about
  routing between teacher and student, so we bolt the pelvis in place at standing height. Real G1
  manipulation work does the same, or runs a separate locomotion controller underneath.

Run:  python build_g1_scene.py   (prints a summary and saves a head-camera image)
"""
from pathlib import Path
import mujoco
import numpy as np

MENAGERIE = Path.home() / "mujoco_menagerie" / "unitree_g1"

RIGHT_ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
             "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
             "right_wrist_yaw_joint"]
RIGHT_FINGERS = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                 "right_hand_index_0_joint", "right_hand_index_1_joint",
                 "right_hand_middle_0_joint", "right_hand_middle_1_joint"]

# Preferred right-arm posture (elbow bent ~70 deg, upper arm forward and slightly out). IK's null-space
# term pulls toward it; chosen by searching postures for a smooth, body-collision-free reach.
ARM_NOMINAL = [-0.8, -0.3, 0.3, 1.2, 0.0, 0.0, 0.0]
READY_PALM = [0.10, -0.25, 0.95]   # hand above the table, close to the body, fingers forward

TABLE_TOP = 0.78
# Real objects (measured 2 Nov): cardboard tube 9.4 cm diameter x 17.8 cm tall; "coaster" = tape roll
# 9 cm diameter x 2 cm tall. The tube (radius 4.7) is wider than the roll (4.5), so it rests ON the roll:
# the roll is a solid, collidable step here, not a flat visual-only disc as in the 7 cm test cup.
CUP_RADIUS, CUP_HALF_H, CUP_MASS = 0.047, 0.089, 0.15    # mass not measured: light tube + small object
COASTER_RADIUS, COASTER_HALF_H = 0.045, 0.010
CUP_START = np.array([0.32, -0.20, TABLE_TOP + CUP_HALF_H])  # right of centre, within right-arm reach
COASTER = np.array([0.32, 0.00, TABLE_TOP + COASTER_HALF_H])  # 20 cm to the left, in front of the body


def build():
    """Returns (spec, stand_qpos). stand_qpos = robot joint angles of the standing pose."""
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / "g1_with_hands.xml"))
    spec.meshdir = str(MENAGERIE / "assets")
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10

    # Fixed base: delete the free joint (pelvis stays at its 0.793 m standing height).
    # The 'stand' keyframe assumed a free joint, so it no longer fits; we store the pose ourselves.
    stand_qpos = mujoco.MjModel.from_xml_path(str(MENAGERIE / "g1_with_hands.xml")).key_qpos[0][7:].copy()
    spec.delete(spec.key("stand"))
    spec.delete(spec.joint("floating_base_joint"))

    # Soften finger servos (both hands). G1's kp=500 is tuned for real motors and flings a sim cup.
    # Edit gains in place rather than set_to_position(), which would wipe ctrlrange.
    for a in spec.actuators:
        if "hand" in a.name:
            g = np.array(a.gainprm); g[0] = 20; a.gainprm = g
            b = np.array(a.biasprm); b[1] = -20; b[2] = -0.5; a.biasprm = b

    # Contact exclusions for neighbouring links whose simplified collision meshes overlap.
    # MuJoCo auto-excludes only parent-child pairs; these are two joints apart, so they registered
    # constant contacts that fought the arm servos.
    for side in ["left", "right"]:
        for a, b in [(f"{side}_wrist_roll_link", f"{side}_wrist_yaw_link"),
                     ("torso_link", f"{side}_shoulder_roll_link"),
                     ("torso_link", f"{side}_shoulder_yaw_link")]:
            spec.add_exclude(bodyname1=a, bodyname2=b)

    w = spec.worldbody
    # g1_with_hands.xml has no floor or light (Menagerie's scene_with_hands.xml adds them).
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, .05], rgba=[.3, .35, .4, 1])
    w.add_light(pos=[0.5, 0, 2.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.42, -0.05, TABLE_TOP - 0.02],
               size=[0.20, 0.35, 0.02], rgba=[.55, .42, .3, 1])
    w.add_geom(name="table_leg", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.50, -0.05, (TABLE_TOP - 0.04) / 2],
               size=[0.04, 0.04, (TABLE_TOP - 0.04) / 2], rgba=[.4, .3, .2, 1])

    cup = w.add_body(name="cup", pos=CUP_START.tolist())
    cup.add_freejoint(name="cup_free")
    cup.add_geom(name="cup_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CUP_RADIUS, CUP_HALF_H],
                 mass=CUP_MASS, friction=[1.0, 0.01, 0.001], rgba=[.85, .75, .55, 1])
    cup.add_site(name="cup_center", size=[0.005])

    w.add_geom(name="coaster", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=COASTER.tolist(),
               size=[COASTER_RADIUS, COASTER_HALF_H], rgba=[.9, .85, .6, 1])
    w.add_site(name="coaster_center", pos=(COASTER + [0, 0, COASTER_HALF_H]).tolist(), size=[0.005])

    # Egocentric camera where the G1's head sensor sits (~1.27 m), pitched ~50 deg down at the table.
    # MuJoCo cameras look along local -z, with image-right = local x and image-up = local y.
    # Build the rotation from explicit axes (Euler angles got the roll wrong in the first try):
    #   look = forward and 50 deg down;  right = robot's right (-y);  up = right-handed completion.
    pitch = np.deg2rad(50)
    look = np.array([np.cos(pitch), 0, -np.sin(pitch)])
    x_ax = np.array([0, -1, 0])
    y_ax = np.array([np.sin(pitch), 0, np.cos(pitch)])
    R = np.column_stack([x_ax, y_ax, -look])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    head_pos = [0.10, 0, 0.40]   # just in front of the face plate, so the head shell isn't in view
    spec.body("torso_link").add_camera(name="ego", pos=head_pos, quat=q, fovy=70)
    spec.body("torso_link").add_site(name="head", pos=head_pos, size=[0.01], rgba=[0, 1, 1, 1])

    # Fingertip and palm sites (same offsets as v1; measured from the collision meshes).
    tips = {"right_hand_thumb_2_link": [-0.003, 0.048, -0.002],
            "right_hand_index_1_link": [0.048, -0.003, 0.002],
            "right_hand_middle_1_link": [0.048, -0.003, 0.002]}
    for link, off in tips.items():
        spec.body(link).add_site(name=link.replace("right_hand_", "").replace("_link", "_tip"),
                                 pos=off, size=[0.006], rgba=[1, .2, .2, 1])
    spec.body("right_wrist_yaw_link").add_site(name="palm_center", size=[0.006], rgba=[.2, .2, 1, 1])

    return spec, stand_qpos


def make():
    """Compile and return (model, data, stand_qpos) in the standing pose with servos holding it."""
    spec, stand_qpos = build()
    m = spec.compile()
    d = mujoco.MjData(m)
    reset(m, d, stand_qpos)
    return m, d, stand_qpos


def reset(m, d, stand_qpos):
    mujoco.mj_resetData(m, d)
    n_robot = len(stand_qpos)
    d.qpos[:n_robot] = stand_qpos           # robot joints come first in qpos; cup's free joint is last
    for a in range(m.nu):                   # every robot actuator holds its stand angle
        d.ctrl[a] = d.qpos[m.jnt_qposadr[m.actuator_trnid[a][0]]]
    mujoco.mj_forward(m, d)


if __name__ == "__main__":
    import os
    if __import__("sys").platform.startswith("linux"):  # headless servers; macOS uses its native GL
        os.environ.setdefault("MUJOCO_GL", "egl")
    m, d, _ = make()
    print(f"nq={m.nq} nu={m.nu}  head={d.site('head').xpos.round(3)}  "
          f"palm={d.site('palm_center').xpos.round(3)}  cup={d.site('cup_center').xpos.round(3)}")
