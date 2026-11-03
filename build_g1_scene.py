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

    spec.body("right_wrist_yaw_link").add_site(name="palm_center", size=[0.006], rgba=[.2, .2, 1, 1])
    from paths import HAND
    if HAND == "inspire":
        _swap_right_hand_to_inspire(spec)
        return spec, stand_qpos

    # Fingertip sites (same offsets as v1; measured from the collision meshes).
    tips = {"right_hand_thumb_2_link": [-0.003, 0.048, -0.002],
            "right_hand_index_1_link": [0.048, -0.003, 0.002],
            "right_hand_middle_1_link": [0.048, -0.003, 0.002]}
    for link, off in tips.items():
        spec.body(link).add_site(name=link.replace("right_hand_", "").replace("_link", "_tip"),
                                 pos=off, size=[0.006], rgba=[1, .2, .2, 1])
    return spec, stand_qpos


INSPIRE = Path.home() / "unitree_ros_g1"     # from unitreerobotics/unitree_ros robots/g1_description (3 Nov)
# The 6 motors of the Inspire RH56DFQ. Each finger's second joint is coupled to its first by the URDF's
# <mimic> tags, which MuJoCo's URDF importer turns into joint equality constraints (thumb: 1.6x and 2.4x).
INSPIRE_ACTS = ["R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint",
                "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint"]


def _swap_right_hand_to_inspire(spec):
    """Replace the right Dex3-1 with the 5-finger Inspire hand; the rest of the G1 stays as it is.

    WHY keep Menagerie's G1 and swap only the hand (instead of loading Unitree's full G1+Inspire URDF)?
    The Menagerie body already has tuned actuators, the stand pose and every fix in 'Pitfalls' (contact
    excludes, implicit integrator, fixed pelvis). The URDF has no actuators at all. Swapping one subtree
    keeps everything the arm IK, the camera and the replay were validated on; only the hand is new.
    The mount (0.0415 m along the wrist_yaw x axis, rpy (pi, 0, -pi/2)) is copied from Unitree's URDF.
    The palm_center site stays at the wrist_yaw origin, the same IK point as with the Dex3."""
    for a in list(spec.actuators):                      # actuators first: they reference the joints
        if a.name.startswith("right_hand_"):
            spec.delete(a)
    wrist = spec.body("right_wrist_yaw_link")
    for b in list(wrist.bodies):
        spec.delete(b)
    hs = mujoco.MjSpec.from_file(str(INSPIRE / "inspire_hand" / "DFQ_right_hand.urdf"))
    for mesh in hs.meshes:                              # absolute paths: the G1's meshes live elsewhere
        mesh.file = str(INSPIRE / mesh.file)
    base = hs.body("R_hand_base_link")
    frame = wrist.add_frame()
    frame.attach_body(base, "", "")
    spec.meshdir = str(MENAGERIE / "assets")
    # The URDF's coupled joints, re-added if the attach didn't carry the equalities over.
    have = {(e.name1, e.name2) for e in spec.equalities}
    for child, parent, mult in [("R_thumb_intermediate_joint", "R_thumb_proximal_pitch_joint", 1.6),
                                ("R_thumb_distal_joint", "R_thumb_proximal_pitch_joint", 2.4)] + \
            [(f"R_{f}_intermediate_joint", f"R_{f}_proximal_joint", 1.0) for f in ["index", "middle", "ring", "pinky"]]:
        if (child, parent) not in have:
            spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=child, name2=parent,
                              data=[0, mult, 0, 0, 0] + [0] * 6)
    # The URDF has no joint damping or rotor inertia, and its finger links weigh ~4 g: even a soft servo
    # snapped the index shut in 54 ms (the real hand takes ~0.5 s). Fingers then tunnelled into the tube
    # between physics steps and the solver threw it off the table (0/144 in the first sweep, 3 Nov).
    # Damping + armature give a ~0.4 s close, like the real geared finger drives.
    for j in spec.joints:
        if j.name.startswith("R_") and j.name.endswith("_joint"):
            # damping is a 3-vector in MuJoCo 3.17. The thumb yaw has no coupled joint (and moves the
            # whole thumb sideways): with the common value it still swung in 34 ms, so it gets more.
            dmp = INSPIRE_DAMPING * (3 if j.name == "R_thumb_proximal_yaw_joint" else 1)
            j.damping, j.armature = [dmp, 0, 0], 0.001
    # Torsional friction on the hand's contacts (condim 4). The Inspire's four fingers are stacked
    # vertically, so their contact normals on the vertical tube point sideways; with plain condim 3 nothing
    # resists the tube ROTATING about those normals, and in the carry it pivoted out of a 30-N grip
    # (3 Nov sweep). Real finger pads are soft and resist that twist; condim 4 + torsional friction
    # (the tube's own 0.01 m) models it. Only the Inspire geoms change: the Dex3 setup stays as validated.
    for g in spec.geoms:
        if g.contype and (g.name or "").startswith("") and _in_inspire(spec, g):
            g.condim = 4
    for j in INSPIRE_ACTS:                              # soft position servos, like the Dex3 (kp=20 there)
        lo, hi = spec.joint(j).range
        spec.add_actuator(name=j, target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrlrange=[lo, hi],
                          ctrllimited=True, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE, gainprm=[INSPIRE_KP] + [0] * 9,
                          biasprm=[0, -INSPIRE_KP, -0.05] + [0] * 7)


def _in_inspire(spec, g):
    """True if geom g belongs to the Inspire hand (its bodies are named R_...)."""
    b = g.parent
    return b is not None and b.name.startswith("R_")


INSPIRE_KP = 0.5        # finger servo stiffness (N m/rad): ~10 N at a fingertip at full squeeze
INSPIRE_DAMPING = 0.04  # N m s/rad (each joint; the coupled joint doubles it): index closes to 1.5 rad in ~0.45 s


def make():
    """Compile and return (model, data, stand_qpos) in the standing pose with servos holding it."""
    spec, stand_qpos = build()
    m = spec.compile()
    from paths import HAND
    if HAND == "inspire":
        # stand_qpos is in the Dex3 model's joint order; re-map it by joint NAME (Inspire fingers open = 0).
        ref = mujoco.MjModel.from_xml_path(str(MENAGERIE / "g1_with_hands.xml"))
        named = {ref.joint(j).name: stand_qpos[ref.jnt_qposadr[j] - 7] for j in range(1, ref.njnt)}
        stand_qpos = np.array([named.get(m.joint(j).name, 0.0) for j in range(m.njnt)
                               if m.joint(j).name != "cup_free"])
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
