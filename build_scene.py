"""Stage 1: build the MuJoCo world — a free-floating G1 Dex3-1 right hand, a cup, and a coaster.

WHY a floating hand instead of the full G1?
  The research question is about *when to call the big model*, not about balancing a humanoid.
  Driving the wrist directly (6 joints) removes 36 irrelevant DoF, makes sim ~10x faster, and keeps
  the action space small: 6 wrist + 7 finger = 13 numbers per step. This is the standard choice in
  dexterous-manipulation papers (DexMV, DexCap, etc.) before moving to a full arm.

Run:  python build_scene.py   ->  writes scene.xml next to this file
"""
from pathlib import Path
import mujoco

MENAGERIE = Path.home() / "mujoco_menagerie" / "unitree_g1"
OUT = Path(__file__).parent / "scene.xml"

FINGER_JOINTS = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]


def build() -> mujoco.MjSpec:
    g1 = mujoco.MjSpec.from_file(str(MENAGERIE / "g1_with_hands.xml"))
    hand_root = g1.body("right_wrist_yaw_link")  # palm geoms live on this body

    spec = mujoco.MjSpec()
    spec.meshdir = str(MENAGERIE / "assets")
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC   # better friction for grasps
    spec.option.impratio = 10                            # reduces slip in fingertip contacts
    # implicitfast: damping is integrated implicitly, so stiff servos on light links stay stable.
    # (With default Euler, kv*dt/inertia > ~2 on the wrist hinges diverged and the hand flew off.)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.compiler.degree = False

    w = spec.worldbody
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[.25, .3, .35],
                     rgb2=[.3, .35, .4], width=300, height=300)
    spec.add_material(name="grid", texrepeat=[8, 8], textures=["", "grid"])
    w.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, .05], material="grid")

    # Table top at z = 0.40 m
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.45, 0, 0.38],
               size=[0.30, 0.40, 0.02], rgba=[.55, .42, .3, 1])

    # Cup: a hollow-ish cylinder approximated by a solid cylinder (fine for grasping from the side).
    # ~7 cm diameter, 9 cm tall, 120 g — a typical small mug. Dex3-1 rated grasp object is 5 cm,
    # so ~7 cm is near the top of its comfortable range; we can shrink it if grasps fail.
    cup = w.add_body(name="cup", pos=[0.45, -0.10, 0.445])
    cup.add_freejoint(name="cup_free")
    cup.add_geom(name="cup_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.035, 0.045],
                 mass=0.12, friction=[1.0, 0.01, 0.001], rgba=[.9, .9, .95, 1])
    cup.add_site(name="cup_center", size=[0.005])

    # Coaster: thin static disc; "placed" = cup center within its radius and cup resting upright.
    w.add_geom(name="coaster", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0.45, 0.12, 0.402],
               size=[0.05, 0.002], rgba=[.2, .5, .3, 1], contype=0, conaffinity=0)
    w.add_site(name="coaster_center", pos=[0.45, 0.12, 0.404], size=[0.005])

    # Flying wrist: 3 slides + 3 hinges, all position-controlled.
    base = w.add_body(name="wrist_base", pos=[0.30, -0.10, 0.60])
    axes = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    parent = base
    for kind, ax in [("slide", "x"), ("slide", "y"), ("slide", "z"),
                     ("hinge", "x"), ("hinge", "y"), ("hinge", "z")]:
        b = parent.add_body(name=f"wrist_{kind}_{ax}")
        jt = mujoco.mjtJoint.mjJNT_SLIDE if kind == "slide" else mujoco.mjtJoint.mjJNT_HINGE
        rng = [-0.4, 0.4] if kind == "slide" else [-3.14, 3.14]
        b.add_joint(name=f"wrist_{kind}_{ax}", type=jt, axis=axes[ax], range=rng,
                    limited=True, damping=5 if kind == "slide" else 1, armature=0.01)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005], mass=0.05,
                   contype=0, conaffinity=0, rgba=[0, 0, 0, 0])
        parent = b

    # Graft the Dex3-1 subtree onto the flying wrist.
    frame = parent.add_frame()
    frame.attach_body(hand_root, "", "")

    # Actuators: wrist servos (stiff, so the hand goes where the policy says) + copy finger servos.
    for kind, ax in [("slide", "x"), ("slide", "y"), ("slide", "z"),
                     ("hinge", "x"), ("hinge", "y"), ("hinge", "z")]:
        kp = 2000 if kind == "slide" else 300
        a = spec.add_actuator(name=f"act_wrist_{kind}_{ax}", target=f"wrist_{kind}_{ax}",
                              trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=kp, kv=kp * 0.1)
        a.ctrlrange = [-0.4, 0.4] if kind == "slide" else [-3.14, 3.14]
        a.ctrllimited = True

    # The attach copied G1's wrist-yaw joint + actuator and the finger actuators.
    # Drop the yaw (our flying wrist already has yaw) and soften the finger servos:
    # G1's kp=500 is tuned for its real motors and launches a 120 g sim cup.
    spec.delete(spec.actuator("right_wrist_yaw_joint"))
    spec.delete(spec.joint("right_wrist_yaw_joint"))
    g1_ref = mujoco.MjModel.from_xml_path(str(MENAGERIE / "g1_with_hands.xml"))
    for j in FINGER_JOINTS:
        a = spec.actuator(j)
        # set_to_position() wipes ctrlrange, so restore it from an untouched compiled G1
        rng = list(g1_ref.actuator(j).ctrlrange)
        a.set_to_position(kp=20, kv=0.5)
        a.ctrlrange = rng
        a.ctrllimited = True

    # Fingertip sites — the retargeting targets and the contact-check anchors.
    # Positions are the farthest collision-mesh vertex of each distal link (measured, ~5.2 cm out),
    # pulled back 4 mm so the site sits on the finger pad that actually touches the cup.
    tip_offsets = {"right_hand_thumb_2_link": [-0.003, 0.048, -0.002],
                   "right_hand_index_1_link": [0.048, -0.003, 0.002],
                   "right_hand_middle_1_link": [0.048, -0.003, 0.002]}
    for link, off in tip_offsets.items():
        spec.body(link).add_site(name=link.replace("right_hand_", "").replace("_link", "_tip"),
                                 pos=off, size=[0.006], rgba=[1, .2, .2, 1])
    spec.body("right_wrist_yaw_link").add_site(name="palm_center", size=[0.006], rgba=[.2, .2, 1, 1])
    return spec


if __name__ == "__main__":
    spec = build()
    model = spec.compile()
    OUT.write_text(spec.to_xml())
    print(f"wrote {OUT}  nq={model.nq} nu={model.nu}")
    print("actuators:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)])
