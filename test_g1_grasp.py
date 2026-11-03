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
from build_g1_scene import make, CUP_START, COASTER, RIGHT_ARM, COASTER_HALF_H
from contacts import GraspChecker
from ik import ArmIK, reset_ready

CTRL_DT = 0.02
m, d, STAND = make()
SUBSTEPS = int(round(CTRL_DT / m.opt.timestep))
ik = ArmIK(m)
gc = GraspChecker(m)
ARM_ACT = [m.actuator(j).id for j in RIGHT_ARM]
# ALL finger motors are commanded explicitly in both states. Leaving any out means it inherits the
# reset pose: G1's 'stand' curls thumb_1 to -1.05, which parked the thumb inside the grasp space.
# Hand-specific values live in hands.py (Dex3-1 or Inspire, chosen by paths.HAND).
import hands as Hd
FINGER_JOINTS, FINGER_ACTS = Hd.FINGER_JOINTS, Hd.FINGER_ACTS
# Targets per joint: motors from hands.OPEN/CLOSE; a coupled joint in FINGER_JOINTS (Inspire thumb
# intermediate = 1.6 x thumb pitch) gets the angle its motor's target implies, so the recorded 7-value
# finger action has the same layout for both hands.
_COUPLED = {"R_thumb_intermediate_joint": ("R_thumb_proximal_pitch_joint", 1.6)}
_full = lambda t: {**t, **{j: t[p] * k for j, (p, k) in _COUPLED.items() if j in FINGER_JOINTS}}
FINGERS_OPEN, FINGERS_CLOSE = _full(Hd.OPEN), _full(Hd.CLOSE)
HAND_QUAT = np.array([1.0, 0, 0, 0])            # fingers forward (+x), hand closes toward +y
# Palm relative to cup centre, for the 9.4 cm tube. Swept 2 Nov: success for x in [-0.135,-0.125] and
# y in [-0.075,-0.05]; x=-0.145 holds but fails the place, x<=-0.115 knocks the tube. Centre of the region.
# (The 7 cm test cup used (-0.104,-0.05).) The x window is only ~1 cm wide: retargeted human demos will
# need the +-1 cm repair step.
OFF = Hd.OFF
# Resting on the tape roll, the cup centre sits one roll-height higher than on the table.
CUP_ON_COASTER = np.array([COASTER[0], COASTER[1], CUP_START[2] + 2 * COASTER_HALF_H])

BACK = Hd.BACK                                  # sideways back-off after the release
UP = Hd.UP                                      # approach / retreat height above the grasp point
CLEAR = Hd.CLEAR                                # descend beside the cup (3 cm away), then slide in


def waypoints(off=OFF, clear=CLEAR):
    """(phase, palm target in world, fingers closed?, seconds). A function so offsets can be swept."""
    return [
        # Come down BESIDE the cup with clearance, then slide in sideways (like a person does).
        # Descending straight onto the grasp pose put the open thumb on the rim and blocked the arm.
        ("approach_above", CUP_START + off + clear + [0, 0, UP], False, 1.2),
        ("approach_down",  CUP_START + off + clear,                False, 1.0),
        ("slide_in",       CUP_START + off,                        False, 0.8),
        ("grasp",          CUP_START + off,                        True,  0.8),
        ("lift",           CUP_START + off + [0, 0, 0.09],         True,  0.8),
        ("transport",      CUP_ON_COASTER + off + [0, 0, 0.09],    True,  1.5),
        ("place",          CUP_ON_COASTER + off + [0, 0, 0.005],   True,  1.0),
        ("release",        CUP_ON_COASTER + off + [0, 0, 0.005],   False, 0.6),
        ("back_off",       CUP_ON_COASTER + off + BACK + [0, 0, 0.005], False, 0.6),
        ("retreat",        CUP_ON_COASTER + off + BACK + [0, 0, UP],    False, 0.8),
    ]


WAYPOINTS = waypoints()


def run(verbose=True, record=False, off=OFF):
    # Start in the 'ready' pose (hand raised above the table). Starting from the hanging-arm stand pose
    # swept the fingers through the table edge and the hip on the way up.
    reset_ready(m, d, STAND, ik)
    frames, log = [], []
    renderer = mujoco.Renderer(m, 240, 320) if record else None
    cur = d.site("palm_center").xpos.copy()
    for phase, target, closed, secs in waypoints(off):
        start, ticks = cur.copy(), int(secs / CTRL_DT)
        for k in range(ticks):
            a = min(1.0, k / (0.7 * ticks))
            a = a * a * (3 - 2 * a)                       # smoothstep: no velocity jumps at waypoints
            cur = start + a * (np.asarray(target) - start)
            arm_q, _, _ = ik.solve(d.qpos, cur, HAND_QUAT, iters=30)
            d.ctrl[ARM_ACT] = arm_q
            for j in FINGER_ACTS:
                d.ctrl[m.actuator(j).id] = (FINGERS_CLOSE if closed else FINGERS_OPEN)[j]
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
