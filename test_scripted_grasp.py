"""Sanity test: a hand-coded pick-and-place must work before any learning.

WHY: if the hand physically can't hold the cup (too big, too slippery, servos too weak), every later
stage fails for reasons that look like "the policy is bad." This test isolates the physics.
It also gives us our first phase labels: approach -> grasp -> transport -> place -> release.
Those phases are exactly what the teacher/student router will key on later.
"""
import mujoco
import numpy as np
from build_scene import build
from contacts import GraspChecker

m = build().compile()
d = mujoco.MjData(m)
gc = GraspChecker(m)

WRIST = ["act_wrist_slide_x", "act_wrist_slide_y", "act_wrist_slide_z"]
FINGERS_CLOSE = {"right_hand_index_0_joint": 1.2, "right_hand_index_1_joint": 1.3,
                 "right_hand_middle_0_joint": 1.2, "right_hand_middle_1_joint": 1.3,
                 "right_hand_thumb_2_joint": -1.3}
# Palm starts at (0.346,-0.10,0.595). The closing "C" is ~(+0.10,+0.07) from the palm, but the OPEN
# thumb must clear the cup on the way down, so the palm goes to ~(0.35,-0.15) and z~0.45 so both
# index (+3 cm) and middle (-3 cm) land on the cup body. Slides are offsets from the start pose.
WAYPOINTS = [  # (phase, wrist xyz offset, fingers closed?, seconds)
    ("approach_above", [0.00, -0.05, 0.00], False, 1.0),
    ("approach_down",  [0.00, -0.05, -0.145], False, 1.0),
    ("grasp",          [0.00, -0.05, -0.145], True, 1.0),
    ("lift",           [0.00, -0.05, -0.05], True, 1.0),
    ("transport",      [0.00,  0.17, -0.05], True, 1.5),
    ("place",          [0.00,  0.17, -0.14], True, 1.0),
    ("release",        [0.00,  0.17, -0.14], False, 0.8),
    ("retreat",        [0.00,  0.17, 0.00], False, 0.8),
]


def run(verbose=True):
    log = []
    mujoco.mj_resetData(m, d)
    cur = np.zeros(3)
    for phase, target, closed, secs in WAYPOINTS:
        target = np.array(target)
        steps = int(secs / m.opt.timestep)
        start = cur.copy()
        for k in range(steps):
            a = min(1.0, k / (0.6 * steps))       # move during first 60%, then settle
            cur = start + a * (target - start)
            for i, name in enumerate(WRIST):
                d.ctrl[m.actuator(name).id] = cur[i]
            for j, q in FINGERS_CLOSE.items():
                d.ctrl[m.actuator(j).id] = q if closed else 0.0
            mujoco.mj_step(m, d)
        ok, f = gc.is_grasped(d)
        cup = d.site("cup_center").xpos.copy()
        log.append((phase, ok, cup))
        if verbose:
            print(f"{phase:15s} grasp={ok!s:5s} cup={cup.round(3)}  forces="
                  + " ".join(f"{k}:{v:.2f}" for k, v in f.items()))
    coaster = d.site("coaster_center").xpos
    dist_xy = np.linalg.norm(d.site("cup_center").xpos[:2] - coaster[:2])
    zaxis = d.body("cup").xmat.reshape(3, 3)[:, 2]
    upright = zaxis[2] > 0.95
    success = dist_xy < 0.03 and upright
    if verbose:
        print(f"\ncup-to-coaster xy distance = {dist_xy*100:.1f} cm, upright={upright}  ->  SUCCESS={success}")
    return success, log


if __name__ == "__main__":
    run()
