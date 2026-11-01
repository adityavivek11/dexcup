"""Stage 1b: fingertip-contact grasp confirmation.

WHY confirm grasps with contact at all?
  Retargeting from a phone video is noisy: the human hand "holds" the cup in the video, but after
  mapping onto a different-sized robot hand the fingers may pass through the cup or miss it by 1 cm.
  If we train on those frames, the teacher learns to close its hand on air. So every demo is replayed
  in MuJoCo and we keep only grasp frames where the robot is *physically* holding the cup.

Rule (Dex3-1 has 3 fingers): grasp is confirmed when
  - the thumb distal link touches the cup, AND
  - at least one of index/middle distal links touches the cup, AND
  - total normal force from those contacts exceeds a small threshold (rules out grazing touches).
This is the classic "opposition" condition: two opposing contacts are the minimum for a stable pinch.
"""
import mujoco
import numpy as np

FINGER_LINKS = {
    "thumb": ["right_hand_thumb_2_link"],
    "index": ["right_hand_index_1_link"],
    "middle": ["right_hand_middle_1_link"],
}
MIN_NORMAL_FORCE = 0.5  # newtons; cup weighs ~1.2 N, so two contacts at >=0.5 N with friction can hold it


class GraspChecker:
    def __init__(self, model: mujoco.MjModel, object_geom: str = "cup_geom"):
        self.m = model
        self.obj = model.geom(object_geom).id
        self.body_to_finger = {model.body(b).id: f for f, bs in FINGER_LINKS.items() for b in bs}

    def finger_forces(self, data: mujoco.MjData) -> dict:
        """Sum of normal force each finger applies to the object this step."""
        forces = {f: 0.0 for f in FINGER_LINKS}
        f6 = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            if self.obj not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.obj else c.geom1
            finger = self.body_to_finger.get(self.m.geom_bodyid[other])
            if finger is None:
                continue
            mujoco.mj_contactForce(self.m, data, i, f6)
            forces[finger] += abs(f6[0])  # f6[0] is the normal component in the contact frame
        return forces

    def is_grasped(self, data: mujoco.MjData) -> tuple[bool, dict]:
        f = self.finger_forces(data)
        thumb = f["thumb"] > MIN_NORMAL_FORCE
        opposing = max(f["index"], f["middle"]) > MIN_NORMAL_FORCE
        return bool(thumb and opposing), f
