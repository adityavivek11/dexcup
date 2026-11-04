"""Hand-specific constants, so the rest of the pipeline is identical for the Dex3-1 and the Inspire hand.

FINGER_JOINTS  the 7 joint angles that go into the policy observation. Kept at 7 for BOTH hands so the
               observation layout (policy.make_obs, 31/38 dims) and every downstream script stay unchanged.
               Inspire: the 6 driven joints + the thumb's coupled intermediate joint. Ring/pinky follow the
               same grip command as index/middle, so their angles add no information for this task.
               obs index 14 (= FINGER_JOINTS[3]) must be a finger that closes on the tube: policy.anchor
               reads it to tell an open hand from a closed one.
FINGER_ACTS    actuators commanded each tick (Dex3: 7, Inspire: 6 motors).
OPEN / CLOSE   actuator targets for the binary grip (single grip scalar).
FINGER_LINKS   which links count as thumb / opposing fingers in the grasp check (contacts.py).
OFF, CLEAR     palm (wrist_yaw origin) offset from the tube centre at the grasp, and the sideways clearance
               for the approach. MEASURED by sweeping test_g1_grasp, never guessed.
"""
import numpy as np
from paths import HAND

if HAND == "dex3":
    FINGER_JOINTS = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                     "right_hand_index_0_joint", "right_hand_index_1_joint",
                     "right_hand_middle_0_joint", "right_hand_middle_1_joint"]
    FINGER_ACTS = FINGER_JOINTS
    OPEN = dict.fromkeys(FINGER_ACTS, 0.0)
    CLOSE = {**OPEN, "right_hand_index_0_joint": 1.2, "right_hand_index_1_joint": 1.3,
             "right_hand_middle_0_joint": 1.2, "right_hand_middle_1_joint": 1.3,
             "right_hand_thumb_2_joint": -1.3}
    FINGER_LINKS = {"thumb": ["right_hand_thumb_2_link"], "index": ["right_hand_index_1_link"],
                    "middle": ["right_hand_middle_1_link"]}
    OFF = np.array([-0.13, -0.0625, 0.0])
    CLEAR = np.array([0, -0.03, 0])
    UP = 0.12                                   # approach/retreat height above the grasp point
    BACK = CLEAR                                # sideways back-off after the release, before lifting
    GRIP_THRESHOLD = 0.5                        # close the hand when the policy's grip probability > this
else:
    FINGER_JOINTS = ["R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_thumb_intermediate_joint",
                     "R_index_proximal_joint", "R_middle_proximal_joint", "R_ring_proximal_joint",
                     "R_pinky_proximal_joint"]
    FINGER_ACTS = ["R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint",
                   "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint"]
    # PRE-SHAPED grasp (3 Nov sweeps). The Dex3-style side grasp (palm against the tube, slide in from
    # the side) failed 0/144: the Inspire's thumb base sticks out of the palm where the tube would rest,
    # so the tube was squeezed out sideways at the lift. Instead the thumb is swung across the palm
    # (yaw 1.3) and kept straight while open: thumb and straight fingers then enclose a tube-sized space,
    # the hand comes STRAIGHT DOWN over the tube (CLEAR = 0), and closing curls the fingers round the
    # front and the thumb round the back. On release the hand reopens to the same shape and lifts off.
    OPEN = {**dict.fromkeys(FINGER_ACTS, 0.0), "R_thumb_proximal_yaw_joint": 1.3,
            "R_thumb_proximal_pitch_joint": -0.1}
    # Closed targets are deliberately PAST where the fingers meet the 9.4 cm tube: the soft servo then
    # presses with kp * (target - angle). Thumb curl 0.3 and 0.6 both worked; 0.3 squeezes less.
    # Ring and pinky close only LIGHTLY (0.6). With all four fingers at 1.7 the finger forces were centred
    # ~2 cm below the thumb's contact; thumb (behind) and fingers (front) then formed a couple that tipped
    # the tube ~30 deg within 4 s of holding (0/82 replays: tube set down leaning, then toppled). Ring/pinky
    # at 0.6 still touch (~1.5 N) but the squeeze comes from thumb + index + middle at matching heights:
    # uprightness after a 4 s hold 0.86 -> 0.994 (ring/pinky fully open: 0.999). Like a human bottle grip.
    CLOSE = {"R_thumb_proximal_yaw_joint": 1.3, "R_thumb_proximal_pitch_joint": 0.3,
             "R_index_proximal_joint": 1.7, "R_middle_proximal_joint": 1.7,
             "R_ring_proximal_joint": 0.6, "R_pinky_proximal_joint": 0.6}
    FINGER_LINKS = {"thumb": ["R_thumb_distal", "R_thumb_intermediate"],
                    "index": ["R_index_intermediate"], "middle": ["R_middle_intermediate"],
                    "ring": ["R_ring_intermediate"], "pinky": ["R_pinky_intermediate"]}
    # Swept 3 Nov (kp 0.5, palm at the tube's mid-height): success for x in [-0.195, -0.185] and y in
    # [-0.06, -0.05]; x = -0.19 worked for every y from -0.045 to -0.06. x >= -0.175 holds the tube but the
    # release drags it off; x = -0.205 fails. ~1 cm x window, like the Dex3's. Centre of the region:
    OFF = np.array([-0.19, -0.055, 0.0])
    CLEAR = np.array([0.0, 0.0, 0.0])           # straight-down approach over the tube (see OPEN above)
    # The fingers reach 24 cm ahead of the wrist and pass OVER the tube on the way in: at 12 cm the lowest
    # finger cleared the tube top by only ~6 mm, so the Inspire approaches from 15 cm.
    UP = 0.15
    # After the release the open, pre-shaped thumb rests against the back of the tube: lifting straight up
    # dragged it along the tube until it caught the rim and flipped it (0/82 replays, 3 Nov). So back off
    # 3 cm toward the palm side first (pulls thumb and straight fingers off the tube), then lift.
    BACK = np.array([0.0, -0.03, 0.0])
    # Grip decision threshold, chosen on TRAINING layouts only (12 layouts, 3 Nov): the Inspire teacher
    # deadlocked with the tube resting on the coaster 1.4-2.4 cm off centre, hand still, grip probability
    # frozen at 0.52-0.62 (demos released at 0.8-1.3 cm). Success at 0.5/0.6/0.7/0.8/0.9: 7/8/10/8/2 of 12;
    # higher values also delay the grasp close. A calibration of an uncertain decision, NOT a fix of the
    # teacher's uncertainty (disclosed in the README). Dex3 keeps 0.5.
    GRIP_THRESHOLD = 0.7
