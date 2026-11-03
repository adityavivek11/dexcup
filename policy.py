"""Shared policy code: observation, action encoding, dataset, teacher and student networks.

ONE observation function for both training data and live rollouts: if they differ even slightly
(units, ordering, a missing feature), a policy that looks great on its validation loss fails in the sim.

Observation (state, 31 numbers), all from the robot's own state + object poses (no contact sensing):
  palm_pos 3 | cup_pos 3 | cup_up 3 (tube axis: is it tipping?) | coaster_xy 2 | finger_q 7 | arm_q 7 |
  cup - palm 3 | coaster - cup 3
  WHY no "grasped" / contact flags: they come from contact sensing. The router must predict phases from
  observations only (CLAUDE.md), and the policies shouldn't lean on them either.

Action, per tick: palm target RELATIVE to an ANCHOR object (3, metres) + grip (1: 1 = fingers closed).
  Anchor = the tube while the hand is open (approach, grasp, retreat), the coaster while it is closed
  (carry, place); see anchor(). WHY object-relative (teacher v1-v4 diagnosis): with targets relative to the
  CURRENT PALM, "move 2 cm" shrinks to ~0 near every goal, small errors accumulate, and the teacher stopped
  short of the grasp point (7/9 held-out in v4), never closed, or set the tube down 2-3 cm off-centre.
  Relative to the object, the instruction at the goal is a constant ("tube + grasp offset", "coaster +
  place offset"): the policy settles AT the goal, and it still generalises across layouts (no absolute
  positions to memorise). The hand orientation is fixed (HAND_QUAT), as in all demos.
Teacher predicts a CHUNK of H future actions (ACT-style); the student predicts one step.

RECIPE = per-model settings stored on the model as model.cfg (so the sim decodes every model correctly,
and old and new models can't be mixed up):
  frame: "palm"   actions relative to the current palm (teacher v1-v4)
         "object" actions relative to the tube (hand open) / coaster (hand closed) (teacher v5)
  vel:   also observe the 7 arm joint velocities (obs 31 -> 38). WHY (3 Nov): every teacher so far got
         stuck where POSITION alone can't tell "arriving" from "leaving" (waypoints, grasp point,
         release); motion information is standard in behaviour-cloning setups (ACT, Diffusion Policy).
Models saved without cfg (v1-v5) are palm-frame without velocity, except v5 (object frame).
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent
H = 16                       # chunk length: 16 ticks = 0.32 s
OBS_DIM, ACT_DIM = 31, 4      # base observation; +VEL_DIM when cfg["vel"]
VEL_DIM = 7
DEFAULT_CFG = {"frame": "palm", "vel": False}
REPLAYED = ROOT / "data" / "replayed"
VAL_EVERY = 5                # hold-out: every 5th successful episode (sorted by clip name) -> ~20%


def quat_up(q):
    """z axis of a body given its wxyz quaternion(s): (N,4) -> (N,3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)


def make_obs(palm_pos, cup_pos, cup_quat, coaster_pos, finger_q, arm_q, arm_dq=None):
    """Works on single steps (1-D inputs) or whole episodes (2-D). arm_dq given -> 7 velocities appended
    at the END (indices 31-37), so the first 31 entries mean the same for every recipe."""
    a = [np.atleast_2d(v) for v in (palm_pos, cup_pos, cup_quat, coaster_pos, finger_q, arm_q)]
    palm, cup, cq, co, fq, aq = a
    parts = [palm, cup, quat_up(cq), co[:, :2], fq, aq, cup - palm, co - cup]
    if arm_dq is not None:
        parts.append(np.atleast_2d(arm_dq))
    return np.concatenate(parts, 1).astype(np.float32)


def obs_dim(cfg):
    return OBS_DIM + (VEL_DIM if cfg["vel"] else 0)


def ref_point(obs, palm, frame):
    """What the actions are relative to: the current palm, or the anchor object (see anchor())."""
    return np.atleast_2d(palm).astype(np.float32) if frame == "palm" else anchor(obs)


GRIP_Q = 0.35               # index_0 angle (rad) above which the hand counts as closed. Measured on the
                            # demos: open <= 0.21 (99%), closed and gripping the tube >= 0.51.


def anchor(obs):
    """Reference point for the action, from the observation only (no contact sensing): the tube centre
    while the hand is open, the coaster while it is closed. obs (N,31) or (31,) -> (N,3)."""
    o = np.atleast_2d(obs)
    cup = o[:, 3:6]
    coaster = cup + o[:, 28:31]                     # obs stores (coaster - cup); recover the coaster itself
    closed = o[:, 11 + 3] > GRIP_Q                  # finger_q starts at index 11; index_0 is its 4th joint
    return np.where(closed[:, None], coaster, cup).astype(np.float32)


GRIP_LEAD = 0              # ticks the grip label leads the executed grip. Tried 8 (teacher v4): held-out
                           # success fell 2/9 -> 1/9 and 7/9 then stopped ~2.5 cm short of the grasp point,
                           # so reverted to 0 (= the v3 recipe). Kept as a knob, documented below.


def episode_arrays(e, cfg=DEFAULT_CFG):
    """(obs (T,31|38), actions (T,4), reference points (T,3)) from a recorded episode, for recipe cfg.
    WHY the grip label leads the executed grip (teacher v3 held-out diagnosis): in the demos the hand eases
    to a stop at the grasp point and only THEN closes, so the state "palm at the grasp point, fingers open"
    appears ~10 ticks labelled open and 1-2 labelled closed. The teacher learned the majority: never close
    (4/9 held-out episodes: palm parked at the grasp pose, hand open). Release had the mirror problem
    (2/9: tube on the roll, never let go). Shifting the switch 0.16 s earlier puts the decision on states
    the policy CAN tell apart (palm still a few mm out, the tube-palm vector says "almost there"), like a
    person closing while still moving. Measured: the palm moves ~4.4 mm in those 0.16 s, inside the +-5 mm
    grasp tolerance, and the fingers need a few tenths of a second to close, so the palm arrives first."""
    obs = make_obs(e["palm_pos"], e["cup_pos"], e["cup_quat"], e["coaster_pos"], e["finger_q"], e["arm_q"],
                   e["arm_dq"] if cfg["vel"] else None)
    ref = ref_point(obs, e["palm_pos"], cfg["frame"])
    rel = e["action"][:, :3] - ref                              # palm target relative to the reference
    grip = (e["action"][:, 7 + 3] > 0.5).astype(np.float32)     # index_0 target 1.2 when closed, 0 open
    if GRIP_LEAD:
        grip = np.r_[grip[GRIP_LEAD:], np.repeat(grip[-1], GRIP_LEAD)]
    return obs, np.column_stack([rel, grip]).astype(np.float32), ref


def chunk_targets(act, anc, H=H):
    """For each t: the next H actions, positions RELATIVE TO THE ANCHOR AT t (the moment the chunk is
    predicted; anc = the reference point per step, palm or anchor), padded with the last action.
    BUG FIXED (3 Nov): this first stacked the per-step relative actions, each relative to the palm at its
    OWN future step, while TeacherAgent adds the whole chunk to the palm at prediction time. The blended
    targets then lagged behind the hand (rollout palm speed 35 vs 54 mm/s in the demo) and the teacher
    stalled on all 9 held-out layouts."""
    tgt = act[:, :3] + anc                                        # absolute palm targets
    tgt = np.concatenate([tgt, np.repeat(tgt[-1:], H, 0)])
    grip = np.concatenate([act[:, 3], np.repeat(act[-1:, 3], H)])
    T = len(act)
    pos = np.stack([tgt[t:t + H] - anc[t] for t in range(T)])
    return np.concatenate([pos, np.stack([grip[t:t + H] for t in range(T)])[..., None]], -1).astype(np.float32)


DART = ROOT / "data" / "dart"


SPLIT = ROOT / "data" / "split.json"


def split(dart=False, fold=None, folds=5):
    """Successful episodes, split BEFORE any training (no peeking at results). The validation CLIPS are
    frozen in data/split.json (every 5th successful replay of the first replay run): re-running replays can
    change which clips succeed, and "every 5th" would then silently pick a different validation set.
    dart=True adds the noise-injected replays (replay.py --noise) of TRAINING clips only: a noisy copy of
    a validation clip in the training set would leak its layout into training."""
    import json
    eps = sorted(p for p in REPLAYED.glob("*.npz") if bool(np.load(p)["success"]))
    if fold is not None:
        # k-fold cross-validation (user's suggestion, 3 Nov): 9 held-out episodes give ~+-14% standard error,
        # too noisy to compare teachers. Fold k holds out every folds-th clip starting at k, so across the
        # folds EVERY clip is tested exactly once by a teacher that never saw it.
        val = eps[fold::folds]
    elif SPLIT.exists():
        vset = set(json.loads(SPLIT.read_text())["val"])
        val = [p for p in eps if p.stem in vset]
    else:
        val = eps[::VAL_EVERY]
    train = [p for p in eps if p not in val]
    if dart:
        clips = {p.stem for p in train}
        train += sorted(p for p in DART.glob("*.npz")
                        if p.stem.rsplit("_n", 1)[0] in clips and bool(np.load(p)["success"]))
    return train, val


def load_set(paths, cfg=DEFAULT_CFG):
    """(obs, actions, phase labels, reference points) per episode, for recipe cfg."""
    O, A, P, refs = [], [], [], []
    for p in paths:
        e = np.load(p)
        o, a, r = episode_arrays(e, cfg)
        O.append(o); A.append(a); P.append(e["phase_auto"]); refs.append(r)
    return O, A, P, refs


class Norm(nn.Module):
    """Per-feature standardisation, stored in the model so deployment uses the training statistics."""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(np.maximum(std, 1e-3), dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


# Feature groups -> teacher tokens (indices into the 31-dim observation)
GROUPS = [range(0, 3), range(3, 9), range(9, 11), range(11, 18), range(18, 25), range(25, 31)]
# palm | cup pos+up | coaster | fingers | arm | relative vectors


class Teacher(nn.Module):
    """ACT-style transformer (no CVAE): observation split into semantic tokens, H learned query tokens
    cross-attend to them and decode the action chunk. 7.4M parameters (measured): much bigger than the
    student, so the router's latency savings are real, not a rounding error."""
    def __init__(self, norm, d=256, layers=4, heads=8, cfg=DEFAULT_CFG):
        super().__init__()
        self.cfg = dict(cfg)
        self.groups = [list(g) for g in GROUPS] + ([list(range(OBS_DIM, OBS_DIM + VEL_DIM))] if cfg["vel"] else [])
        self.norm = norm
        self.embed = nn.ModuleList(nn.Linear(len(g), d) for g in self.groups)
        self.tok_pos = nn.Parameter(torch.randn(len(self.groups), d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, layers)
        self.queries = nn.Parameter(torch.randn(H, d) * 0.02)
        self.head = nn.Linear(d, ACT_DIM)

    def forward(self, obs):                       # (B,31) -> (B,H,4); grip channel = logit
        x = self.norm(obs)
        groups = getattr(self, "groups", GROUPS)        # models saved before cfg existed use GROUPS
        tok = torch.stack([emb(x[:, list(g)]) for emb, g in zip(self.embed, groups)], 1) + self.tok_pos
        mem = self.encoder(tok)
        q = self.queries.unsqueeze(0).expand(len(obs), -1, -1)
        return self.head(self.decoder(q, mem))


class Student(nn.Module):
    """Small MLP: the cheap model the router uses outside grasp/place.

    chunk=True (default since 3 Nov): predicts the same H-step chunk as the teacher and is run with the same
    temporal ensembling. WHY: one-step students were unstable in closed loop. With arm velocities in the
    input, the one-step MLP learned "next move ~ current velocity" (the copycat problem) and its targets ran
    away to metres within 10 ticks; without velocities it drifted and never grasped (0/6 training layouts).
    The chunked student grasped 4/6 from behaviour cloning alone: predicting 16 steps ahead forces it to
    model where the motion is GOING, and blending overlapping chunks smooths single-tick errors.
    The student does NOT get velocities even when the teacher does (each agent slices its own obs)."""
    def __init__(self, norm, hidden=128, cfg=DEFAULT_CFG, chunk=False):
        super().__init__()
        self.cfg = dict(cfg)
        self.norm = norm
        self.chunk = chunk
        self.net = nn.Sequential(nn.Linear(obs_dim(cfg), hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, ACT_DIM * (H if chunk else 1)))

    def forward(self, obs):                       # (B,D) -> (B,4) or (B,H,4) if chunked; grip channel = logit
        out = self.net(self.norm(obs))
        return out.view(len(obs), H, ACT_DIM) if getattr(self, "chunk", False) else out


class PhaseNet(nn.Module):
    """Router's phase classifier (obs -> 5 phase logits). Lives here, not in router.py: a model saved
    while router.py runs as a script would be pickled as __main__.PhaseNet and fail to load elsewhere."""
    def __init__(self, norm, hidden=64):
        super().__init__()
        self.norm = norm
        self.net = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 5))

    def forward(self, obs):
        return self.net(self.norm(obs))


POS_SCALE = 100.0           # loss in cm for positions (metres are tiny numbers). A plain float, not a tensor,
                            # so the loss works on CPU and GPU alike.


def action_loss(pred, target):
    """Position error in cm (MSE) + grip as binary cross-entropy (it is a yes/no decision)."""
    pos = ((pred[..., :3] - target[..., :3]) * POS_SCALE).pow(2).mean()
    grip = nn.functional.binary_cross_entropy_with_logits(pred[..., 3], target[..., 3])
    return pos + grip, pos.detach(), grip.detach()
