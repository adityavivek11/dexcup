"""Stage 8: the router. Predicts the phase ONLINE from the observation, calls the teacher only in
grasp/place, the student otherwise. Plus the random router, the key control.

WHY a learned phase classifier (not contact sensing)?
  At deployment there is no ground-truth phase. The router may only use what the policy sees (the
  31-d observation). Phase LABELS (from physics rules, gen_demos.label_phases) are used for TRAINING
  only, never read at test time.

WHY the random router must match segment length, not just the call fraction:
  A router that flips teacher/student every tick would be handicapped for a reason unrelated to WHEN it
  calls the teacher (the teacher's chunk blending needs consecutive calls). So the random router is a
  two-state Markov chain with the SAME teacher-call fraction p AND the same mean teacher-segment length
  L as the real router: switch to teacher with prob a = p/(L(1-p)), back with prob b = 1/L. The only
  remaining difference from the real router is the timing: that is exactly what the control isolates.

Run:  python router.py          -> models/phase.pt (trained on the training episodes, checked on val)
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from policy import split, load_set, Norm, PhaseNet

ROOT = Path(__file__).parent
from paths import MODELS                # per-hand output folder (paths.py)
TEACHER_PHASES = (1, 3)                      # grasp, place


def train_phase(epochs=60):
    torch.manual_seed(0)
    tr, va = split()
    O, _, P, _ = load_set(tr); Ov, _, Pv, _ = load_set(va)
    X, Y = torch.from_numpy(np.concatenate(O)), torch.from_numpy(np.concatenate(P)).long()
    Xv, Yv = torch.from_numpy(np.concatenate(Ov)), torch.from_numpy(np.concatenate(Pv)).long()
    net = PhaseNet(Norm(X.mean(0).numpy(), X.std(0).numpy()))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 512):
            b = perm[i:i + 512]
            x = X[b] + torch.randn_like(X[b]) * net.norm.std * 0.05
            loss = nn.functional.cross_entropy(net(x), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pv = net(Xv).argmax(1)
    acc = (pv == Yv).float().mean().item()
    t_true = torch.isin(Yv, torch.tensor(TEACHER_PHASES)); t_pred = torch.isin(pv, torch.tensor(TEACHER_PHASES))
    agree = (t_true == t_pred).float().mean().item()
    print(f"phase net: val accuracy {acc:.0%} | teacher-or-not decision agrees with labels {agree:.0%} "
          f"| teacher fraction labels {t_true.float().mean():.0%}, predicted {t_pred.float().mean():.0%}")
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(net, MODELS / "phase.pt")
    return net


class RouterAgent:
    """mode 'router': teacher iff the phase net predicts grasp/place. mode 'random': Markov chain with the
    router's teacher-call fraction p and mean teacher-segment length L (see module doc)."""
    def __init__(self, teacher, student, mode="router", phase=None, p=None, L=None, seed=0):
        from sim_env import TeacherAgent, StudentAgent
        self.teacher, self.student = TeacherAgent(teacher), StudentAgent(student)
        self.mode, self.phase = mode, phase
        if mode.startswith("random"):
            self.b = 1.0 / L
            self.a = p / (L * (1 - p))
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.teacher.reset(); self.student.reset()
        self.use_teacher, self.teacher_calls, self.segments = False, 0, 0

    def __call__(self, obs, palm, t):
        if self.mode == "router":
            with torch.no_grad():
                ph = int(self.phase(torch.from_numpy(obs[:31])[None]).argmax(1))   # phase net: base obs
            want = ph in TEACHER_PHASES
        else:
            r = self.rng.random()
            want = (r >= self.b) if self.use_teacher else (r < self.a)
        if want and not self.use_teacher:
            self.teacher.reset()                 # stale chunks from an earlier segment must not be blended in
            self.segments += 1
        if not want and self.use_teacher:
            self.student.reset()                 # same for the (chunked) student
        self.use_teacher = want
        if want:
            self.teacher_calls += 1
            return self.teacher(obs, palm, t)
        return self.student(obs, palm, t)


if __name__ == "__main__":
    train_phase()
