"""Stage 7: distil the teacher into a small student with DAgger.

WHY DAgger instead of just copying the demos (behaviour cloning)?
  A student trained only on demo states has never seen its own mistakes. In the sim, its small errors
  push it into states that are slightly off the demos, where it was never taught what to do, so errors
  compound. DAgger fixes this: let the student drive, record the states it actually visits, ask the
  teacher what it would have done there, add those (state, teacher action) pairs to the data, retrain.
  After a few rounds the student has been taught on its own state distribution.

Student recipe (3 Nov, after the one-step students failed, see policy.Student): CHUNKED output (same H-step
chunks + temporal ensembling as the teacher), NO velocity input (the teacher keeps its own recipe), and
DAgger labels = the teacher's raw chunk at each visited state. Normalisation statistics are fixed from the
demos: the first velocity student's rollouts ran away and refitting the stats on that data inflated them
~50x, so every round's student saw a distorted input scale.

Rounds:
  0  behaviour cloning on the demos + DART replays (chunk targets) -> student v0
  k  roll out a mix (teacher drives with probability beta_k, else the student; beta decays), label every
     visited state with the teacher's action, aggregate, retrain from scratch on everything.
  Rollouts use the TRAINING layouts only (the val/random eval layouts stay unseen).

--router (3 Nov, after the first 100-layout eval): DAgger INSIDE the router. Each rollout is driven by the
real RouterAgent (phase net: teacher in grasp/place, student elsewhere), starting from the current
models/student.pt; a separate copy of the teacher labels every tick the STUDENT drives. WHY: with
per-tick beta mixing, teacher and student swap control every tick, a jerkier state distribution than
deployment, and the router (42%) was no better than a random router (37%) because the student (2% alone)
failed in its own segments. Router-DAgger trains the student on exactly the states it meets when deployed:
after a teacher-run grasp, mid-carry, after release. Layouts: half training clips, half fresh uniform
layouts from eval_policy.layouts with seeds 1000+ (the 100 eval layouts use seed 0: never trained on).

Run:  python train_student.py [--teacher models/teacher_pv.pt --rounds 4 --rollouts 24]  -> models/student.pt
      python train_student.py --router --rounds 5 --rollouts 36   (continues from models/student.pt)
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from policy import split, load_set, Norm, Student, action_loss

ROOT = Path(__file__).parent
from paths import MODELS, RESULTS
from hands import GRIP_THRESHOLD        # per-hand grip threshold (labels use the executed decision)


def fit_student(X, Y, epochs=60, seed=0, cfg=None, norm=None):
    """Train a fresh student on (obs, action or action chunk). Small net: seconds to a minute on CPU.
    norm: (mean, std) to reuse; default = statistics of X."""
    torch.manual_seed(seed)
    X, Y = torch.from_numpy(X), torch.from_numpy(Y)
    from policy import DEFAULT_CFG
    mean, std = norm if norm is not None else (X.mean(0).numpy(), X.std(0).numpy())
    model = Student(Norm(mean, std), cfg=cfg or DEFAULT_CFG, chunk=Y.dim() == 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    std = model.norm.std
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 512):
            b = perm[i:i + 512]
            x = X[b] + torch.randn_like(X[b]) * std * 0.05          # same obs noise as the teacher
            loss, _, _ = action_loss(model(x), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def _router_rollout(args):
    """One router-driven rollout. Returns (obs, labels, ok) for the ticks the student drove."""
    cup, co, seed, teacher_path, student_path, s_cfg = args
    import warnings; warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    from sim_env import Env, TeacherAgent, T_MAX
    from router import RouterAgent
    from policy import ref_point, obs_dim
    load = lambda f: torch.load(f, weights_only=False)
    env = Env()
    router = RouterAgent(load(teacher_path), load(student_path), mode="router", phase=load(MODELS / "phase.pt"))
    labeler = TeacherAgent(load(teacher_path))               # separate copy: its buffer doesn't disturb the router's
    env.reset(np.array(cup), np.array(co)); router.reset(); labeler.reset()
    obs, lab, status = [], [], None
    for t in range(T_MAX):
        o, palm = env.obs(), env.palm()
        labeler(o, palm, t)
        n0 = router.teacher_calls
        tgt, grip = router(o, palm, t)
        if router.teacher_calls == n0:                       # student drove this tick -> keep a labelled sample
            c_tgt, c_grip = labeler.last_chunk
            obs.append(o[:obs_dim(s_cfg)])
            lab.append(np.c_[c_tgt - ref_point(o[:31], palm, s_cfg["frame"])[0], (c_grip > GRIP_THRESHOLD).astype(np.float32)])
        env.step(tgt, grip > GRIP_THRESHOLD)
        status = env.status()
        if status:
            break
    D = obs_dim(s_cfg)
    from policy import H
    return (np.array(obs, np.float32).reshape(-1, D), np.array(lab, np.float32).reshape(-1, H, 4), status == "success")


def _rollout(args):
    """One DAgger rollout: mixed control, teacher labels every visited state. Returns (obs, labels, ok)."""
    cup, co, beta, seed, teacher_path, student_path, s_cfg = args
    import warnings; warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    from sim_env import Env, TeacherAgent, StudentAgent, T_MAX
    rng = np.random.default_rng(seed)
    env = Env()
    teacher = TeacherAgent(torch.load(teacher_path, weights_only=False))
    student = StudentAgent(torch.load(student_path, weights_only=False)) if student_path else None
    env.reset(np.array(cup), np.array(co)); teacher.reset()
    obs, lab, status = [], [], None
    for t in range(T_MAX):
        o, palm = env.obs(), env.palm()
        t_tgt, t_grip = teacher(o, palm, t)                 # teacher is queried EVERY tick (label + its buffer)
        from policy import ref_point, obs_dim                # label = teacher's raw chunk, in the student's recipe
        obs.append(o[:obs_dim(s_cfg)])
        c_tgt, c_grip = teacher.last_chunk
        lab.append(np.c_[c_tgt - ref_point(o[:31], palm, s_cfg["frame"])[0], (c_grip > GRIP_THRESHOLD).astype(np.float32)])
        if student is None or rng.random() < beta:
            tgt, grip = t_tgt, t_grip
        else:
            tgt, grip = student(o, palm, t)
        env.step(tgt, grip > GRIP_THRESHOLD)
        status = env.status()
        if status:
            break
    return np.array(obs, np.float32), np.array(lab, np.float32), status == "success"


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--rollouts", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--teacher", default=str(MODELS / "teacher.pt"))
    ap.add_argument("--router", action="store_true", help="DAgger inside the router (see module doc)")
    a = ap.parse_args()
    train, _ = split()
    from policy import DEFAULT_CFG, chunk_targets, H
    t_cfg = getattr(torch.load(a.teacher, weights_only=False), "cfg", DEFAULT_CFG)
    cfg = {"frame": t_cfg["frame"], "vel": False}            # same frame as the teacher, no velocities
    print(f"teacher recipe {t_cfg} -> student recipe {cfg} (chunked)", flush=True)
    O, A, _, refs = load_set(split(dart=True)[0], cfg)       # BC data: demos + DART replays (training clips only)
    X = np.concatenate(O)
    Y = np.concatenate([chunk_targets(act, r, H) for act, r in zip(A, refs)])
    norm = (X.mean(0), X.std(0))                             # fixed for all rounds (see module docstring)
    layouts = [(np.load(p)["cup_pos"][0, :2].tolist(), np.load(p)["coaster_pos"][0, :2].tolist()) for p in train]
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / "student.pt"
    t0 = time.time()
    if a.router:
        return router_dagger(a, X, Y, norm, cfg, layouts, path, t0)
    model = fit_student(X, Y, cfg=cfg, norm=norm)
    torch.save(model, path)
    print(f"round 0 (behaviour cloning on {len(X)} demo samples) -> student v0  {time.time()-t0:.0f}s", flush=True)
    log = []
    rng = np.random.default_rng(0)
    for k in range(1, a.rounds + 1):
        beta = 0.5 ** k                                     # teacher drives 50%, 25%, 12%, 6% of ticks
        jobs = [(*layouts[rng.integers(len(layouts))], beta, int(rng.integers(1e9)), a.teacher, str(path), cfg)
                for _ in range(a.rollouts)]
        with mp.get_context("spawn").Pool(a.workers) as pool:
            res = pool.map(_rollout, jobs)
        X = np.concatenate([X] + [r[0] for r in res]); Y = np.concatenate([Y] + [r[1] for r in res])
        succ = np.mean([r[2] for r in res])
        model = fit_student(X, Y, seed=k, cfg=cfg, norm=norm)
        torch.save(model, path)
        log.append(dict(round=k, beta=beta, rollout_success=float(succ), samples=int(len(X))))
        print(f"round {k}: beta={beta:.3f}  mixed-rollout success {succ:.0%}  dataset {len(X)}  "
              f"{time.time()-t0:.0f}s", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dagger_log.json").write_text(json.dumps(log, indent=1))
    print(f"student params {sum(p.numel() for p in model.parameters())/1e3:.1f}k -> {path}")


def router_dagger(a, X, Y, norm, cfg, layouts, path, t0):
    import multiprocessing as mp
    import shutil
    from eval_policy import layouts as eval_layouts
    # Checkpoint after EVERY round (aggregated data + log + student): the laptop shut down mid-run on 3 Nov
    # and the in-memory dataset was lost. A rerun with the same flags resumes after the last finished round.
    ck = RESULTS / "router_dagger_ckpt.npz"
    RESULTS.mkdir(parents=True, exist_ok=True)
    start, log = 1, []
    if ck.exists():
        c = np.load(ck, allow_pickle=True)
        X, Y, start, log = c["X"], c["Y"], int(c["round"]) + 1, list(c["log"])
        print(f"resuming after round {start - 1}: dataset {len(X)}", flush=True)
    elif not path.with_name("student_dagger_v1.pt").exists():
        shutil.copy(path, path.with_name("student_dagger_v1.pt"))      # keep the starting student
    for k in range(start, a.rounds + 1):
        rng = np.random.default_rng(1000 + k)                           # per-round stream: resume-safe
        fresh = eval_layouts("random", n=a.rollouts // 2, seed=1000 + k)  # NOT the eval set (seed 0)
        lay = [layouts[i] for i in rng.integers(len(layouts), size=a.rollouts - len(fresh))] + fresh
        jobs = [(c, co, int(rng.integers(1e9)), a.teacher, str(path), cfg) for c, co in lay]
        with mp.get_context("spawn").Pool(a.workers) as pool:
            res = pool.map(_router_rollout, jobs)
        X = np.concatenate([X] + [r[0] for r in res]); Y = np.concatenate([Y] + [r[1] for r in res])
        succ = float(np.mean([r[2] for r in res]))
        model = fit_student(X, Y, seed=k, cfg=cfg, norm=norm)
        torch.save(model, path)
        # keep every round's student: the best round is chosen afterwards on fresh layouts (seed 2000), and
        # on 3 Nov an overwritten 81% student was lost because only the latest one was kept
        torch.save(model, path.with_name(f"student_router_r{k}.pt"))
        log.append(dict(round=k, router_rollout_success=succ, samples=int(len(X))))
        np.savez(ck, X=X, Y=Y, round=k, log=np.array(log, dtype=object))
        print(f"router round {k}: router-rollout success {succ:.0%} (student used in that round = previous)  "
              f"dataset {len(X)}  {time.time()-t0:.0f}s", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dagger_router_log.json").write_text(json.dumps(log, indent=1))
    print(f"student params {sum(p.numel() for p in model.parameters())/1e3:.1f}k -> {path}")


if __name__ == "__main__":
    main()
