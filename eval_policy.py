"""Stage 9 (harness): closed-loop evaluation of a policy in the sim.

Layout sets (fixed, so every row of the results table sees the SAME episodes):
  val    : the held-out human demos' tube/coaster layouts (policy.split(), every 5th episode)
  random : N layouts, seed 0: tube and coaster uniform over the range the TRAINING demos covered,
           >= 10 cm apart, both inside the measured robot workspace (reach_map.py)

Run:  python eval_policy.py teacher --layouts val
      python eval_policy.py teacher --layouts random --n 100   -> results/<agent>_<layouts>.json
"""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent


def layouts(kind, n=100, seed=0):
    from policy import split
    train, val = split()
    first = lambda p: (np.load(p)["cup_pos"][0, :2], np.load(p)["coaster_pos"][0, :2])
    if kind == "val":
        return [tuple(map(np.ndarray.tolist, first(p))) for p in val]
    if kind.startswith("fold"):                         # cross-validation: fold k's held-out clips
        _, v = split(fold=int(kind[4:]))
        return [tuple(map(np.ndarray.tolist, first(p))) for p in v]
    from retarget import Reach
    reach = Reach()
    tr = np.array([np.r_[first(p)[0], first(p)[1]] for p in train])
    lo, hi = tr.min(0), tr.max(0)
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        s = rng.uniform(lo, hi)
        cup, co = s[:2], s[2:]
        if np.linalg.norm(cup - co) >= 0.10 and reach(cup) and reach(co):
            out.append((cup.tolist(), co.tolist()))
    return out


def make_agent(name, layouts="val", seed=0):
    import torch
    from sim_env import TeacherAgent, StudentAgent
    load = lambda f: torch.load(ROOT / "models" / f, weights_only=False)
    if name == "teacher":
        return TeacherAgent(load("teacher.pt"))
    if name.startswith("teacher_"):                     # e.g. teacher_fold2 -> models/teacher_fold2.pt
        return TeacherAgent(load(f"{name}.pt"))
    if name == "student":
        return StudentAgent(load("student.pt"))
    from router import RouterAgent
    if name == "router":
        return RouterAgent(load("teacher.pt"), load("student.pt"), mode="router", phase=load("phase.pt"))
    if name == "random":
        # same teacher-call fraction AND mean teacher-segment length as the router on the SAME layouts
        r = json.loads((ROOT / "results" / f"router_{layouts}.json").read_text())
        return RouterAgent(load("teacher.pt"), load("student.pt"), mode="random",
                           p=r["teacher_call_fraction"], L=r["mean_segment_ticks"], seed=seed)
    raise ValueError(name)


def _run(args):
    name, i, cup, co, kind = args
    import torch
    torch.set_num_threads(1)                 # one core per worker: fair, comparable latency numbers
    from sim_env import Env, rollout
    env = Env()
    agent = make_agent(name, kind, seed=i)          # random router: a different random stream per episode
    r = rollout(env, agent, np.array(cup), np.array(co))
    out = dict(i=i, success=r["success"], ticks=r["ticks"], call_ms=r["call_ms"])
    if hasattr(agent, "teacher_calls"):
        out["teacher_calls"], out["segments"] = agent.teacher_calls, agent.segments
    return out


if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("agent")
    ap.add_argument("--layouts", default="val")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    L = layouts(a.layouts, a.n)
    with mp.get_context("spawn").Pool(a.workers) as p:
        res = p.map(_run, [(a.agent, i, c, o, a.layouts) for i, (c, o) in enumerate(L)])
    ms = np.concatenate([r["call_ms"] for r in res])
    succ = np.mean([r["success"] for r in res])
    tfrac = (sum(r.get("teacher_calls", r["ticks"] if a.agent.startswith("teacher") else 0) for r in res)
             / sum(r["ticks"] for r in res))
    summary = dict(agent=a.agent, layouts=a.layouts, n=len(res), success=float(succ),
                   latency_ms_mean=float(ms.mean()), latency_ms_p95=float(np.percentile(ms, 95)),
                   teacher_call_fraction=float(tfrac),
                   mean_segment_ticks=float(sum(r.get("teacher_calls", 0) for r in res) /
                                            max(1, sum(r.get("segments", 0) for r in res))),
                   per_episode=[{k: v for k, v in r.items() if k != "call_ms"}
                                                                    for r in res])
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{a.agent}_{a.layouts}.json").write_text(json.dumps(summary, indent=1))
    print(f"{a.agent} on {a.layouts} ({len(res)} eps): success {succ:.0%} | latency per call mean "
          f"{ms.mean():.3f} ms, p95 {np.percentile(ms, 95):.3f} ms | teacher-call fraction {tfrac:.0%}")
