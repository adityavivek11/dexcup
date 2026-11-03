"""Pick the best router-DAgger round's student by ROUTER success on fresh layouts (not the test set).

WHY a separate selection set: the 100 test layouts (eval_policy layouts("random"), seed 0) must stay unseen,
otherwise picking the best round on them inflates the reported number. Seed 2000 draws 40 different
layouts from the same distribution (router-DAgger trained on seeds 1000+k, also different).
Run:  python select_student.py   -> prints each round's success, copies the best to MODELS/student.pt
"""
import shutil
import numpy as np
import torch
from paths import MODELS


def _run(args):
    sp, cup, co = args
    import warnings; warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    from sim_env import Env, rollout
    from router import RouterAgent
    load = lambda f: torch.load(MODELS / f, weights_only=False)
    ag = RouterAgent(load("teacher.pt"), load(sp), mode="router", phase=load("phase.pt"))
    r = rollout(Env(), ag, np.array(cup), np.array(co))
    return sp, r["success"], ag.teacher_calls / r["ticks"]


if __name__ == "__main__":
    import multiprocessing as mp
    from eval_policy import layouts
    lay = layouts("random", n=40, seed=2000)
    cands = sorted(p.name for p in MODELS.glob("student_router_r*.pt"))
    with mp.get_context("spawn").Pool(6) as pool:
        res = pool.map(_run, [(s, c, k) for s in cands for c, k in lay])
    score = {}
    for s in cands:
        r = [x for x in res if x[0] == s]
        score[s] = np.mean([x[1] for x in r])
        print(f"{s}: router success {score[s]:.0%}  teacher fraction {np.mean([x[2] for x in r]):.0%}", flush=True)
    best = max(cands, key=lambda s: score[s])
    shutil.copy(MODELS / best, MODELS / "student.pt")
    print(f"selected {best} -> {MODELS / 'student.pt'}")
