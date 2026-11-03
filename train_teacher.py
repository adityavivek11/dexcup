"""Stage 6: train the teacher (transformer, 16-step action chunks) on the replayed human demos.

Behaviour cloning: predict the demo's next 16 actions from the current observation.
WHY observation noise during training: the policy will see states slightly off the demos (its own small
errors compound). Training on noisy copies of the observations teaches it to steer back instead of
falling apart at the first deviation. Cheap version of what DAgger does properly (used for the student).

Run:  python train_teacher.py [--epochs 80]   ->  models/teacher.pt
"""
import argparse
import copy
import time
from pathlib import Path
import numpy as np
import torch
from policy import (split, load_set, chunk_targets, Norm, Teacher, action_loss, H)

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--noise", type=float, default=0.05, help="obs noise, in units of feature std")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N batches (smoke test)")
    ap.add_argument("--dart", action="store_true", help="add noise-injected replays of the training clips")
    ap.add_argument("--fold", type=int, default=None, help="cross-validation fold k (val = every 5th clip from k)")
    ap.add_argument("--frame", choices=["palm", "object"], default="palm", help="action reference (see policy.py)")
    ap.add_argument("--vel", action="store_true", help="observe arm joint velocities")
    ap.add_argument("--name", default=None, help="output file name in models/")
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    # GPU on the cloud machine, CPU on the laptop: same script. The whole dataset (~30k samples) fits in
    # GPU memory, so it is moved there once instead of copying every batch.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""), flush=True)
    train, val = split(dart=a.dart, fold=a.fold)
    cfg = {"frame": a.frame, "vel": a.vel}
    name = a.name or ("teacher.pt" if a.fold is None else f"teacher_fold{a.fold}.pt")
    print(f"recipe: {cfg} -> models/{name}", flush=True)
    print(f"episodes: train {len(train)}, val {len(val)} "
          + ("(val clips frozen in data/split.json)" if a.fold is None else f"(cross-validation fold {a.fold})"), flush=True)
    O, A, _, Pm = load_set(train, cfg)
    Ov, Av, _, Pv = load_set(val, cfg)
    X = torch.from_numpy(np.concatenate(O))
    Y = torch.from_numpy(np.concatenate([chunk_targets(x, p) for x, p in zip(A, Pm)]))
    Xv = torch.from_numpy(np.concatenate(Ov))
    Yv = torch.from_numpy(np.concatenate([chunk_targets(x, p) for x, p in zip(Av, Pv)]))
    norm = Norm(X.mean(0).numpy(), X.std(0).numpy())
    model = Teacher(norm, cfg=cfg).to(dev)
    X, Y, Xv, Yv = X.to(dev), Y.to(dev), Xv.to(dev), Yv.to(dev)
    print(f"samples: train {len(X)}, val {len(Xv)} | teacher params {sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    std = norm.std
    best, t0, steps = 1e9, time.time(), 0
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), a.batch):
            b = perm[i:i + a.batch]
            x = X[b] + torch.randn_like(X[b]) * std * a.noise
            loss, _, _ = action_loss(model(x), Y[b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1
            if a.max_steps and steps >= a.max_steps:
                break
        sched.step()
        model.eval()
        with torch.no_grad():
            lv, pv, gv = action_loss(model(Xv), Yv)
        # Normal run: keep the best-val-loss checkpoint. Cross-validation: keep the FINAL model instead, since
        # picking a checkpoint by the fold's held-out clips would let the test clips choose the model.
        if (lv < best and a.fold is None) or (a.fold is not None and ep == a.epochs - 1):
            best = float(lv)
            (ROOT / "models").mkdir(exist_ok=True)
            # saved as a CPU model: the laptop loads it for the sim evaluation, which runs on CPU
            torch.save(copy.deepcopy(model).cpu(), ROOT / "models" / name)
        # every epoch, flushed: buffered output hid all progress on the first (CPU) attempt
        print(f"epoch {ep:3d}  val loss {lv:.3f}  (pos rmse {pv.sqrt():.2f} cm, grip bce {gv:.3f})  "
              f"{time.time()-t0:.0f}s", flush=True)
        if a.max_steps and steps >= a.max_steps:
            break
    print(f"best val loss {best:.3f} -> models/{name}", flush=True)


if __name__ == "__main__":
    main()
