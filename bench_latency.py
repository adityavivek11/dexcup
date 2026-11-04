"""Time one forward pass of each model: one process, one thread, nothing else running -> results/latency.json.

WHY separately from eval_policy.py: its timings come from 6 simulations sharing the CPU, which inflates them
(teacher ~20 ms there vs ~10 ms here). Numbers depend on the machine and its power state; only compare models
timed in the same run.

Run:  python bench_latency.py
"""
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch

try:
    from paths import MODELS, RESULTS    # inspire-hand branch: per-hand folders
except ImportError:
    MODELS, RESULTS = Path(__file__).parent / "models", Path(__file__).parent / "results"

if __name__ == "__main__":
    torch.set_num_threads(1)
    from sim_env import Env
    env = Env()
    env.reset(np.array([0.36, -0.12]), np.array([0.36, 0.0]))
    obs = env.obs()                                  # a real observation; timing doesn't depend on its values
    out = {"machine": platform.processor() or platform.machine(), "calls": 3000}
    for name, dim in [("teacher", 38), ("student", 31), ("phase", 31)]:
        model = torch.load(MODELS / f"{name}.pt", weights_only=False).eval()
        x = torch.from_numpy(obs[:dim])[None]
        ts = []
        with torch.no_grad():
            for _ in range(50):                      # warm-up: the first calls are slower
                model(x)
            for _ in range(out["calls"]):
                t0 = time.perf_counter()
                model(x)
                ts.append((time.perf_counter() - t0) * 1000)
        out[name] = dict(mean_ms=float(np.mean(ts)), p95_ms=float(np.percentile(ts, 95)),
                         params=sum(p.numel() for p in model.parameters()))
        print(f"{name}: mean {out[name]['mean_ms']:.3f} ms, p95 {out[name]['p95_ms']:.3f} ms, "
              f"{out[name]['params'] / 1e3:.1f}k params")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "latency.json").write_text(json.dumps(out, indent=1))
