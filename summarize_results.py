"""Recompute every number in the README's results table from the stored per-episode results. No sim, no training.

Reads results/{teacher,router,random,student}_random.json (written by eval_policy.py) and results/latency.json
(written by bench_latency.py). Compute per step is an ESTIMATE: per-model timings weighted by the measured
teacher-call fraction, because latencies inside the parallel evaluation are inflated by CPU sharing.

Run:  python summarize_results.py
"""
import json
from math import comb
from pathlib import Path
import numpy as np

try:
    from paths import RESULTS            # inspire-hand branch: per-hand folders
except ImportError:
    RESULTS = Path(__file__).parent / "results"

AGENTS = ["teacher", "router", "random", "student"]


def mcnemar(a, b):
    """Exact two-sided McNemar test on layouts where the two agents disagree: under 'no difference' each
    disagreement is a fair coin flip. Returns (wins for a, wins for b, p)."""
    wa = sum(a[i] and not b[i] for i in a)
    wb = sum(b[i] and not a[i] for i in a)
    n = wa + wb
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(wa, wb) + 1)) / 2 ** n) if n else 1.0
    return wa, wb, p


if __name__ == "__main__":
    R = {a: json.loads((RESULTS / f"{a}_random.json").read_text()) for a in AGENTS}
    L = json.loads((RESULTS / "latency.json").read_text())
    ok = {a: {e["i"]: bool(e["success"]) for e in R[a]["per_episode"]} for a in AGENTS}
    T, S, P = L["teacher"]["mean_ms"], L["student"]["mean_ms"], L["phase"]["mean_ms"]
    print(f"{'policy':8s} {'success':>8s} {'teacher calls':>14s} {'est. ms/step':>13s} {'teacher calls/success':>22s}")
    for a in AGENTS:
        f = R[a]["teacher_call_fraction"]
        ms = f * T + (1 - f) * S + (P if a == "router" else 0)     # the random router needs no phase net
        calls = np.mean([e.get("teacher_calls", e["ticks"] if a == "teacher" else 0) for e in R[a]["per_episode"]])
        s = R[a]["success"]
        print(f"{a:8s} {s:8.0%} {f:14.0%} {ms:13.2f} {(f'{calls / s:.0f}' if calls else '-'):>22s}")
        if a == "router":
            router_ms = ms
    print(f"\ncompute saving, router vs teacher: {T / router_ms:.1f}x (estimated)")
    for x, y in [("router", "random"), ("teacher", "router")]:
        wa, wb, p = mcnemar(ok[x], ok[y])
        print(f"{x} vs {y}, same {len(ok[x])} layouts: {x} better on {wa}, {y} better on {wb}, exact McNemar p = {p:.2g}")
    print("\nlatency (1 process, 1 thread), mean / p95 ms:",
          " | ".join(f"{k} {v['mean_ms']:.2f} / {v['p95_ms']:.2f}" for k, v in L.items() if isinstance(v, dict)))
