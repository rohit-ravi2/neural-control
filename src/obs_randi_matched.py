"""Disentangle leakage from sample size in OBS-Randi: zero-lag correlation from train animals' clean frames at exclusion w,
subsampled per animal to a FIXED frame budget (same budget for every w), 10 random subsamples each. Also budget curve at w=10.
    python obs_randi_matched.py -> results/obs_randi_matched.json
"""
import json

import numpy as np

import obs_randi_strict as ors

rng = np.random.default_rng(0)


def subsampled(w, frac_fn, reps=10):
    vals = []
    for _ in range(reps):
        base = ors.after(w)
        def f(u, base=base):
            ok = base(u); idx = np.where(ok)[0]
            keep = rng.choice(idx, int(round(len(idx) * frac_fn(len(idx)))), replace=False) if len(idx) else idx
            m = np.zeros_like(ok); m[keep] = True
            return m
        S, fr, _ = ors.pooled_corr(ors.tr_n, f, min_frames=10)
        vals.append((ors.c(S, ors.TA["T_auth_ours"]), fr))
    return vals


totals = {10: 196680, 20: 157400, 30: 119253, 45: 101666, 60: 75203}   # from obs_randi / obs_randi_strict
budget = 70000
out = {}
for w in (10, 20, 30, 45, 60):
    v = subsampled(w, lambda n_, w=w: min(1.0, budget / totals[w]))
    out[f"w{w}_budget{budget}"] = {"r_mean": float(np.mean([a for a, _ in v])), "r_sd": float(np.std([a for a, _ in v])), "frames": int(np.mean([b for _, b in v]))}
    print(w, out[f"w{w}_budget{budget}"], flush=True)
for b in (10000, 20000, 40000):
    v = subsampled(10, lambda n_, b=b: b / totals[10], reps=5)
    out[f"w10_budget{b}"] = {"r_mean": float(np.mean([a for a, _ in v])), "r_sd": float(np.std([a for a, _ in v])), "frames": int(np.mean([x for _, x in v]))}
    print("w10 budget", b, out[f"w10_budget{b}"], flush=True)
json.dump(out, open("../results/obs_randi_matched.json", "w"), indent=1)
