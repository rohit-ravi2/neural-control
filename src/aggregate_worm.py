"""Aggregate worm result JSONs (mean ± SD over seeds) into a markdown table.

    python aggregate_worm.py results/worm_v2
"""
import collections, glob, json, sys

import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "results/worm_v2"
R = collections.defaultdict(dict)
for p in sorted(glob.glob(f"{d}/*_s[0-9].json")):
    r = json.load(open(p))
    R[(r["scheme"], r["level"])][r["seed"]] = r


def ms(v):
    v = np.array([x for x in v if x is not None], float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else "n/a"


print("| model | seeds | params core/tables | train loss | zero-shot r | n_events | rho |")
print("|---|---|---|---|---|---|---|")
for k, s in sorted(R.items()):
    rs = list(s.values())
    zs = [r["zero_shot"]["r"] if r.get("zero_shot") else None for r in rs]
    ne = {r["zero_shot"]["n_events"] for r in rs if r.get("zero_shot")}
    rho = {round(r["zero_shot"]["rho_same_events"], 4) for r in rs if r.get("zero_shot")}
    print(f"| {k[0]} {k[1]} | {len(rs)} | {rs[0]['params_core']}/{rs[0]['params_tables']} | "
          f"{ms([r['train_loss_last500'] for r in rs])} | {ms(zs)} | {ne or '-'} | {rho or '-'} |")

print("\n| n_cal | model | few-shot r | zero-shot r (same events) | rho | n_events |")
print("|---|---|---|---|---|---|")
for n in ("5", "10", "20"):
    for k, s in sorted(R.items()):
        rs = [r for r in s.values() if n in r.get("few_shot", {})]
        if not rs:
            continue
        fs = [r["few_shot"][n]["few_shot"]["r"] for r in rs]
        zs = [r["few_shot"][n].get("zero_shot_same_events", {}).get("r") for r in rs]
        rho = rs[0]["few_shot"][n]["few_shot"]["rho_same_events"]
        ne = rs[0]["few_shot"][n]["few_shot"]["n_events"]
        print(f"| {n} | {k[0]} {k[1]} | {ms(fs)} | {ms(zs)} | {rho:.3f} | {ne} |")

# paired M1 - M2 zero-shot
for lvl in ("neuron", "class"):
    a, b = R.get(("M1", lvl), {}), R.get(("M2", lvl), {})
    seeds = sorted(set(a) & set(b))
    if seeds:
        diff = [a[s]["zero_shot"]["r"] - b[s]["zero_shot"]["r"] for s in seeds]
        print(f"\npaired M1-M2 zero-shot ({lvl}): {ms(diff)} over {len(seeds)} seeds")
