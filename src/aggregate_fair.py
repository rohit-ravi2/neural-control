"""Fair-M0 protocol summary.
(i) default few-shot (lr 3e-3, anchor 0), 5 seeds; (ii) symmetric oracle over the trimmed grid
lr {3e-3, 3e-4} x anchor {0, 10}, seeds 0-2, best grid point per model per n_cal (selected on the test events,
so biased upward equally for every model); (iii) seen-target subset.
    python aggregate_fair.py -> stdout + results/worm_fair/summary.json
"""
import glob, json, pathlib

import numpy as np

R = pathlib.Path(__file__).resolve().parent.parent / "results"
F, V2 = R / "worm_fair", R / "worm_v2"
NC = ("5", "10", "20")


def load(p):
    p = list(glob.glob(str(p)))
    return json.load(open(p[0])) if p else None


def fs(r, n, key="few_shot"):
    return r["few_shot"][n].get(key, {}).get("r") if r and n in r["few_shot"] else None


def ms(v):
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), len(v)) if v else None


def fmt(t):
    return "—" if t is None else f"{t[0]:.3f} ± {t[1]:.3f} (n={t[2]})"


default = {  # model -> seed -> result json
    "M1 neuron (v2)": lambda s: load(V2 / f"M1_neuron_s{s}.json"),
    "M1 class (v2)": lambda s: load(V2 / f"M1_class_s{s}.json"),
    "M0 random (v2)": lambda s: load(V2 / f"M0_neuron_s{s}.json"),
    "M1t": lambda s: load(F / f"*_M1t_s{s}.json"),
    "M0t random": lambda s: load(F / f"*_M0t_s{s}.json"),
    "M0r random": lambda s: load(F / f"*_M0r_s{s}.json"),
}
grid = {  # model -> (lr, anchor, seed) -> result json
    "M1 neuron": lambda lr, a, s: load(V2 / f"M1_neuron_s{s}.json") if (lr, a) == ("3e-3", "0") else load(F / f"*_M1n_lr{lr}_a{a}_s{s}.json"),
    "M0 popmean": lambda lr, a, s: load(F / f"*_M0pop_lr{lr}_a{a}_s{s}.json"),
    "M1t": lambda lr, a, s: load(F / f"*_M1t_s{s}.json") if (lr, a) == ("3e-3", "0") else load(F / f"*_M1t_lr{lr}_a{a}_s{s}.json"),
    "M0t popmean": lambda lr, a, s: load(F / f"*_M0tpop_lr{lr}_a{a}_s{s}.json"),
}
out = {"default": {}, "oracle": {}, "grid": {}, "seen": {}}
print("## (i) default few-shot, 5 seeds")
print("| model | " + " | ".join(f"n_cal {n}" for n in NC) + " | zero-shot |")
print("|---|" + "---|" * 4)
for m, f in default.items():
    rs = [f(s) for s in range(5)]
    row = [ms([fs(r, n) for r in rs]) for n in NC]
    zs = ms([(r.get("zero_shot") or {}).get("r") for r in rs if r])
    out["default"][m] = {"fs": row, "zs": zs}
    print(f"| {m} | " + " | ".join(fmt(t) for t in row) + f" | {fmt(zs)} |")

print("\n## grid (seeds 0-2): mean few-shot r per grid point")
G = [("3e-3", "0"), ("3e-3", "10"), ("3e-4", "0"), ("3e-4", "10")]
for m, f in grid.items():
    for lr, a in G:
        rs = [f(lr, a, s) for s in range(3)]
        out["grid"][f"{m} lr{lr} a{a}"] = [ms([fs(r, n) for r in rs]) for n in NC]
        print(f"  {m:12s} lr {lr} anchor {a:>2s}: " + "  ".join(fmt(t) for t in out["grid"][f"{m} lr{lr} a{a}"]))

print("\n## (ii) symmetric oracle: best grid point per model per n_cal (selected on test events)")
for m in grid:
    best = []
    for i, n in enumerate(NC):
        cands = [(out["grid"][f"{m} lr{lr} a{a}"][i], f"lr{lr}/a{a}") for lr, a in G if out["grid"][f"{m} lr{lr} a{a}"][i]]
        best.append(max(cands, key=lambda c: c[0][0]) if cands else (None, ""))
    out["oracle"][m] = best
    print(f"  {m:12s} " + "  ".join(f"{fmt(b[0])} [{b[1]}]" for b in best))

print("\n## (iii) seen-target subset (eval events whose target was stimulated in calibration), default config")
for m in ("M1t", "M0t random", "M0r random"):
    rs = [default[m](s) for s in range(5)]
    out["seen"][m] = [ms([fs(r, n, "seen_targets") for r in rs]) for n in NC]
    print(f"  {m:12s} " + "  ".join(fmt(t) for t in out["seen"][m]))
json.dump(out, open(F / "summary.json", "w"), indent=1)
