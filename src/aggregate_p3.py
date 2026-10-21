"""Phase 3 summary: M3/M4 zero-shot (5 seeds) and the held-out-class test (P5).
    python aggregate_p3.py -> stdout + results/worm_p3/summary.json
"""
import glob, json, pathlib

import numpy as np

P = pathlib.Path(__file__).resolve().parent.parent / "results" / "worm_p3"


def ms(v):
    v = [x for x in v if x is not None]
    return [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), len(v)] if v else None


out = {}
print("| model | zero-shot r (all) | held-out-class entries | seen entries |\n|---|---|---|---|")
for tag in ("M3", "M4", "M3h", "M4h", "M1h"):
    rs = [json.load(open(p)) for p in sorted(glob.glob(str(P / f"*_{tag}_s[0-9].json")))]
    row = {"zs": ms([r["zero_shot"]["r"] for r in rs]),
           "held": ms([r.get("zero_shot_heldout_entries", {}).get("r") for r in rs]),
           "seen": ms([r.get("zero_shot_seen_entries", {}).get("r") for r in rs])}
    out[tag] = row
    f = lambda t: "—" if not t else f"{t[0]:.3f} ± {t[1]:.3f} (n={t[2]})"
    print(f"| {tag} | {f(row['zs'])} | {f(row['held'])} | {f(row['seen'])} |")
for a, b in (("M3", "M4"), ("M3h", "M4h"), ("M3h", "M1h")):
    A = {json.load(open(p))["seed"]: json.load(open(p)) for p in glob.glob(str(P / f"*_{a}_s[0-9].json"))}
    B = {json.load(open(p))["seed"]: json.load(open(p)) for p in glob.glob(str(P / f"*_{b}_s[0-9].json"))}
    key = "zero_shot_heldout_entries" if a.endswith("h") else "zero_shot"
    d = [A[s][key]["r"] - B[s][key]["r"] for s in sorted(set(A) & set(B))]
    out[f"{a}-{b}"] = ms(d)
    print(f"paired {a} - {b} ({key}): {out[f'{a}-{b}']}")
json.dump(out, open(P / "summary.json", "w"), indent=1)
