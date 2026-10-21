"""Phase 3b summary: priors for never-trained neurons under 5-fold class CV (P6/P7).
    python aggregate_p3b.py -> stdout + results/worm_p3b/summary.json
"""
import collections, glob, json, pathlib

import numpy as np

P = pathlib.Path(__file__).resolve().parent.parent / "results" / "worm_p3b"
R = collections.defaultdict(dict)  # model -> (fold, seed) -> result
for p in glob.glob(str(P / "*_f[0-9]_s[0-9].json")):
    tag = p.rsplit("_", 3)
    r = json.load(open(p))
    m = pathlib.Path(p).stem.split("_")[-3]
    R[m][(r["variant"]["holdout_fold"], r["seed"])] = r


def ms(v):
    v = [x for x in v if x is not None]
    return [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), len(v)] if v else None


f = lambda t: "—" if not t else f"{t[0]:.3f} ± {t[1]:.3f} (n={t[2]})"
out = {"per_model": {}, "per_fold": {}, "paired": {}}
print("| model | held-out entries | seen entries | zero-shot all |\n|---|---|---|---|")
for m in ("M3h", "Mch", "Mbh", "M1h"):
    rs = list(R[m].values())
    row = [ms([r["zero_shot_heldout_entries"]["r"] for r in rs]), ms([r["zero_shot_seen_entries"]["r"] for r in rs]),
           ms([r["zero_shot"]["r"] for r in rs])]
    out["per_model"][m] = row
    print(f"| {m} | " + " | ".join(f(t) for t in row) + " |")
print("\nper fold (mean over seeds), held-out entries:")
for k in range(5):
    out["per_fold"][k] = {m: ms([r["zero_shot_heldout_entries"]["r"] for (fk, s), r in R[m].items() if fk == k]) for m in R}
    print(f"  fold {k}: " + "  ".join(f"{m} {f(out['per_fold'][k][m])}" for m in ("M3h", "Mch", "Mbh", "M1h") if m in R))
for a, b in (("M3h", "M1h"), ("M3h", "Mch"), ("Mbh", "M3h"), ("Mbh", "Mch"), ("Mch", "M1h")):
    keys = sorted(set(R[a]) & set(R[b]))
    d = [R[a][k]["zero_shot_heldout_entries"]["r"] - R[b][k]["zero_shot_heldout_entries"]["r"] for k in keys]
    wins = sum(np.mean([R[a][k]["zero_shot_heldout_entries"]["r"] - R[b][k]["zero_shot_heldout_entries"]["r"]
                        for k in keys if k[0] == fk]) > 0 for fk in sorted({k[0] for k in keys}))
    out["paired"][f"{a}-{b}"] = {"diff": ms(d), "folds_won": int(wins), "folds": len({k[0] for k in keys})}
    print(f"paired {a} - {b}: {f(ms(d))}, folds won {wins}/{len({k[0] for k in keys})}")
json.dump(out, open(P / "summary.json", "w"), indent=1)
