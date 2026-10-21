"""Block 2 summary: N1 open-loop cores, N2 rank-8/16 M0, N3 gene groups (MLP instrument).
    python aggregate_block2.py -> stdout + results/worm_block2/summary.json
"""
import collections, glob, json, pathlib, re

import numpy as np

D = pathlib.Path(__file__).resolve().parent.parent / "results" / "worm_block2"
G = collections.defaultdict(list)
for p in glob.glob(str(D / "*.json")):
    if p.endswith("summary.json"):
        continue
    stem = pathlib.Path(p).stem.split("_", 3)[-1]
    G[re.sub(r"_(f\d|s\d)$", "", re.sub(r"_f\d$", "", re.sub(r"_s\d$", "", stem)))].append(json.load(open(p)))


def ms(v):
    v = [x for x in v if x is not None]
    return [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), len(v)] if v else None


f = lambda t: "—" if not t else f"{t[0]:.3f} ± {t[1]:.3f} (n={t[2]})"
out = {}
print("## N1 open-loop cores / N2 rank M0")
print("| arm | params core | train loss | ZS ours | ZS auth | FS n_cal 5 / 10 / 20 |\n|---|---|---|---|---|---|")
for k in sorted(G):
    if k.startswith("G_"):
        continue
    rs = G[k]
    row = {"loss": ms([r["train_loss_last500"] for r in rs]), "zs": ms([(r.get("zero_shot") or {}).get("r") for r in rs]),
           "auth": ms([r.get("auth", {}).get("r") for r in rs]),
           "fs": [ms([r["few_shot"].get(n, {}).get("few_shot", {}).get("r") for r in rs]) for n in ("5", "10", "20")]}
    out[k] = row
    print(f"| {k} | {rs[0]['params_core']} | {f(row['loss'])} | {f(row['zs'])} | {f(row['auth'])} | " + " / ".join(f(t) for t in row["fs"]) + " |")
print("\n## N3 gene groups (MLP, 5 folds × 1 seed): held-out entries, ours / auth")
for g in ("channels", "receptors", "neuropeptide", "tf"):
    a, b = G.get(f"G_{g}", []), G.get(f"G_random_{g}", [])
    ra = [r["zero_shot_heldout_entries"]["r"] for r in a]; rb = [r["zero_shot_heldout_entries"]["r"] for r in b]
    aa = [r["auth"]["held_r"] for r in a]; ab = [r["auth"]["held_r"] for r in b]
    fa = {r["variant"]["holdout_fold"]: r["auth"]["held_r"] for r in a}; fb = {r["variant"]["holdout_fold"]: r["auth"]["held_r"] for r in b}
    wins = sum(fa[k] > fb[k] for k in set(fa) & set(fb))
    out[f"N3_{g}"] = {"group": [ms(ra), ms(aa)], "random": [ms(rb), ms(ab)], "folds_won_auth": wins}
    print(f"  {g:13s} group {f(ms(ra))} / {f(ms(aa))}   random {f(ms(rb))} / {f(ms(ab))}   folds won {wins}/{len(set(fa) & set(fb))}")
json.dump(out, open(D / "summary.json", "w"), indent=1)
