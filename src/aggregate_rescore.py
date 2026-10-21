"""N4 summary: every zero-shot model on both estimators (ours = event-pooled; auth = authors' clip-mean).
    python aggregate_rescore.py [results dir, default results/rescore_auth] -> stdout + <dir>/summary.json
"""
import collections, glob, json, pathlib, re, sys

import numpy as np

D = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent.parent / "results" / "rescore_auth"
G = collections.defaultdict(list)
for p in glob.glob(str(D / "*.json")):
    if p.endswith("summary.json"):
        continue
    r = json.load(open(p))
    if "auth" not in r:
        continue
    tag = r["variant"].get("tag") or pathlib.Path(p).stem.split("_", 3)[-1]
    G[re.sub(r"_s\d", "", re.sub(r"_f\d", "_cv", tag))].append(r)


def ms(v):
    v = [x for x in v if x is not None]
    return [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), len(v)] if v else None


f = lambda t: "—" if not t else f"{t[0]:.3f} ± {t[1]:.3f}"
out = {}
print("| model | n | ours: all | auth: all (rho %.3f) | ours: held-out | auth: held-out | auth held rho | auth: seen |" %
      next(iter(G.values()))[0]["auth"]["rho"])
print("|---|---|---|---|---|---|---|---|")
for k in sorted(G):
    rs = G[k]
    row = {"n": len(rs), "ours_all": ms([r["zero_shot"]["r"] for r in rs]), "auth_all": ms([r["auth"]["r"] for r in rs]),
           "ours_held": ms([r.get("zero_shot_heldout_entries", {}).get("r") for r in rs]),
           "auth_held": ms([r["auth"].get("held_r") for r in rs]), "auth_held_rho": ms([r["auth"].get("held_rho") for r in rs]),
           "auth_seen": ms([r["auth"].get("seen_r") for r in rs])}
    out[k] = row
    print(f"| {k} | {row['n']} | {f(row['ours_all'])} | {f(row['auth_all'])} | {f(row['ours_held'])} | {f(row['auth_held'])} | "
          f"{f(row['auth_held_rho'])} | {f(row['auth_seen'])} |")
json.dump(out, open(D / "summary.json", "w"), indent=1)
