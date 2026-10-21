"""Combine 5-fold CV held-out predictions (all 110 animals) and score against the all-110 authors' STAM (ceiling in
results/ceiling_all.json). Creamer's published models are scored on the same target for reference (they were trained on
80 of these animals, so the comparison is tilted toward Creamer).
    python cv_combine.py -> results/cv_combine.json
"""
import glob, json, pickle, re, sys

import numpy as np

import phase0_ceiling as p0
import worm_data as wd

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402
import lgssm_utilities as ssmu  # noqa: E402

cell = wd.model_cell_ids(); eye = np.eye(len(cell), dtype=bool)
T = np.load(p0.OUT / "targets_test.npz")["T_auth_all"]
c = lambda a: float(met.nan_corr(np.where(eye, np.nan, a), np.where(eye, np.nan, T))[0])
ceil = json.load(open("../results/ceiling_all.json"))["noiseless_ceiling_110"]
groups = {}
import os
SRC = os.environ.get("CVDIR", "worm_final")
for p in glob.glob(f"../results/{SRC}/*_acc.npz"):
    m = re.search(r"_(CV[AB]|CFCV|CFCVSH)_f(\d)_s(\d)_acc\.npz", p)
    if m:
        groups.setdefault((m.group(1), int(m.group(3))), []).append(np.load(p))
out = {"ceiling_110": ceil, "per_seed": {}}
stams = {}
for (cfg, s), parts in sorted(groups.items()):
    if len(parts) < 5:
        print("incomplete", cfg, s, len(parts)); continue
    acc = sum(z["acc"] for z in parts); cnt = sum(z["cnt"] for z in parts)
    S = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan); S = (S - np.nanmean(S)) / np.nanstd(S)
    stams.setdefault(cfg, []).append(S); out["per_seed"][f"{cfg}_s{s}"] = c(S)
for cfg, v in stams.items():
    out[f"{cfg}_seed_ensemble"] = c(np.nanmean(v, 0))
if stams:
    out["A+B_ensemble"] = c(np.nanmean([np.nanmean(v, 0) for v in stams.values()], 0))
for mn in ("connectome_constrained", "fully_connected", "shuffled_constrained"):
    m = pickle.load(open(wd.REPO / "models" / f"{mn}.pkl", "rb"))
    out[f"Creamer {mn} (contaminated)"] = c(ssmu.calculate_irfs(m, window=(15, 30))[15:].sum(0))
for k in list(out):
    if isinstance(out[k], float) and k != "ceiling_110":
        print(f"{k:45s} {out[k]:.3f}  ({out[k] / ceil:.0%} of ceiling)")
json.dump(out, open("../results/cv_combine.json" if SRC == "worm_final" else f"../results/cv_combine_{SRC}.json", "w"), indent=1)
