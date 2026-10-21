"""Noise ceiling of the all-110-animal authors' STAM (target for 5-fold CV): split-half 55/55 × 20, Spearman-Brown.
    python ceiling_all.py -> results/ceiling_all.json
"""
import json, sys, warnings

import numpy as np

import phase0_ceiling as p0
import worm_data as wd

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402

cell = wd.model_cell_ids(); eye = np.eye(len(cell), dtype=bool)
c = lambda a, b: float(met.nan_corr(np.where(eye, np.nan, a), np.where(eye, np.nan, b))[0])
names = wd.recording_names(); rng = np.random.default_rng(1); rh = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _ in range(20):
        p = rng.permutation(names); rh.append(c(p0.authors_stams(sorted(p[:55]), cell), p0.authors_stams(sorted(p[55:]), cell)))
r = float(np.mean(rh)); rtt = 2 * r / (1 + r)
out = {"r_half_55": r, "r_half_sd": float(np.std(rh)), "r_tt_110": rtt, "noiseless_ceiling_110": float(np.sqrt(rtt))}
print(out); json.dump(out, open("../results/ceiling_all.json", "w"), indent=1)
