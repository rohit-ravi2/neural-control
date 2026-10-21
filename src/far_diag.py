"""Far-field diagnostics. Requires ensemble_final.py run first (re-computes the A+B STAM).
    python far_diag.py -> results/far_diag.json
"""
import json, sys

import numpy as np

import near_far as nf
import worm_data as wd

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402

T, TR = nf.TA["T_auth_ours"], nf.TA["T_auth_train"]
c = lambda a, m: float(met.nan_corr(np.where(m, a, np.nan), np.where(m, T, np.nan))[0])
masks = {"near": nf.near, "far": nf.far, "all": ~nf.eye}


def impute_lowrank(M, k, it=60):
    X = np.where(np.isfinite(M), M, 0.0); obs = np.isfinite(M); Z = X.copy()
    for _ in range(it):
        U, S, Vt = np.linalg.svd(Z, full_matrices=False); L = (U[:, :k] * S[:k]) @ Vt[:k]; Z = np.where(obs, X, L)
    return L


out = {}
rm, cm, gm = np.nanmean(TR, 1), np.nanmean(TR, 0), np.nanmean(TR)
out["train additive row+col"] = {k: c(rm[:, None] + cm[None, :] - gm, m) for k, m in masks.items()}
for k in (1, 2, 3, 5, 10):
    out[f"train SVD rank {k}"] = {kk: c(impute_lowrank(TR, k), m) for kk, m in masks.items()}
P = np.load("../results/ensemble_final_AB.npy")
Pf = np.where(np.isfinite(P), P, 0.0)
U, S, Vt = np.linalg.svd(Pf, full_matrices=False)
for k in (1, 2, 3, 5, 10, 20, 50):
    out[f"model rank {k}"] = {kk: c((U[:, :k] * S[:k]) @ Vt[:k], m) for kk, m in masks.items()}
out["model full"] = {kk: c(P, m) for kk, m in masks.items()}
rmP, cmP, gmP = np.nanmean(P, 1), np.nanmean(P, 0), np.nanmean(P)
out["model additive row+col only"] = {kk: c(rmP[:, None] + cmP[None, :] - gmP, m) for kk, m in masks.items()}
for k, v in out.items():
    print(f"{k:30s} " + "  ".join(f"{kk} {vv:.3f}" for kk, vv in v.items()))
json.dump(out, open("../results/far_diag.json", "w"), indent=1)
