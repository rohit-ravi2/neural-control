"""Confound test for the banked kNN prior (both, k=40, T=5; selected on training animals): does held-out prediction survive
when neurons of the same BASE class (CeNGEN subclass siblings: RMD_DV/RMD_LR, IL2_DV/IL2_LR, RME_DV/RME_LR, DB/DB01,
VB/VB01/VB02, ASEL/ASER, DA/DA9, VA/VA12, VC/VC_4_5, AWC_*) are also excluded from the similarity pool?
    python knn_confound.py -> results/knn_confound.json
"""
import json, re

import numpy as np

import knn_prior as kp

zn = lambda G: G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
SRC = {"cengen": kp.Ge, "connectome": kp.Gc, "both": np.concatenate([zn(kp.Ge), zn(kp.Gc)], 1)}


def base(c):
    if c is None:
        return None
    c = re.sub(r"_(DV|LR|ON|OFF|4_5)$", "", c)
    c = {"ASEL": "ASE", "ASER": "ASE", "DB01": "DB", "VB01": "VB", "VB02": "VB", "DA9": "DA", "VA12": "VA"}.get(c, c)
    return c


bases = np.array([base(c) or "" for c in kp.cls_of])
_predict = kp.predict


def predict_excl(G, h, k, excl):
    """kp.predict with an extra per-neuron exclusion of same-base neurons."""
    Z = zn(G); S = Z @ Z.T; n = len(h)
    P = np.full((n, n), np.nan)
    for i in np.where(h)[0]:
        src = np.where(~h & ((bases != bases[i]) if excl else True))[0]
        nb = src[np.argsort(-S[i, src])[:k]]
        w = np.exp(S[i, nb] * 5); w /= w.sum()
        with np.errstate(all="ignore"):
            P[i, :] = np.nansum(kp.TR[nb] * w[:, None], 0) / np.nansum(np.isfinite(kp.TR[nb]) * w[:, None], 0)
            col = np.nansum(kp.TR[:, nb] * w[None, :], 1) / np.nansum(np.isfinite(kp.TR[:, nb]) * w[None, :], 1)
        P[~h, i] = col[~h]
    return P


out = {}
for s, G in SRC.items():
    for excl in (False, True):
        sc = []
        for f in range(5):
            held = set(np.random.default_rng(12345).permutation(kp.all_cls)[f::5])
            h = np.array([c in held for c in kp.cls_of])
            sc.append(kp.corr(predict_excl(G, h, 40, excl), kp.T, h[:, None] | h[None, :]))
        out[f"{s}_{'excl_siblings' if excl else 'standard'}"] = {"mean": float(np.mean(sc)), "per_fold": sc}
        print(f"{s:10s} {'exclude same-base' if excl else 'standard':18s} {np.mean(sc):.3f}  {np.round(sc, 3)}", flush=True)
n_sib = sum(1 for c in set(kp.cls_of) if c and sum(base(x) == base(c) for x in set(kp.cls_of) if x) > 1)
print("classes with a sibling subclass among the 80:", n_sib)
json.dump(out, open("../results/knn_confound.json", "w"), indent=1)
