"""PEPTIDE-DiD: does the neuropeptidergic connectome predict causal signal propagation
beyond anatomy, and does that predictive power vanish in unc-31 (no dense-core-vesicle release) more than anatomy's?
    python peptide_did.py -> results/peptide_did.json
"""
import json

import numpy as np
from scipy.stats import rankdata
import wormneuroatlas as wa

wa.WormBase.assert_db_version_consistency = lambda self: None   # online version check fails; data are local
na = wa.NeuroAtlas(verbose=False)
N = len(na.neuron_ids)
P = np.log1p(na.get_peptidergic_connectome().astype(float))
A = np.log1p(na.get_chemical_synapses().astype(float)) + np.log1p(na.get_gap_junctions().astype(float))
ALL = {s: na.get_signal_propagation_map_all(strain=s) for s in ("wt", "unc31")}
OCC = {s: na.get_signal_propagation_occurrence_matrix(strain=s) for s in ("wt", "unc31")}
def nfin(x):
    try:
        v = np.asarray(x, float).ravel(); return int(np.isfinite(v).sum())
    except (TypeError, ValueError):
        return 0
FIN = {s: np.vectorize(nfin, otypes=[int])(ALL[s]) for s in ("wt", "unc31")}   # finite occurrences per pair
for s in FIN:
    print(s, "pairs with occ>=3:", int((OCC[s] >= 3).sum()), " with >=3 FINITE occurrences:", int((FIN[s] >= 3).sum()), flush=True)
OCC = FIN
pairs = np.argwhere((OCC["wt"] >= 3) & (OCC["unc31"] >= 3) & ~np.eye(N, dtype=bool))
ri, ci = pairs[:, 0], pairs[:, 1]
print("pairs with >=3 occurrences in both genotypes:", len(pairs), flush=True)


def resid(v, r, c):
    """residualise v on additive row + column effects (least squares, one-hot design)."""
    ur, uc = np.unique(r, return_inverse=True)[1], np.unique(c, return_inverse=True)[1]
    X = np.zeros((len(v), ur.max() + uc.max() + 2)); X[np.arange(len(v)), ur] = 1; X[np.arange(len(v)), ur.max() + 1 + uc] = 1
    beta = np.linalg.lstsq(X, v, rcond=None)[0]
    return v - X @ beta


def pspear(y, x, z):
    ry, rx, rz = (rankdata(a) for a in (y, x, z))
    def res(a, b):
        b1 = np.c_[np.ones_like(b), b]; return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    return float(np.corrcoef(res(ry, rz), res(rx, rz))[0, 1])


def matched_y(rng, idx):
    out = {}
    for s in ("wt", "unc31"):
        y = np.empty(len(idx))
        for k, (i, j) in enumerate(pairs[idx]):
            m = min(OCC["wt"][i, j], OCC["unc31"][i, j])
            v = np.asarray(ALL[s][i, j], float).ravel(); v = v[np.isfinite(v)]
            y[k] = abs(rng.choice(v, min(m, len(v)), replace=False).mean())
        out[s] = y
    return out


def stats(rng, idx, Pm=P):
    y = matched_y(rng, idx); r, c = ri[idx], ci[idx]
    p_, a_ = resid(Pm[r, c], r, c), resid(A[r, c], r, c)
    o = {}
    for s in ("wt", "unc31"):
        yr = resid(y[s], r, c)
        o[s] = {"P|A": pspear(yr, p_, a_), "A|P": pspear(yr, a_, p_)}
    o["DiD"] = (o["wt"]["P|A"] - o["unc31"]["P|A"]) - (o["wt"]["A|P"] - o["unc31"]["A|P"])
    return o


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    point = stats(rng, np.arange(len(pairs)))
    print("point:", point, flush=True)
    boot = [stats(rng, rng.integers(0, len(pairs), len(pairs))) for _ in range(200)]
    ci_ = lambda key: [float(np.percentile([key(b) for b in boot], q)) for q in (2.5, 97.5)]
    CI = {"wt P|A": ci_(lambda b: b["wt"]["P|A"]), "unc31 P|A": ci_(lambda b: b["unc31"]["P|A"]),
          "wt A|P": ci_(lambda b: b["wt"]["A|P"]), "unc31 A|P": ci_(lambda b: b["unc31"]["A|P"]), "DiD": ci_(lambda b: b["DiD"])}
    print("95% CI:", CI, flush=True)
    null = []
    for _ in range(200):
        pm = rng.permutation(N); Pp = P[np.ix_(pm, pm)]
        null.append(stats(rng, np.arange(len(pairs)), Pp))
    nwt = [b["wt"]["P|A"] for b in null]; nd = [b["DiD"] for b in null]
    res = {"n_pairs": int(len(pairs)), "point": point, "ci95": CI,
           "null_wt_P|A_p95": float(np.percentile(nwt, 95)), "null_DiD_p95": float(np.percentile(nd, 95)),
           "p_perm_wt_P|A": float(np.mean(np.array(nwt) >= point["wt"]["P|A"])), "p_perm_DiD": float(np.mean(np.array(nd) >= point["DiD"]))}
    print(json.dumps(res, indent=1))
    json.dump(res, open("../results/peptide_did.json", "w"), indent=1)
