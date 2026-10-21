"""Within-near-field dissection of Creamer's connectome advantage.
    python near_dissect.py -> results/near_dissect.json
"""
import json, pickle, sys

import numpy as np
from scipy.stats import rankdata

import near_far as nf
import worm_data as wd

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402
import lgssm_utilities as ssmu  # noqa: E402

c = lambda a, b, m: float(met.nan_corr(np.where(m, a, np.nan), np.where(m, b, np.nan))[0])
shipped = pickle.load(open(wd.REPO / "data" / "measured_stams.pkl", "rb"))
targets = {"T_ship": shipped["test"], "T_auth_ours": nf.TA["T_auth_ours"]}
P = {mn: ssmu.calculate_irfs(pickle.load(open(wd.REPO / "models" / f"{mn}.pkl", "rb")), window=(15, 30))[15:].sum(0)
     for mn in ("connectome_constrained", "shuffled_constrained")}
ids = [str(x) for x in pickle.load(open(wd.REPO / "anatomical_data" / "cell_ids.pkl", "rb"))]
ix = [ids.index(cn) for cn in nf.cell]
chem = np.asarray(pickle.load(open(wd.REPO / "anatomical_data" / "chemical.pkl", "rb")), float)[np.ix_(ix, ix)]
gap = np.asarray(pickle.load(open(wd.REPO / "anatomical_data" / "gap.pkl", "rb")), float)[np.ix_(ix, ix)]
P["anatomy"] = np.log1p(chem) + np.log1p(gap)
D, eye = nf.D, nf.eye
strata = {"<5": (D < 5), "5-10": (D >= 5) & (D < 10), "10-15": (D >= 10) & (D < 15), ">=15": (D >= 15)}
out = {"strata": {}}
for tn, T in targets.items():
    for sn, m in strata.items():
        m = m & ~eye & np.isfinite(T)
        out["strata"][f"{tn} {sn}"] = {"n": int(m.sum()), **{k: c(M, T, m) for k, M in P.items()}}
        print(tn, sn, out["strata"][f"{tn} {sn}"], flush=True)


def pspear(y, x, z):
    ry, rx, rz = (rankdata(a) for a in (y, x, z))
    res = lambda a: a - np.c_[np.ones_like(rz), rz] @ np.linalg.lstsq(np.c_[np.ones_like(rz), rz], a, rcond=None)[0]
    return float(np.corrcoef(res(ry), res(rx))[0, 1])


rng = np.random.default_rng(0); out["partial_given_distance"] = {}
for tn, T in targets.items():
    m = (D < 10) & ~eye & np.isfinite(T)
    for k in ("connectome_constrained", "shuffled_constrained", "anatomy"):
        y, x, z = T[m], P[k][m], D[m]
        ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(z); y, x, z = y[ok], x[ok], z[ok]
        pt = pspear(y, x, z); bs = [pspear(*(a[idx] for a in (y, x, z))) for idx in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
        out["partial_given_distance"][f"{tn} {k}"] = {"rho": pt, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(len(y))}
        print(tn, k, "partial rho | distance (near<10):", out["partial_given_distance"][f"{tn} {k}"], flush=True)
json.dump(out, open("../results/near_dissect.json", "w"), indent=1)
