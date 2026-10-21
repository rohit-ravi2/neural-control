"""Is Creamer's connectome advantage (CC vs shuffled) a near-field effect?
    python near_far_cc.py -> results/near_far_cc.json
"""
import json, pickle, sys

import numpy as np

import near_far as nf   # reuses D, near, far, cell, eye, TA (runs its model scoring on import: guarded below)
import worm_data as wd

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402
import lgssm_utilities as ssmu  # noqa: E402

c = lambda a, b, m: float(met.nan_corr(np.where(m, a, np.nan), np.where(m, b, np.nan))[0])
shipped = pickle.load(open(wd.REPO / "data" / "measured_stams.pkl", "rb"))
targets = {"T_ship": shipped["test"], "T_auth_ours": nf.TA["T_auth_ours"]}
P = {}
for mn in ("connectome_constrained", "shuffled_constrained", "fully_connected"):
    mdl = pickle.load(open(wd.REPO / "models" / f"{mn}.pkl", "rb")); P[mn] = ssmu.calculate_irfs(mdl, window=(15, 30))[15:].sum(0)
ids = [str(x) for x in pickle.load(open(wd.REPO / "anatomical_data" / "cell_ids.pkl", "rb"))]
chem = np.asarray(pickle.load(open(wd.REPO / "anatomical_data" / "chemical.pkl", "rb")), float)
gap = np.asarray(pickle.load(open(wd.REPO / "anatomical_data" / "gap.pkl", "rb")), float)
ix = [ids.index(cname) for cname in nf.cell]
P["anatomy (chem+gap, i<-j)"] = (np.log1p(chem) + np.log1p(gap))[np.ix_(ix, ix)]
P["anatomy transposed"] = P["anatomy (chem+gap, i<-j)"].T
masks = {"all": ~nf.eye, "near<10": nf.near, "far>=15": nf.far}
out = {}
for tn, T in targets.items():
    for k, M in P.items():
        out[f"{tn} | {k}"] = {mk: c(M, T, m) for mk, m in masks.items()}
        print(f"{tn:11s} {k:28s} " + "  ".join(f"{mk} {v:.3f}" for mk, v in out[f'{tn} | {k}'].items()), flush=True)
json.dump(out, open("../results/near_far_cc.json", "w"), indent=1)
