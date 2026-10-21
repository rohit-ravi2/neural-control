"""Distance-preserving shuffle control for the near-field connectome effect.
    python dist_shuffle.py -> results/dist_shuffle.json
"""
import json, os, pickle, sys

import numpy as np
import wormneuroatlas as wa_

import near_far as nf
import worm_data as wd

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402

c = lambda a, b, m: float(met.nan_corr(np.where(m, a, np.nan), np.where(m, b, np.nan))[0])
shipped = pickle.load(open(wd.REPO / "data" / "measured_stams.pkl", "rb"))
targets = {"T_ship": shipped["test"], "T_auth_ours": nf.TA["T_auth_ours"]}
ids = [str(x) for x in pickle.load(open(wd.REPO / "anatomical_data" / "cell_ids.pkl", "rb"))]
chem = np.asarray(pickle.load(open(wd.REPO / "anatomical_data" / "chemical.pkl", "rb")), float)
gap = np.asarray(pickle.load(open(wd.REPO / "anatomical_data" / "gap.pkl", "rb")), float)
A300 = np.log1p(chem) + np.log1p(gap)
lines = open(os.path.join(os.path.dirname(wa_.__file__), "data", "anatlas_neuron_positions.txt")).read().splitlines()
pn = lines[0].lstrip("#").split(); XYZ = np.array([[float(v) for v in l.split()] for l in lines[1:] if l.strip()])
pos = {n: XYZ[i] for i, n in enumerate(pn)}
cell = nf.cell; ix = [ids.index(cn) for cn in cell]
have = [i for i, cn in enumerate(cell) if cn in pos]
P = np.array([pos[cell[i]] for i in have])
Dc = np.linalg.norm(P[:, None] - P[None], axis=2); nbr = np.argsort(Dc, 1)[:, 1:7]
rng = np.random.default_rng(0)


def local_perm():
    perm = np.arange(len(have))
    for _ in range(5 * len(have)):            # random swaps restricted to canonical 6-NN pairs
        a = rng.integers(len(have)); b = nbr[a, rng.integers(6)]
        perm[[a, b]] = perm[[b, a]]
    return perm


def anat_for(perm):
    lab = np.arange(len(cell)); h = np.array(have); lab[h] = h[perm]
    return A300[np.ix_([ix[k] for k in lab], [ix[k] for k in lab])]


near = nf.near & np.isin(np.arange(len(cell)), have)[:, None] & np.isin(np.arange(len(cell)), have)[None, :]
real = anat_for(np.arange(len(have)))
out = {}
for tn, T in targets.items():
    r0 = c(real, T, near)
    null = [c(anat_for(local_perm()), T, near) for _ in range(1000)]
    out[tn] = {"real_near": r0, "null_mean": float(np.mean(null)), "null_p95": float(np.percentile(null, 95)), "p": float(np.mean(np.array(null) >= r0))}
    gnull = [c(anat_for(rng.permutation(len(have))), T, near) for _ in range(1000)]
    out[tn].update(global_null_mean=float(np.mean(gnull)), global_null_p95=float(np.percentile(gnull, 95)), p_global=float(np.mean(np.array(gnull) >= r0)))
    print(tn, json.dumps(out[tn]), flush=True)
json.dump(out, open("../results/dist_shuffle.json", "w"), indent=1)
