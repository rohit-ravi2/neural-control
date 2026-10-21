"""Angle 1b (exploratory): does the perturbation atlas predict pairwise LAGGED coupling in spontaneous, freely moving data
once the global brain state is removed?
For each Atanas animal: z-scored class-mean traces; residualise on the top-k PCs of that animal (global state);
G[c, s] = corr(r_s(t), r_c(t+h) - r_c(t)) (source s leads target c). Average G over animals per class pair; compare to
the Randi wt dFF map (both orientations), anatomy, and a class-permutation null.
    python spont_coupling.py [--k 3] [--h 2]  -> results/forecast/spont_coupling_k{k}_h{h}.json
"""
import argparse, json, pathlib

import numpy as np
from scipy.stats import spearmanr
import wormneuroatlas as wa

import atanas_data as ad
import worm_models as wm

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=3); ap.add_argument("--h", type=int, default=2)
ap.add_argument("--min-animals", type=int, default=5)
args = ap.parse_args()
wa.WormBase.assert_db_version_consistency = lambda self: None
na = wa.NeuroAtlas(verbose=False)

A = ad.load(); S, C, S0 = {}, {}, {}
for name, x, ids in A:
    cl = np.array([wm.neuron_class(i) if i else "" for i in ids]); U = sorted(set(cl) - {""})
    X = np.stack([x[:, cl == c].mean(1) for c in U], 1); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    if args.k:
        Uv = np.linalg.svd(x - x.mean(0), full_matrices=False)[0][:, :args.k]     # global state from ALL neurons of the animal
        X = X - Uv @ (Uv.T @ X)
    h = args.h; X0, X1 = X[:-h], X[h:]
    # Granger-style partial: corr(c(t+h), s(t) | c(t)) -> residualise both on the target's own present
    b1 = (X0 * X1).sum(0) / (X0 ** 2).sum(0); R1 = X1 - X0 * b1                   # target future | own present
    Bs = (X0.T @ X0) / (X0 ** 2).sum(0)[None, :]                                   # Bs[s, c]: regress s(t) on c(t)
    G = np.zeros((len(U), len(U)))
    for a in range(len(U)):
        Rs = X0 - X0[:, [a]] * Bs[:, a][None, :]                                    # sources | target present
        G[a] = (R1[:, a] @ Rs) / (np.linalg.norm(R1[:, a]) * np.linalg.norm(Rs, axis=0) + 1e-9)
    G0 = np.corrcoef(X.T)                          # zero-lag residual correlation
    for a, c in enumerate(U):
        for b, s in enumerate(U):
            if a != b:
                S.setdefault((c, s), []).append(G[a, b]); S0.setdefault((c, s), []).append(G0[a, b])

pairs = [p for p, v in S.items() if len(v) >= args.min_animals]
g = np.array([np.mean(S[p]) for p in pairs]); g0 = np.array([np.mean(S0[p]) for p in pairs])

ac = np.array([wm.neuron_class(str(n)) for n in na.neuron_ids])


def class_matrix(M):
    out = {}
    ok = np.isfinite(M); r, c = np.nonzero(ok & ~np.eye(len(M), dtype=bool))
    for i, j in zip(r, c):
        out.setdefault((ac[i], ac[j]), []).append(M[i, j])
    return {k: float(np.mean(v)) for k, v in out.items()}


dff = na.get_signal_propagation_map("wt").astype(float)          # [responder, stimulated]
occ = na.get_signal_propagation_occurrence_matrix("wt")
dff = np.where(occ >= 3, dff, np.nan)
anat = np.log1p(na.get_chemical_synapses().astype(float)) + np.log1p(na.get_gap_junctions().astype(float))   # [post, pre]
PRI = {"causal[resp,stim]": class_matrix(dff), "causal_T": class_matrix(dff.T), "anatomy[post,pre]": class_matrix(anat),
       "anatomy_T": class_matrix(anat.T)}
rng = np.random.default_rng(0); res = {"args": vars(args), "n_pairs_spont": len(pairs)}
for nm, P in PRI.items():
    idx = [i for i, p in enumerate(pairs) if p in P]
    pv = np.array([P[pairs[i]] for i in idx])
    out = {"n": len(idx)}
    for lab, y in [("lagged", g[idx]), ("zerolag", g0[idx])]:
        rho = spearmanr(pv, y)[0]
        # null: permute class labels of the prior (keeps the prior's marginal structure)
        cls_ = sorted({c for p in P for c in p}); null = []
        for _ in range(200):
            m = dict(zip(cls_, rng.permutation(cls_)))
            pv_n = np.array([P.get((m[pairs[i][0]], m[pairs[i][1]]), np.nan) for i in idx]); ok = np.isfinite(pv_n)
            null.append(spearmanr(pv_n[ok], y[ok])[0])
        out[lab] = {"rho": float(rho), "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
                    "z": float((rho - np.mean(null)) / np.std(null)), "p_perm": float((np.sum(np.abs(null) >= abs(rho)) + 1) / 201)}
    res[nm] = out
    print(nm, json.dumps({k: (v if not isinstance(v, dict) else {a: round(b, 3) for a, b in v.items()}) for k, v in out.items()}), flush=True)
# stress tests on the symmetric (zero-lag / lagged) correspondence: partial out anatomy and class-pair physical distance
import worm_data as wd
POS = dict(np.load(pathlib.Path(__file__).resolve().parents[1].joinpath("work/roi_pos.npz").expanduser()))
DD = {}
for rec, P_ in POS.items():
    ids = [wm.neuron_class(str(x)) if str(x) else "" for x in wd.load_recording(rec)["cell_ids"]]
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if a and b and a != b and i < len(P_) and j < len(P_):
                DD.setdefault((a, b), []).append(float(np.linalg.norm(P_[i] - P_[j])))
DIST = {k: float(np.median(v)) for k, v in DD.items()}


def pspear(y, x, Z):
    from scipy.stats import rankdata
    ry, rx = rankdata(y), rankdata(x); RZ = np.column_stack([np.ones(len(y))] + [rankdata(z) for z in Z])
    e = lambda v: v - RZ @ np.linalg.lstsq(RZ, v, rcond=None)[0]
    return float(np.corrcoef(e(ry), e(rx))[0, 1])


gz = dict(zip(pairs, g0)); gl = dict(zip(pairs, g)); C_, A_ = PRI["causal[resp,stim]"], PRI["anatomy[post,pre]"]
both = [p for p in pairs if p in C_ and p in A_ and p in DIST]
cv, av, dv = (np.array([M[p] for p in both]) for M in (C_, A_, DIST))
for lab, D in [("zerolag", gz), ("lagged", gl)]:
    y = np.array([D[p] for p in both])
    r = {"n": len(both), "causal": float(spearmanr(cv, y)[0]), "anat": float(spearmanr(av, y)[0]), "dist": float(spearmanr(dv, y)[0]),
         "causal|anat": pspear(y, cv, [av]), "causal|dist": pspear(y, cv, [dv]), "causal|anat,dist": pspear(y, cv, [av, dv]),
         "anat|causal,dist": pspear(y, av, [cv, dv]), "causal~dist": float(spearmanr(cv, dv)[0])}
    far = dv > 10
    r["far>10um_n"] = int(far.sum()); r["far_causal"] = float(spearmanr(cv[far], y[far])[0]); r["far_anat"] = float(spearmanr(av[far], y[far])[0])
    cls_ = sorted({c for p in C_ for c in p}); null = []
    for _ in range(300):
        m = dict(zip(cls_, rng.permutation(cls_)))
        v = np.array([C_.get((m[a], m[b]), np.nan) for a, b in both]); ok = np.isfinite(v)
        null.append(pspear(y[ok], v[ok], [av[ok], dv[ok]]))
    r["causal|anat,dist_z"] = float((r["causal|anat,dist"] - np.mean(null)) / np.std(null))
    r["causal|anat,dist_p"] = float((np.sum(np.abs(np.array(null)) >= abs(r["causal|anat,dist"])) + 1) / 301)
    res["STRESS_" + lab] = r; print("STRESS", lab, json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)

# directionality: does the asymmetry of the prior predict the asymmetry of spontaneous lead-lag?
gd = dict(zip(pairs, g))
for nm in ["causal[resp,stim]", "anatomy[post,pre]"]:
    P = PRI[nm]; und = [(c, s_) for (c, s_) in pairs if c < s_ and (s_, c) in gd and (c, s_) in P and (s_, c) in P]
    asp = np.array([gd[(c, s_)] - gd[(s_, c)] for c, s_ in und]); apr = np.array([P[(c, s_)] - P[(s_, c)] for c, s_ in und])
    rho = spearmanr(apr, asp)[0]; cls_ = sorted({c for p in P for c in p}); null = []
    for _ in range(500):
        m = dict(zip(cls_, rng.permutation(cls_)))
        v = np.array([P.get((m[c], m[s_]), np.nan) - P.get((m[s_], m[c]), np.nan) for c, s_ in und]); ok = np.isfinite(v)
        null.append(spearmanr(v[ok], asp[ok])[0])
    res[nm + "_ASYM"] = {"n": len(und), "rho": float(rho), "null_sd": float(np.std(null)), "z": float(rho / np.std(null)),
                        "p_perm": float((np.sum(np.abs(null) >= abs(rho)) + 1) / 501)}
    print(nm, "ASYMMETRY", json.dumps(res[nm + "_ASYM"]), flush=True)
OUT = pathlib.Path(__file__).resolve().parents[1].joinpath("results/forecast").expanduser()
json.dump(res, open(OUT / f"spont_coupling_k{args.k}_h{args.h}.json", "w"), indent=1)
