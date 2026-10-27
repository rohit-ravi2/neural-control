"""Geometry test 1, neuropil arm (exploratory): soma geometry vs nerve-ring CONTACT geometry vs synapses, against the causal atlas.

Wiring economy acts where the wires are: head synapses form in the nerve-ring neuropil, so soma-soma distance (the axis of
the locality finding) is not the geometry wiring minimisation shapes. Here three layers compete for the atlas:
  D  soma distance (per-animal median, µm; Randi segmentation)            -> the near-field locality axis
  C  nerve-ring membrane contact (adult / L4 'nerve ring neighbors' sheet,  -> wiring-relevant geometry
     data/connectome/nerve_ring_neighbors/; log1p contact)
  A  synapses (log1p chemical + log1p gap)                                   -> the connectome
Questions:
  (a) Does the atlas's soma-proximity effect survive neuropil contact + synapses (confound: spill / soma-level coupling),
      or is it carried by contact (substrate: geometry -> contact -> function)?
  (b) Is the distance-surviving causal signal carried by contact beyond synapses (unused contact; ephaptic / unannotated),
      and does contact predict FAR-field (soma ≥ 15 µm) responses the connectome misses?
  (c) The 1b spontaneous co-activity partial: does it survive contact?
    python geometry/contact_geometry.py  -> geometry/results/contact_geometry.json
"""
import json, os, pathlib, pickle, sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import near_far as nf  # noqa: E402
import worm_data as wd  # noqa: E402
import worm_models as wm  # noqa: E402
import wormneuroatlas as wa  # noqa: E402

rng = np.random.default_rng(0)
OUT = ROOT / "geometry" / "results"; OUT.mkdir(parents=True, exist_ok=True)
XL = ROOT / "data" / "connectome" / "nerve_ring_neighbors" / "Adult and L4 nerve ring neighbors.xlsx"
UM = 111.4

cell = nf.cell; nC = len(cell); eye = np.eye(nC, dtype=bool)
AD = wd.REPO / "anatomical_data"
ids = [str(x) for x in pickle.load(open(AD / "cell_ids.pkl", "rb"))]; ix = {n: i for i, n in enumerate(ids)}; cix = [ix[c] for c in cell]
chem = np.asarray(pickle.load(open(AD / "chemical.pkl", "rb")), float); gap = np.asarray(pickle.load(open(AD / "gap.pkl", "rb")), float)
A = np.log1p(chem[np.ix_(cix, cix)]) + np.log1p(gap[np.ix_(cix, cix)])
A_sym = np.log1p((chem + chem.T + gap)[np.ix_(cix, cix)])
L = open(os.path.join(os.path.dirname(wa.__file__), "data", "anatlas_neuron_positions.txt")).read().splitlines()
pn = L[0].lstrip("#").split(); XYZ = np.array([[float(v) for v in l.split()] for l in L[1:] if l.strip()]) * UM
POS = dict(zip(pn, XYZ)); Pc = np.array([POS[c] for c in cell])
D = np.where(np.isfinite(nf.D), nf.D, np.linalg.norm(Pc[:, None] - Pc[None], axis=2))


def contact(sheet):
    d = pd.read_excel(XL, sheet_name=sheet, header=None)
    names = [str(x).strip() for x in d.iloc[0, 1:]]; rows = [str(x).strip() for x in d.iloc[1:, 0]]
    M = d.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy()   # the sheet repeats the labels in a trailing row/column
    # rows and columns share one order; one label disagrees in the source (row SABD vs column SABVR): drop it, map by position
    k = {n: i for i, n in enumerate(rows) if n != "nan" and i < len(names) and names[i] == n}; kc = k
    C = np.full((nC, nC), np.nan); covered = np.array([c in k for c in cell])
    for a, ca in enumerate(cell):
        for b, cb in enumerate(cell):
            if ca in k and cb in k and a != b:
                v = M[k[ca], kc[cb]]; C[a, b] = np.log1p(v) if np.isfinite(v) else 0.0
    C = np.where(np.isfinite(C) & np.isfinite(C.T), (C + C.T) / 2, C)    # symmetrise (sheet is near-symmetric)
    return C, covered


CONTACT = {s.split()[0]: contact(s) for s in ("adult nerve ring neighbors", "L4 nerve ring neighbors")}
for k, (C, cov) in CONTACT.items():
    print(k, "cells covered", int(cov.sum()), "/", nC, "| contact density", float(np.nanmean(C[~eye] > 0)), flush=True)

wa.WormBase.assert_db_version_consistency = lambda self: None
na_ = wa.NeuroAtlas(verbose=False); nid = [str(x) for x in na_.neuron_ids]; nix = {n: i for i, n in enumerate(nid)}
dff = na_.get_signal_propagation_map("wt").astype(float); occ = na_.get_signal_propagation_occurrence_matrix("wt")
dff = np.where(occ >= 3, dff, np.nan)
fun = np.array([[dff[nix[a], nix[b]] if a in nix and b in nix else np.nan for b in cell] for a in cell])
shipped = pickle.load(open(wd.REPO / "data" / "measured_stams.pkl", "rb"))
TARGETS = {"T_auth_all": nf.TA["T_auth_all"], "T_ship": shipped["test"], "funatlas_wt_dFF": fun}


def pspear(y, x, Z):
    ry, rx = rankdata(y), rankdata(x); RZ = np.column_stack([np.ones(len(y))] + [rankdata(z) for z in Z])
    e = lambda v: v - RZ @ np.linalg.lstsq(RZ, v, rcond=None)[0]
    return float(np.corrcoef(e(ry), e(rx))[0, 1])


# local (distance-preserving) permutation of identities among canonical 6-NN, as in src/dist_shuffle.py
Dcan = np.linalg.norm(Pc[:, None] - Pc[None], axis=2); nbr = np.argsort(Dcan, 1)[:, 1:7]


def local_perm():
    perm = np.arange(nC)
    for _ in range(5 * nC):
        a = rng.integers(nC); b = nbr[a, rng.integers(6)]; perm[[a, b]] = perm[[b, a]]
    return perm


def main():
    res = {}
    for cname, (C, cov) in CONTACT.items():
        pair_ok = cov[:, None] & cov[None, :] & ~eye
        s = {"rho_C_minusD": float(spearmanr(C[pair_ok], -D[pair_ok])[0]), "rho_C_Asym": float(spearmanr(C[pair_ok], A_sym[pair_ok])[0]),
             "frac_synaptic_pairs_with_contact": float(np.mean(C[pair_ok & (A_sym > 0)] > 0)),
             "frac_contact_pairs_with_synapse": float(np.mean(A_sym[pair_ok & (C > 0)] > 0))}
        res[f"structure|{cname}"] = s; print("structure", cname, json.dumps({k: round(v, 3) for k, v in s.items()}), flush=True)
        for tn, T in TARGETS.items():
            m = pair_ok & np.isfinite(T)
            t, d, c, a = T[m], -D[m], C[m], A[m]
            r = {"n": int(m.sum()), "T~-D": float(spearmanr(t, d)[0]), "T~C": float(spearmanr(t, c)[0]), "T~A": float(spearmanr(t, a)[0]),
                 "T~-D|C,A": pspear(t, d, [c, a]), "T~C|D,A": pspear(t, c, [d, a]), "T~A|D": pspear(t, a, [d]), "T~A|D,C": pspear(t, a, [d, c])}
            # near and far strata by soma distance
            for zone, zm in (("near<10", D[m] < 10), ("far>=15", D[m] >= 15)):
                r[f"{zone}:n"] = int(zm.sum())
                r[f"{zone}:T~C|D,A"] = pspear(t[zm], c[zm], [d[zm], a[zm]]); r[f"{zone}:T~A|D,C"] = pspear(t[zm], a[zm], [d[zm], c[zm]])
                r[f"{zone}:T~A|D"] = pspear(t[zm], a[zm], [d[zm]])
                # unused contact: contact without synapse vs neither, mean T difference within soma-distance quintiles
                nos = zm & (a == 0); dif = []
                qb = np.percentile(-d[nos], [0, 20, 40, 60, 80, 100])
                for lo, hi in zip(qb[:-1], qb[1:]):
                    bm = nos & (-d >= lo) & (-d <= hi)
                    if (bm & (c > 0)).sum() >= 10 and (bm & (c == 0)).sum() >= 10:
                        dif.append((np.nanmean(t[bm & (c > 0)]) - np.nanmean(t[bm & (c == 0)])) / np.nanstd(t[nos]))
                r[f"{zone}:unused_contact_effect_sd"] = float(np.mean(dif)) if dif else None
            # distance-preserving null for T~C|D,A (contact identities swapped among canonical neighbours)
            null = []
            for _ in range(300):
                p = local_perm(); Cp = C[np.ix_(p, p)]; mm = m & np.isfinite(Cp)
                null.append(pspear(T[mm], Cp[mm], [-D[mm], A[mm]]))
            r["T~C|D,A_localnull_p95"] = float(np.percentile(null, 95)); r["T~C|D,A_p"] = float((np.sum(np.array(null) >= r["T~C|D,A"]) + 1) / 301)
            res[f"{tn}|{cname}"] = r
            print(tn, cname, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    json.dump(res, open(OUT / "contact_geometry.json", "w"), indent=1)

    # (c) 1b: spontaneous residual co-activity (Atanas, class level, top-3 PCs removed) vs causal | anatomy, distance, contact
    import atanas_data as ad
    S0 = {}
    for name, x, ids_ in ad.load():
        cl = np.array([wm.neuron_class(i) if i else "" for i in ids_]); U = sorted(set(cl) - {""})
        Xc = np.stack([x[:, cl == c].mean(1) for c in U], 1); Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-6)
        Uv = np.linalg.svd(x - x.mean(0), full_matrices=False)[0][:, :3]; Xc = Xc - Uv @ (Uv.T @ Xc)
        G0 = np.corrcoef(Xc.T)
        for a_, c in enumerate(U):
            for b_, s_ in enumerate(U):
                if a_ != b_:
                    S0.setdefault((c, s_), []).append(G0[a_, b_])
    gz = {p: float(np.mean(v)) for p, v in S0.items() if len(v) >= 5}
    ac = np.array([wm.neuron_class(n) for n in nid]); cc = np.array([wm.neuron_class(c) for c in cell])


    def cmean(M, rc, colc):
        o = {}
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if i != j and np.isfinite(M[i, j]) and rc[i] and colc[j] and rc[i] != colc[j]:
                    o.setdefault((rc[i], colc[j]), []).append(M[i, j])
        return {k: float(np.mean(v)) for k, v in o.items()}


    Cc = cmean(dff, ac, ac)
    anat300 = np.log1p(na_.get_chemical_synapses().astype(float)) + np.log1p(na_.get_gap_junctions().astype(float))
    Ac = cmean(anat300, ac, ac); Dc = cmean(D, cc, cc)
    for cname, (C, cov) in CONTACT.items():
        Kc = cmean(C, cc, cc)
        both = [p for p in gz if p in Cc and p in Ac and p in Dc and p in Kc]
        y, cv, av, dv, kv = (np.array([M[p] for p in both]) for M in (gz, Cc, Ac, Dc, Kc))
        r = {"n_class_pairs": len(both), "spont~causal|anat,dist": pspear(y, cv, [av, dv]), "spont~causal|anat,dist,contact": pspear(y, cv, [av, dv, kv]),
             "spont~contact|anat,dist": pspear(y, kv, [av, dv]), "spont~contact|anat,dist,causal": pspear(y, kv, [av, dv, cv]),
             "causal~contact|anat,dist": pspear(cv, kv, [av, dv])}
        cls_ = sorted({c for p in Cc for c in p}); null = []
        for _ in range(300):
            mp = dict(zip(cls_, rng.permutation(cls_)))
            v = np.array([Cc.get((mp[a_], mp[b_]), np.nan) for a_, b_ in both]); okv = np.isfinite(v)
            null.append(pspear(y[okv], v[okv], [av[okv], dv[okv], kv[okv]]))
        r["z_causal|anat,dist,contact"] = float((r["spont~causal|anat,dist,contact"] - np.mean(null)) / np.std(null))
        r["p_causal|anat,dist,contact"] = float((np.sum(np.abs(null) >= abs(r["spont~causal|anat,dist,contact"])) + 1) / 301)
        res[f"spont1b|{cname}"] = r
        print("1b", cname, json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
    json.dump(res, open(OUT / "contact_geometry.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
