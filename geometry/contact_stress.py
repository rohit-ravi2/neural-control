"""Stress tests for the contact positive (geometry/contact_geometry.py): nerve-ring membrane contact predicts the causal atlas beyond
soma distance and synapses (partial rho ~0.04-0.055, distance-preserving null p = 0.003).

  S1 main effects: responders/targets with a lot of contact might simply respond/drive more. Covariates add each neuron's total
     contact (responder and target side) and leave-one-out row/column means of the atlas itself.
  S2 degree-preserving null: contact graph rewired by double-edge swaps (each neuron keeps its contact partners' count and
     weights travel with edges), so only WHO touches WHOM is scrambled.
  S3 same-class pairs excluded (L/R and D/V homologs share bundles and responses).
  S4 synapse-free pairs only (A = 0 in both directions): is contact informative where the connectome is silent?
    python geometry/contact_stress.py -> geometry/results/contact_stress.json
"""
import json, pathlib, sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import contact_geometry as cg  # noqa: E402
import worm_models as wm  # noqa: E402

rng = np.random.default_rng(1)
nC, eye, D, A, A_sym = cg.nC, cg.eye, cg.D, cg.A, cg.A_sym
cls = np.array([wm.neuron_class(c) for c in cg.cell]); same_cls = cls[:, None] == cls[None, :]


def loo_means(T):
    """Leave-one-out row and column means of T (nan-aware)."""
    F = np.isfinite(T); Tz = np.where(F, T, 0)
    rs, rn = Tz.sum(1, keepdims=True), F.sum(1, keepdims=True); cs, cn = Tz.sum(0, keepdims=True), F.sum(0, keepdims=True)
    row = (rs - Tz) / np.maximum(rn - F, 1); col = (cs - Tz) / np.maximum(cn - F, 1)
    return row, col


def rewire(C, cov):
    """Degree-preserving double-edge swap on the undirected contact graph among covered cells (weights travel with edges)."""
    idx = np.where(cov)[0]; S = np.nan_to_num(C[np.ix_(idx, idx)]); iu = np.triu_indices(len(idx), 1)
    e = np.where(S[iu] > 0)[0]; a, b, w = iu[0][e].copy(), iu[1][e].copy(), S[iu][e].copy()
    E = set(zip(a, b))
    for _ in range(10 * len(a)):
        i, j = rng.integers(len(a), size=2)
        if len({a[i], b[i], a[j], b[j]}) < 4:
            continue
        n1, n2 = (min(a[i], b[j]), max(a[i], b[j])), (min(a[j], b[i]), max(a[j], b[i]))
        if n1 in E or n2 in E:
            continue
        E -= {(min(a[i], b[i]), max(a[i], b[i])), (min(a[j], b[j]), max(a[j], b[j]))}; E |= {n1, n2}
        b[i], b[j] = b[j], b[i]
    R = np.zeros_like(S); R[a, b] = w; R[b, a] = w
    out = np.full_like(C, np.nan); out[np.ix_(idx, idx)] = R; out[eye] = np.nan
    return out


def main():
    res = {}
    for cname, (C, cov) in cg.CONTACT.items():
        deg = np.nansum(C, 1)
        for tn, T in cg.TARGETS.items():
            row, col = loo_means(T)
            base = cov[:, None] & cov[None, :] & ~eye & np.isfinite(T)
            masks = {"all": base, "excl_same_class": base & ~same_cls, "synapse_free": base & (A_sym == 0),
                     "far>=15_synapse_free": base & (A_sym == 0) & (D >= 15)}

            def stats(Cx, m):
                t, c = T[m], Cx[m]
                covs = {"D,A": [-D[m], A[m]], "D,A,main": [-D[m], A[m], row[m], col[m], deg[:, None].repeat(nC, 1)[m], deg[None, :].repeat(nC, 0)[m]]}
                return {k: cg.pspear(t, c, Z) for k, Z in covs.items()}

            out = {}
            for mn, m in masks.items():
                real = stats(C, m); null = {k: [] for k in real}
                for _ in range(200):
                    Cn = rewire(C, cov); mm = m & np.isfinite(Cn)
                    for k, v in stats(Cn, mm).items():
                        null[k].append(v)
                out[mn] = {"n": int(m.sum()), **{f"{k}": real[k] for k in real},
                           **{f"{k}:degnull_p95": float(np.percentile(null[k], 95)) for k in real},
                           **{f"{k}:p": float((np.sum(np.array(null[k]) >= real[k]) + 1) / 201) for k in real}}
            res[f"{tn}|{cname}"] = out
            print(tn, cname, json.dumps({mn: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()} for mn, o in out.items()}), flush=True)
            json.dump(res, open(cg.OUT / "contact_stress.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
