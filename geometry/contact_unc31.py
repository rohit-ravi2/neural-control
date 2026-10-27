"""Mechanism probe for the contact positive (exploratory): does contact's predictive power beyond soma distance + synapses survive
in unc-31 (no dense-core-vesicle / neuropeptide release; Randi 2023 funatlas)? Peptidergic short-range signalling predicts it
vanishes in unc-31; unannotated synapses / gap junctions / ephaptic coupling predict it persists.
Pairs measured (occurrence ≥ 3) in BOTH genotypes, so the comparison is on identical pairs; bootstrap over pairs for the
wt - unc31 difference; degree-preserving contact null per genotype (contact_stress.rewire).
    python geometry/contact_unc31.py -> geometry/results/contact_unc31.json
"""
import json, pathlib, sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import contact_geometry as cg  # noqa: E402
import contact_stress as cs  # noqa: E402

rng = np.random.default_rng(2)
na_ = cg.na_; nix = cg.nix; cell = cg.cell
MAP = {}
for s in ("wt", "unc31"):
    dff = na_.get_signal_propagation_map(s).astype(float); occ = na_.get_signal_propagation_occurrence_matrix(s)
    dff = np.where(occ >= 3, dff, np.nan)
    MAP[s] = np.array([[dff[nix[a], nix[b]] if a in nix and b in nix else np.nan for b in cell] for a in cell])
res = {}
for cname, (C, cov) in cg.CONTACT.items():
    m = cov[:, None] & cov[None, :] & ~cg.eye & np.isfinite(MAP["wt"]) & np.isfinite(MAP["unc31"])
    out = {"n_pairs_both": int(m.sum())}
    for s in ("wt", "unc31"):
        T = MAP[s]; real = cg.pspear(T[m], C[m], [-cg.D[m], cg.A[m]])
        null = []
        for _ in range(200):
            Cn = cs.rewire(C, cov); mm = m & np.isfinite(Cn); null.append(cg.pspear(T[mm], Cn[mm], [-cg.D[mm], cg.A[mm]]))
        out[s] = {"T~C|D,A": real, "degnull_p95": float(np.percentile(null, 95)), "p": float((np.sum(np.array(null) >= real) + 1) / 201),
                  "T~-D|C,A": cg.pspear(T[m], -cg.D[m], [C[m], cg.A[m]]), "T~A|D": cg.pspear(T[m], cg.A[m], [-cg.D[m]])}
    idx = np.where(m.ravel())[0]; Tw, Tu, Cc, Dd, Aa = (X.ravel() for X in (MAP["wt"], MAP["unc31"], C, -cg.D, cg.A))
    bs = []
    for _ in range(500):
        b = rng.choice(idx, len(idx))
        bs.append(cg.pspear(Tw[b], Cc[b], [Dd[b], Aa[b]]) - cg.pspear(Tu[b], Cc[b], [Dd[b], Aa[b]]))
    out["wt_minus_unc31"] = float(out["wt"]["T~C|D,A"] - out["unc31"]["T~C|D,A"]); out["diff_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res[cname] = out
    print(cname, json.dumps(out, default=lambda x: round(x, 3)), flush=True)
json.dump(res, open(cg.OUT / "contact_unc31.json", "w"), indent=1)
