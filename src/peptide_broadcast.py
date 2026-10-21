"""PEPTIDE broadcast variant: does a neuron's peptidergic out-degree predict how broadly its
stimulation propagates, beyond anatomical out-degree, and is that lost in unc-31?
    python peptide_broadcast.py -> results/peptide_broadcast.json
"""
import json

import numpy as np
from scipy.stats import rankdata

import peptide_did as pd0
import peptide_did2 as pd2

na, N = pd0.na, pd0.N
Q = {s: na.get_signal_propagation_q(strain=s) for s in ("wt", "unc31")}
anat = ((na.get_chemical_synapses() > 0) | (na.get_gap_junctions() > 0)).sum(0)      # partners of j (column = source)
pdeg = {"P": (pd0.P > 0).sum(0), "Pstrong": (pd2.Pstrong > 0).sum(0)}
FIN, ALL = pd0.FIN, pd0.ALL


def per_neuron(s):
    br, amp, n = np.full(N, np.nan), np.full(N, np.nan), np.zeros(N, int)
    for j in range(N):
        resp = [i for i in range(N) if i != j and FIN[s][i, j] >= 1]
        n[j] = sum(FIN[s][i, j] for i in resp)
        if len(resp) >= 5:
            q = Q[s][resp, j]; br[j] = np.mean(q[np.isfinite(q)] < 0.05) if np.isfinite(q).any() else np.nan
            amp[j] = np.mean([abs(np.nanmean(np.asarray(ALL[s][i, j], float).ravel())) for i in resp])
    return br, amp, n


def pspear(y, x, z):
    ry, rx, rz = (rankdata(a) for a in (y, x, z))
    def res(a, b):
        b1 = np.c_[np.ones_like(b, dtype=float), b]; return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    return float(np.corrcoef(res(ry, rz), res(rx, rz))[0, 1])


W, U = per_neuron("wt"), per_neuron("unc31")
keep = np.where((W[2] >= 10) & (U[2] >= 10) & np.isfinite(W[0]) & np.isfinite(U[0]))[0]
print("neurons stimulated in both genotypes (>=10 responder occurrences each):", len(keep), flush=True)
rng = np.random.default_rng(0); out = {"n_neurons": int(len(keep))}
for yname, k in (("breadth", 0), ("amplitude", 1)):
    for pname, pdg in pdeg.items():
        def st(idx):
            o = {g: pspear(Y[k][keep][idx], pdg[keep][idx], anat[keep][idx]) for g, Y in (("wt", W), ("unc31", U))}
            a = {g: pspear(Y[k][keep][idx], anat[keep][idx], pdg[keep][idx]) for g, Y in (("wt", W), ("unc31", U))}
            return o, a, (o["wt"] - o["unc31"]) - (a["wt"] - a["unc31"])
        o, a, did = st(np.arange(len(keep)))
        boots = [st(rng.integers(0, len(keep), len(keep))) for _ in range(1000)]
        ci = lambda f: [float(np.nanpercentile([f(b) for b in boots], q)) for q in (2.5, 97.5)]
        out[f"{yname}_{pname}"] = {"pep|anat": o, "anat|pep": a, "DiD": did, "ci_wt_pep": ci(lambda b: b[0]["wt"]), "ci_DiD": ci(lambda b: b[2])}
        print(yname, pname, json.dumps(out[f"{yname}_{pname}"]), flush=True)
json.dump(out, open("../results/peptide_broadcast.json", "w"), indent=1)
