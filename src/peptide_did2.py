"""PEPTIDE-DiD second operationalizations: (i) H1 at full wt depth; (ii) potency-filtered P_strong.
    python peptide_did2.py -> results/peptide_did2.json
"""
import json, re

import numpy as np
import pandas as pd

import peptide_did as pd0   # reuses atlas, A, ALL, FIN, resid, pspear, matched_y, stats machinery (runs its main? no: guarded)

na, N = pd0.na, pd0.N
D = pd.read_csv(pd0.wa.__path__[0] + "/data/deorphanization_media_6.csv")
pep_gene = lambda s: "-".join(str(s).strip().upper().split("-")[:2])     # NLP-13-1 -> NLP-13 ; FLP-21 -> FLP-21
seq_id = lambda s: str(s).strip().upper().rsplit("-", 1)[0] if re.search(r"-\d+$", str(s).strip()) else str(s).strip().upper()
best = {}
for _, r in D.iterrows():
    k = (pep_gene(r["Peptide"]), seq_id(r["GPCR ID"]))
    try:
        ec = float(r["EC50 (M)"])
    except (TypeError, ValueError):
        continue
    best[k] = min(best.get(k, np.inf), ec)
_, combos = na.get_peptidergic_connectome(return_combos=True)
Pstrong = np.zeros((N, N)); n_unmatched = 0
for i in range(N):
    for j in range(N):
        for pep, gpcr in (combos[i, j] or []):
            ec = best.get((pep_gene(pep), seq_id(gpcr)))
            if ec is None:
                n_unmatched += 1
            elif ec <= 1e-8:
                Pstrong[i, j] += 1
Pstrong = np.log1p(Pstrong)
print("P_strong density", float((Pstrong > 0).mean()), "vs P", float((pd0.P > 0).mean()), "; combos without EC50 match:", n_unmatched, flush=True)

rng = np.random.default_rng(1)
FIN, ALL = pd0.FIN, pd0.ALL


def full_wt(Pm, perms=200):
    pr = np.argwhere((FIN["wt"] >= 3) & ~np.eye(N, dtype=bool)); r, c = pr[:, 0], pr[:, 1]
    y = np.array([abs(np.nanmean(np.asarray(ALL["wt"][i, j], float).ravel())) for i, j in pr])
    yr, ar = pd0.resid(y, r, c), pd0.resid(pd0.A[r, c], r, c)
    obs = pd0.pspear(yr, pd0.resid(Pm[r, c], r, c), ar)
    a_obs = pd0.pspear(yr, ar, pd0.resid(Pm[r, c], r, c))
    null = []
    for _ in range(perms):
        pm = rng.permutation(N); Pp = Pm[np.ix_(pm, pm)]
        null.append(pd0.pspear(yr, pd0.resid(Pp[r, c], r, c), ar))
    return {"n_pairs": int(len(pr)), "rho_P|A": obs, "rho_A|P": a_obs, "null_p95": float(np.percentile(null, 95)),
            "p_perm": float(np.mean(np.array(null) >= obs))}


if __name__ == "__main__":
    out = {"full_wt_P": full_wt(pd0.P), "full_wt_Pstrong": full_wt(Pstrong)}
    print(out, flush=True)
    point = pd0.stats(rng, np.arange(len(pd0.pairs)), Pstrong)
    boot = [pd0.stats(rng, rng.integers(0, len(pd0.pairs), len(pd0.pairs)), Pstrong) for _ in range(200)]
    null = []
    for _ in range(200):
        pm = rng.permutation(N); null.append(pd0.stats(rng, np.arange(len(pd0.pairs)), Pstrong[np.ix_(pm, pm)])["DiD"])
    out["matched_Pstrong"] = {"point": point, "DiD_ci95": [float(np.percentile([b["DiD"] for b in boot], q)) for q in (2.5, 97.5)],
                              "wtPA_ci95": [float(np.percentile([b["wt"]["P|A"] for b in boot], q)) for q in (2.5, 97.5)],
                              "null_DiD_p95": float(np.percentile(null, 95)), "p_perm_DiD": float(np.mean(np.array(null) >= point["DiD"]))}
    print(json.dumps(out["matched_Pstrong"], indent=1))
    json.dump(out, open("../results/peptide_did2.json", "w"), indent=1)
