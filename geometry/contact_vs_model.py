"""Does nerve-ring contact explain what the best causal model misses? (exploratory)
Model: state-free identity model C, 5-fold CV over all 110 animals (results/worm_causal_final/*CFCV*_acc.npz, 2 seeds; 0.451 vs
T_auth_all). Statistic: partial Spearman rho(T_auth_all, contact | model prediction, soma distance, synapses), all pairs and
near / far, against the degree-preserving contact null of contact_stress.py. Also: how much does contact add to the model when
combined linearly (in-sample rank regression; descriptive only)?
    python geometry/contact_vs_model.py -> geometry/results/contact_vs_model.json
"""
import glob, json, pathlib, re, sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import contact_geometry as cg  # noqa: E402
import contact_stress as cs  # noqa: E402  (rewire)

ROOT = cg.ROOT
parts = {}
for p in glob.glob(str(ROOT / "results" / "worm_causal_final" / "*_CFCV_f*_s*_acc.npz")):
    m = re.search(r"_CFCV_f(\d)_s(\d)_acc\.npz", p); parts.setdefault(int(m.group(2)), []).append(np.load(p))
S = []
for s, ps in sorted(parts.items()):
    assert len(ps) == 5, (s, len(ps))
    acc = sum(z["acc"] for z in ps); cnt = sum(z["cnt"] for z in ps)
    X = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan); S.append((X - np.nanmean(X)) / np.nanstd(X))
P = np.nanmean(S, 0)
T = cg.TARGETS["T_auth_all"]; D, A, eye = cg.D, cg.A, cg.eye
res = {"model_seeds": len(S)}
for cname, (C, cov) in cg.CONTACT.items():
    base = cov[:, None] & cov[None, :] & ~eye & np.isfinite(T) & np.isfinite(P)
    out = {}
    for zone, m in (("all", base), ("near<10", base & (D < 10)), ("far>=15", base & (D >= 15))):
        real = cg.pspear(T[m], C[m], [P[m], -D[m], A[m]])
        null = []
        for _ in range(200):
            Cn = cs.rewire(C, cov); mm = m & np.isfinite(Cn)
            null.append(cg.pspear(T[mm], Cn[mm], [P[mm], -D[mm], A[mm]]))
        r_model = float(np.corrcoef(rankdata(T[m]), rankdata(P[m]))[0, 1])
        # descriptive: rank-regression of T on [P] vs [P, C]; correlation of the fit with T
        def fit(Z):
            Zr = np.column_stack([np.ones(m.sum())] + [rankdata(z) for z in Z]); b = np.linalg.lstsq(Zr, rankdata(T[m]), rcond=None)[0]
            return float(np.corrcoef(Zr @ b, rankdata(T[m]))[0, 1])
        out[zone] = {"n": int(m.sum()), "rho_T_model": r_model, "T~C|model,D,A": real, "degnull_p95": float(np.percentile(null, 95)),
                     "p": float((np.sum(np.array(null) >= real) + 1) / 201), "fit_model": fit([P[m]]), "fit_model+C": fit([P[m], C[m]]),
                     "fit_model+D+A": fit([P[m], -D[m], A[m]]), "fit_model+D+A+C": fit([P[m], -D[m], A[m], C[m]])}
    res[cname] = out
    print(cname, json.dumps({z: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()} for z, o in out.items()}), flush=True)
json.dump(res, open(cg.OUT / "contact_vs_model.json", "w"), indent=1)
