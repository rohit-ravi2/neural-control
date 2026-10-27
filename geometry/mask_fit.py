"""Which graph does propagation run on? Support-masked linear propagation fits (exploratory; a static mini-Creamer).

Model: S = (I - W)^-1 - I  (steady-state response of responder i to a unit drive of target j), W free only on a SUPPORT mask,
fitted to the training animals' authors'-estimator map (T_auth_train, 80 animals) by MSE on finite off-diagonal entries + L2;
scored by correlation with the held-out test animals' map (T_auth_ours, 30 animals) on all / near (< 10 µm) / far (≥ 15 µm)
entries. Cells restricted to the 125 covered by the nerve-ring contact sheet (adult) so every support is defined.
Supports (undirected masks; W itself is directed):
  SYN       chemical or gap synapse in either direction
  CON       nerve-ring membrane contact > 0
  CON@SYN   contact pairs with the largest contact areas, thinned to SYN's density
  NEAR@SYN  the soma-nearest pairs (per-animal median distance), at SYN's density   (geometry-as-confound support)
  RAND@SYN  random pairs at SYN's density (5 draws);  RAND@CON at CON's density (5 draws)
  SYN∪CON   union
    python geometry/mask_fit.py -> geometry/results/mask_fit.json
"""
import json, pathlib, sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import contact_geometry as cg  # noqa: E402

torch.set_num_threads(8)
rng = np.random.default_rng(0)
C, cov = cg.CONTACT["adult"]
keep = np.where(cov)[0]; n = len(keep); eye = np.eye(n, dtype=bool)
TR = cg.nf.TA["T_auth_train"][np.ix_(keep, keep)]; TE = cg.nf.TA["T_auth_ours"][np.ix_(keep, keep)]
D = cg.D[np.ix_(keep, keep)]; Ck = np.nan_to_num(C[np.ix_(keep, keep)]); Ak = cg.A_sym[np.ix_(keep, keep)]
iu = np.triu_indices(n, 1)
SYN = (Ak > 0) & ~eye; CON = (Ck > 0) & ~eye
dens = lambda M: M[iu].mean()


def top_k_sym(score, k):
    v = score[iu]; order = np.argsort(-v)[:k]; M = np.zeros((n, n), bool); M[iu[0][order], iu[1][order]] = True
    return M | M.T


def rand_sym(k):
    order = rng.choice(len(iu[0]), k, replace=False); M = np.zeros((n, n), bool); M[iu[0][order], iu[1][order]] = True
    return M | M.T


k_syn, k_con = int(SYN[iu].sum()), int(CON[iu].sum())
SUP = {"SYN": SYN, "CON": CON, "CON@SYN": top_k_sym(Ck, k_syn), "NEAR@SYN": top_k_sym(-D, k_syn), "SYN|CON": SYN | CON}
for r in range(5):
    SUP[f"RAND@SYN_{r}"] = rand_sym(k_syn); SUP[f"RAND@CON_{r}"] = rand_sym(k_con)
print("densities", {k: round(float(dens(v)), 3) for k, v in SUP.items() if not k.startswith("RAND") or k.endswith("_0")}, flush=True)

Ttr = torch.tensor(np.nan_to_num(TR), dtype=torch.float64); Mtr = torch.tensor(np.isfinite(TR) & ~eye, dtype=torch.float64)
scale = float(np.nanstd(TR[~eye])); Ttr = Ttr / scale
corr = lambda P, T, m: float(np.corrcoef(P[m & np.isfinite(T)], T[m & np.isfinite(T)])[0, 1])
masks_eval = {"all": ~eye, "near<10": (D < 10) & ~eye, "far>=15": (D >= 15) & ~eye}


def fit(support, l2, steps=1500):
    S_ = torch.tensor(support, dtype=torch.float64); W = torch.zeros((n, n), dtype=torch.float64, requires_grad=True)
    g = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([W, g], lr=0.01); I = torch.eye(n, dtype=torch.float64)
    for _ in range(steps):
        Wm = W * S_; P = torch.exp(g) * (torch.linalg.inv(I - Wm) - I)
        loss = (((P - Ttr) ** 2) * Mtr).sum() / Mtr.sum() + l2 * (Wm ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():          # keep the propagator stable: spectral radius < 0.95
            rad = torch.linalg.eigvals(W * S_).abs().max()
            if rad > 0.95:
                W.mul_(0.95 / rad)
    with torch.no_grad():
        P = (torch.exp(g) * (torch.linalg.inv(I - W * S_) - I)).numpy()
    return P, float(loss)


# choose L2 per support on an inner split of the training map: no -- the training map is one pooled matrix. Instead pick L2
# from a small grid by a held-out-ENTRY split of the training map (random 20% of finite entries), then refit on all entries.
fin = np.argwhere(np.isfinite(TR) & ~eye); val = fin[rng.permutation(len(fin))[: len(fin) // 5]]
Mval = np.zeros((n, n), bool); Mval[val[:, 0], val[:, 1]] = True
res = {"n_cells": n, "k_syn": k_syn, "k_con": k_con}
for name, sup in SUP.items():
    best = None
    for l2 in (1e-4, 1e-3, 1e-2):
        Mtr_full = Mtr.clone(); Mtr.copy_(torch.tensor(np.isfinite(TR) & ~eye & ~Mval, dtype=torch.float64))
        P, _ = fit(sup, l2, steps=800); Mtr.copy_(Mtr_full)
        v = corr(P, TR, Mval)
        if best is None or v > best[1]:
            best = (l2, v)
    P, loss = fit(sup, best[0])
    r = {"l2": best[0], "val_r": best[1], "train_r": corr(P, TR, ~eye), **{f"test_{k}": corr(P, TE, m) for k, m in masks_eval.items()},
         "n_params": int(sup.sum())}
    res[name] = r
    print(f"{name:12s}", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    json.dump(res, open(cg.OUT / "mask_fit.json", "w"), indent=1)
for base in ("RAND@SYN", "RAND@CON"):
    vals = [res[f"{base}_{r}"] for r in range(5)]
    res[base] = {k: [float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals]))] for k in vals[0] if isinstance(vals[0][k], float)}
    print(base, json.dumps({k: [round(a, 3), round(b, 3)] for k, (a, b) in res[base].items()}), flush=True)
reference = {"train_map_as_prediction(rho)": {k: corr(TR, TE, m) for k, m in masks_eval.items()}}
res.update(reference); print(reference)
json.dump(res, open(cg.OUT / "mask_fit.json", "w"), indent=1)
