"""L2 recalibration: fit a shrinkage factor alpha for the background forecast on TRAINING animals' pseudo events; re-apply the gate on test.
    python background_recal.py -> results/background_recal.json
"""
import json

import numpy as np

import background_layer as bl   # reuses the background model, helpers, test pseudo events (module runs its main block on import)
import train_worm as tw
import worm_data as wd
import worm_models as wm

tr_n, _, _ = tw.get_split(wd.recording_names())
rng = np.random.default_rng(3)


def pseudo_items(recs, step=15):
    out = []
    for ri, r in enumerate(recs):
        st = np.where(r.u.any(1))[0]
        for t in range(wm.PRE + 40, r.x.shape[0] - wm.POST, step):
            if not ((st >= t - wm.PRE - 40) & (st < t + wm.POST)).any():
                out.append((ri, t, 0))
    return out


tr = tw.load_recs(tr_n); wm.assign_keys(tr, "M1", "class", bl.ck["vocab"], np.random.default_rng(0))
saved = bl.recs; bl.recs = tr
it = pseudo_items(tr); M = [bl.measured(tr[ri], t) for ri, t, _ in it]; B = bl.background(it, max(r.x.shape[1] for r in tr))
bl.recs = saved
Mv, Bv = np.concatenate(M), np.concatenate(B); ok = np.isfinite(Mv) & np.isfinite(Bv)
alpha = float((Mv[ok] * Bv[ok]).sum() / (Bv[ok] ** 2).sum())
out = {"n_train_pseudo": len(it), "alpha_fit_on_train": alpha, "train_corr_b_measured": float(np.corrcoef(Mv[ok], Bv[ok])[0, 1])}
# test pseudo events (from the background_layer module)
mp = [bl.measured(bl.recs[ri], t) for ri, t, _ in bl.pseudo]; bp = bl.background(bl.pseudo, bl.max_n)
Mt, Bt = np.concatenate(mp), np.concatenate(bp); ok = np.isfinite(Mt) & np.isfinite(Bt)
out["test_corr_b_measured"] = float(np.corrcoef(Mt[ok], Bt[ok])[0, 1])
out["test_var_reduction_alpha"] = float(1 - np.var(Mt[ok] - alpha * Bt[ok]) / np.var(Mt[ok]))
Spres = bl.pool(bl.pseudo, [m - alpha * b for m, b in zip(mp, bp)])
out["test_pseudo_residual_map_vs_causal"] = bl.c(Spres, bl.TA["T_auth_ours"])
out["gate_pass"] = bool(out["test_var_reduction_alpha"] >= 0.10 and abs(out["test_pseudo_residual_map_vs_causal"]) < 0.03)
print(json.dumps(out, indent=1)); json.dump(out, open("../results/background_recal.json", "w"), indent=1)
