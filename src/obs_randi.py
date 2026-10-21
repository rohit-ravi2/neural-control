"""OBS-Randi: observational predictors from Randi's own inter-stimulus periods (immobilised; same preparation).
    python obs_randi.py -> results/obs_randi.json
"""
import json, sys, warnings

import numpy as np

import phase0_ceiling as p0
import train_worm as tw
import worm_data as wd

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402

cell = wd.model_cell_ids(); n = len(cell); eye = np.eye(n, dtype=bool)
TA = np.load(p0.OUT / "targets_test.npz")
c = lambda a, b: float(met.nan_corr(np.where(eye, np.nan, a), np.where(eye, np.nan, b))[0])
tr_n, te_n, _ = tw.get_split(wd.recording_names())
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ems, ins, _ = wd.load_aligned(tr_n, cell)
out = {}
for w in (10, 20, 30):
    acc0, accR, cnt = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    nfr = 0
    for e, u in zip(ems, ins):
        st = np.unique(np.where(u == 1)[0]); T = e.shape[0]
        last = np.full(T, -10**9); nxt = np.full(T, 10**9)
        for s in st:
            last[s:] = np.maximum(last[s:], s); nxt[:s + 1] = np.minimum(nxt[:s + 1], s)
        ok = (np.arange(T) - last >= 2 * w) & (nxt - np.arange(T) >= 4)
        obs = np.where(np.isfinite(e[ok]).mean(0) > 0.9)[0]
        if ok.sum() < 50 or len(obs) < 5:
            continue
        X = e[:, obs].copy()
        mu, sd = np.nanmean(X[ok], 0), np.nanstd(X[ok], 0) + 1e-6
        X = np.nan_to_num((X - mu) / sd)
        Xo = X[ok]; C0 = Xo.T @ Xo / len(Xo)
        pair = ok[:-1] & ok[1:]
        C1 = X[1:][pair].T @ X[:-1][pair] / max(pair.sum(), 1)
        A = C1 @ np.linalg.inv(C0 + 0.1 * np.eye(len(obs)))
        R, P = np.zeros_like(A), np.eye(len(obs))
        for _ in range(60):
            P = A @ P; R += P
        acc0[np.ix_(obs, obs)] += C0; accR[np.ix_(obs, obs)] += R; cnt[np.ix_(obs, obs)] += 1
        nfr += ok.sum()
    S0, SR = (np.where(cnt > 0, M / np.maximum(cnt, 1), np.nan) for M in (acc0, accR))
    out[f"w{w}"] = {"frames": int(nfr), "O1_corr0": c(S0, TA["T_auth_ours"]), "O3_var1": c(SR, TA["T_auth_ours"])}
    print(w, out[f"w{w}"], flush=True)
json.dump(out, open("../results/obs_randi.json", "w"), indent=1)
