"""OBS->CAUSAL baselines O1-O3: predict Randi optogenetic STAMs from Atanas spontaneous activity.
    python obs_baselines.py -> results/obs_baselines.json  (also caches T_auth_all into targets_test.npz)
"""
import json, sys, warnings

import numpy as np

import atanas_data as ad
import phase0_ceiling as p0
import worm_data as wd

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402

cell = wd.model_cell_ids(); n = len(cell); col = {c: i for i, c in enumerate(cell)}; eye = np.eye(n, dtype=bool)
f = p0.OUT / "targets_test.npz"; TA = dict(np.load(f))
if "T_auth_all" not in TA:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TA["T_auth_all"] = p0.authors_stams(wd.recording_names(), cell)
    np.savez(f, **TA)
c = lambda a, b: float(met.nan_corr(np.where(eye, np.nan, a), np.where(eye, np.nan, b))[0])
L = 60
acc = {k: np.zeros((n, n)) for k in ("O1_corr0", "O2_lagcorr", "O3_var1")}
cnt = np.zeros((n, n))
for name, x, ids in ad.load():
    idx = [(i, col[s]) for i, s in enumerate(ids) if s in col]
    loc, glob_ = np.array([a for a, _ in idx]), np.array([b for _, b in idx])
    X = x[:, loc]; T, m = X.shape
    C0 = X.T @ X / T
    lag = np.zeros((m, m))
    for t in range(1, L + 1):
        lag += X[t:].T @ X[:-t] / (T - t)          # [i, j] = E[x_i(t+τ) x_j(t)]
    C1 = X[1:].T @ X[:-1] / (T - 1)
    A = C1 @ np.linalg.inv(C0 + 0.1 * np.eye(m))
    R, P = np.zeros((m, m)), np.eye(m)
    for t in range(L):
        P = A @ P; R += P                           # response of i to unit impulse in j, summed over 60 frames
    for k, M in (("O1_corr0", C0), ("O2_lagcorr", lag), ("O3_var1", R)):
        acc[k][np.ix_(glob_, glob_)] += M
    cnt[np.ix_(glob_, glob_)] += 1
out = {"n_pairs_covered": int((cnt > 0).sum())}
for k, M in acc.items():
    S = np.where(cnt > 0, M / np.maximum(cnt, 1), np.nan)
    out[k] = {t: c(S, TA[t]) for t in ("T_auth_ours", "T_auth_all")}
    print(k, {t: round(v, 3) for t, v in out[k].items()}, flush=True)
out["rho_train_to_test"] = c(TA["T_auth_train"], TA["T_auth_ours"])
json.dump(out, open("../results/obs_baselines.json", "w"), indent=1)
print(out)
