"""Strict leakage checks for OBS-Randi: (a) frames BEFORE the first stimulus of each recording (stimulus-naive), pooled over
animals; (b) exclusion windows of 45 and 60 s. O1 = zero-lag correlation (per-animal, pooled by name). Fit animals: train split
(scored on test animals) and, for (a) only, all 110 animals (scored on the all-animal STAM; no stimulation data used anyway).
    python obs_randi_strict.py -> results/obs_randi_strict.json
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
names = wd.recording_names(); tr_n, te_n, _ = tw.get_split(names)


def pooled_corr(nms, mask_fn, min_frames=20):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ems, ins, _ = wd.load_aligned(nms, cell)
    acc, cnt, nfr, nan_ = np.zeros((n, n)), np.zeros((n, n)), 0, 0
    for e, u in zip(ems, ins):
        ok = mask_fn(u)
        if ok.sum() < min_frames:
            continue
        obs = np.where(np.isfinite(e[ok]).mean(0) > 0.9)[0]
        X = e[ok][:, obs]
        X = np.nan_to_num((X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-6))
        acc[np.ix_(obs, obs)] += X.T @ X / len(X); cnt[np.ix_(obs, obs)] += 1; nfr += ok.sum(); nan_ += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan), int(nfr), nan_


def pre_first(u):
    st = np.where(u == 1)[0]; T = u.shape[0]
    return np.arange(T) < (st.min() - 4 if len(st) else T)


def after(w):
    def f(u):
        st = np.unique(np.where(u == 1)[0]); T = u.shape[0]
        last = np.full(T, -10**9); nxt = np.full(T, 10**9)
        for s in st:
            last[s:] = np.maximum(last[s:], s); nxt[:s + 1] = np.minimum(nxt[:s + 1], s)
        return (np.arange(T) - last >= 2 * w) & (nxt - np.arange(T) >= 4)
    return f


if __name__ == "__main__":
    out = {}
    S, fr, na = pooled_corr(tr_n, pre_first); out["prefirst_train->test"] = {"r": c(S, TA["T_auth_ours"]), "frames": fr, "animals": na}
    S, fr, na = pooled_corr(names, pre_first); out["prefirst_all->all"] = {"r": c(S, TA["T_auth_all"]), "frames": fr, "animals": na}
    for w in (45, 60):
        S, fr, na = pooled_corr(tr_n, after(w), min_frames=10); out[f"w{w}_train->test"] = {"r": c(S, TA["T_auth_ours"]), "frames": fr, "animals": na}
    for k, v in out.items():
        print(k, v, flush=True)
    json.dump(out, open("../results/obs_randi_strict.json", "w"), indent=1)
