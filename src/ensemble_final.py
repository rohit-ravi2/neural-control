"""Final headline: seed ensembles of configs A (GRU class, resp-loss 0.03, h128) and B (GRU neuron, 0.03, h64) on the fixed
split, authors' estimator; plus near/far split (near_far masks) for each.
    python ensemble_final.py -> results/ensemble_final.json
"""
import glob, json, re, sys

import numpy as np
import torch

import near_far as nf
import train_worm as tw
import worm_data as wd
import worm_models as wm

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402

wm.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T = nf.TA["T_auth_ours"]; c = lambda a, m: float(met.nan_corr(np.where(m, a, np.nan), np.where(m, T, np.nan))[0])
_, te_n, _ = tw.get_split(wd.recording_names())
CFG = {"A": ("class", 128), "B": ("neuron", 64)}
S = {}
for k, (lvl, h) in CFG.items():
    S[k] = []
    for p in sorted(glob.glob(f"../results/worm_final/M1_{lvl}_s[0-9]_F{k}_s[0-9].pt")):
        ck = torch.load(p, map_location=wm.DEVICE, weights_only=False)
        m = wm.WormNet(ck["n_keys"], hidden=h, core="gru").to(wm.DEVICE); m.load_state_dict(ck["state"]); m.openloop = True
        te = tw.load_recs(te_n); wm.assign_keys(te, "M1", lvl, ck["vocab"], np.random.default_rng(0)); ev = wm.event_list(te)
        M = wm.pooled_stam(te, ev, wm.predict_events(m, te, ev), nf.cell); S[k].append((M - np.nanmean(M)) / np.nanstd(M))
out = {}
masks = {"all": ~nf.eye, "near": nf.near, "far": nf.far}
for k, v in S.items():
    E = np.nanmean(v, 0); out[f"{k}_ensemble"] = {mk: c(E, mm) for mk, mm in masks.items()}; out[f"{k}_n"] = len(v)
AB = np.nanmean([np.nanmean(S["A"], 0), np.nanmean(S["B"], 0)], 0); np.save("../results/ensemble_final_AB.npy", AB); out["A+B_ensemble"] = {mk: c(AB, mm) for mk, mm in masks.items()}
out["reference_rho"] = {mk: c(nf.TA["T_auth_train"], mm) for mk, mm in masks.items()}
print(json.dumps(out, indent=1)); json.dump(out, open("../results/ensemble_final.json", "w"), indent=1)
