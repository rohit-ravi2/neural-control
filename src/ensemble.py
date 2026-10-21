"""Ensemble of saved zero-shot models (eval-only): average of per-model pooled STAMs (each z-scored over finite entries).
Scored on the authors' estimator (test animals); ceiling 0.44 (results/headroom.json).
    python ensemble.py -> results/ensemble.json
"""
import glob, json, re, sys

import numpy as np
import torch

import phase0_ceiling as p0
import train_worm as tw
import worm_data as wd
import worm_models as wm

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402

wm.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cell = wd.model_cell_ids(); eye = np.eye(len(cell), dtype=bool)
T = np.load(p0.OUT / "targets_test.npz")["T_auth_ours"]
c = lambda a: float(met.nan_corr(np.where(eye, np.nan, a), np.where(eye, np.nan, T))[0])
_, te_n, _ = tw.get_split(wd.recording_names())
R = "../results"
FAM = {"M1n": [(f"{R}/worm_v2/M1_neuron_s{s}.pt", "neuron", "gru", 64, False) for s in range(5)],
       "M1c": [(f"{R}/worm_v2/M1_class_s{s}.pt", "class", "gru", 64, False) for s in range(5)],
       "GRUol": [(f"{R}/worm_block2/M1_neuron_s{s}_GRUol_s{s}.pt", "neuron", "gru", 64, True) for s in range(5)],
       "LINol": [(f"{R}/worm_block2/M1_neuron_s{s}_LINdiagol_s{s}.pt", "neuron", "lineardiag", 64, True) for s in range(5)],
       "K32": [(p, "neuron", "lineardiag", 32, True) for p in sorted(glob.glob(f"{R}/worm_modes/*_K32_s*.pt"))]}
S = {}
for fam, items in FAM.items():
    S[fam] = []
    for path, lvl, core, h, ol in items:
        ck = torch.load(path, map_location=wm.DEVICE, weights_only=False)
        m = wm.WormNet(ck["n_keys"], hidden=h, core=core).to(wm.DEVICE); m.load_state_dict(ck["state"]); m.openloop = ol
        te = tw.load_recs(te_n); wm.assign_keys(te, "M1", lvl, ck["vocab"], np.random.default_rng(0)); ev = wm.event_list(te)
        M = wm.pooled_stam(te, ev, wm.predict_events(m, te, ev), cell)
        M = (M - np.nanmean(M)) / np.nanstd(M); S[fam].append(M)
    print(fam, "singles", [round(c(M), 3) for M in S[fam]], "| seed-ensemble", round(c(np.nanmean(S[fam], 0)), 3), flush=True)
out = {f: {"singles": [c(M) for M in v], "seed_ensemble": c(np.nanmean(v, 0))} for f, v in S.items()}
fams = list(S)
allm = np.nanmean([np.nanmean(S[f], 0) for f in fams], 0); out["all_families"] = c(allm)
out["M1c+LINol"] = c(np.nanmean([np.nanmean(S["M1c"], 0), np.nanmean(S["LINol"], 0)], 0))
print("all families:", round(out["all_families"], 3), " M1c+LINol:", round(out["M1c+LINol"], 3))
json.dump(out, open(f"{R}/ensemble.json", "w"), indent=1)
