"""State-free (blind burn-in) evaluation of the final models: the causal headline, comparable to Creamer's impulse STAM.
    python eval_blind_final.py -> results/blind_final.json
"""
import glob, json, sys

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
masks = {"all": ~nf.eye, "near": nf.near, "far": nf.far}
_, te_n, _ = tw.get_split(wd.recording_names())
_crop = wm.crop_batch


def blind(*a, **k):
    x, m, u, keys = _crop(*a, **k); m[:, :wm.PRE] = 0; x[:, :wm.PRE] = 0
    return x, m, u, keys


out = {}; Z = {}
import os
SETS = {"final_rl": (("A", "../results/worm_final/M1_class_s[0-9]_FA_s[0-9].pt", "M1", "class", 128, "gru"),
                     ("B", "../results/worm_final/M1_neuron_s[0-9]_FB_s[0-9].pt", "M1", "neuron", 64, "gru"),
                     ("shuffled A", "../results/worm_final_shuf/M2_class_s*_FAshuf_s*.pt", "M2", "class", 128, "gru")),
        "causal": (("A", "../results/worm_causal_final/M1_class_s[0-9]_CF_C_s[0-9].pt", "M1", "class", 128, "gru"),
                   ("B", "../results/worm_causal_final/M1_class_s[0-9]_CF_L_s[0-9].pt", "M1", "class", 64, "lineardiag"))}
WHICH = os.environ.get("WHICH", "final_rl")
for name, pat, scheme, lvl, h, core in SETS[WHICH]:
    for mode in ("state-conditioned", "state-free"):
        wm.crop_batch = blind if mode == "state-free" else _crop
        rows, Ss = [], []
        for p in sorted(glob.glob(pat)):
            ck = torch.load(p, map_location=wm.DEVICE, weights_only=False)
            mdl = wm.WormNet(ck["n_keys"], hidden=h, core=core).to(wm.DEVICE); mdl.load_state_dict(ck["state"]); mdl.openloop = True
            te = tw.load_recs(te_n); wm.assign_keys(te, scheme, lvl, ck["vocab"], np.random.default_rng(7)); ev = wm.event_list(te)
            S = wm.pooled_stam(te, ev, wm.predict_events(mdl, te, ev), nf.cell); Ss.append((S - np.nanmean(S)) / np.nanstd(S))
            rows.append({k: c(S, m) for k, m in masks.items()})
        wm.crop_batch = _crop
        out[f"{name} | {mode}"] = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in masks}
        Z[(name, mode)] = np.nanmean(Ss, 0)
        print(f"{name:11s} {mode:18s} " + "  ".join(f"{k} {v[0]:.3f}±{v[1]:.3f}" for k, v in out[f'{name} | {mode}'].items()), flush=True)
for mode in ("state-conditioned", "state-free"):
    E = np.nanmean([Z[("A", mode)], Z[("B", mode)]], 0); out[f"A+B ensemble | {mode}"] = {k: c(E, m) for k, m in masks.items()}
    print(f"A+B ensemble {mode}: " + "  ".join(f"{k} {v:.3f}" for k, v in out[f'A+B ensemble | {mode}'].items()), flush=True)
json.dump(out, open("../results/blind_final.json" if WHICH == "final_rl" else f"../results/blind_{WHICH}.json", "w"), indent=1)
