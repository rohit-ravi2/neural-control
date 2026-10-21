"""Why does the class-level shuffled-identity model (FAshuf) score 0.21? Test: exclude (responder, target) entries of the SAME class
(bilateral / D-V partners keep a shared key under per-animal class permutation). Also report the real config A on the same masks.
    python eval_shuffle_sameclass.py -> results/shuffle_sameclass.json
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
T = nf.TA["T_auth_ours"]; cell = nf.cell
c = lambda a, m: float(met.nan_corr(np.where(m, a, np.nan), np.where(m, T, np.nan))[0])
cls = np.array([wm.neuron_class(x) for x in cell]); same = (cls[:, None] == cls[None, :]) & ~nf.eye
masks = {"all": ~nf.eye, "excl_same_class": ~nf.eye & ~same, "same_class_only": same, "far_excl_same": nf.far & ~same}
_, te_n, _ = tw.get_split(wd.recording_names())
out = {}
for name, pat, scheme in (("shuffled (FAshuf)", "../results/worm_final_shuf/M2_class_s*_FAshuf_s*.pt", "M2"),
                          ("real A", "../results/worm_final/M1_class_s[0-2]_FA_s[0-2].pt", "M1")):
    rows = []
    for p in sorted(glob.glob(pat)):
        ck = torch.load(p, map_location=wm.DEVICE, weights_only=False)
        m = wm.WormNet(ck["n_keys"], hidden=128, core="gru").to(wm.DEVICE); m.load_state_dict(ck["state"]); m.openloop = True
        te = tw.load_recs(te_n); wm.assign_keys(te, scheme, "class", ck["vocab"], np.random.default_rng(7)); ev = wm.event_list(te)
        S = wm.pooled_stam(te, ev, wm.predict_events(m, te, ev), cell); rows.append({k: c(S, mm) for k, mm in masks.items()})
    out[name] = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in masks}
    print(name, {k: f"{v[0]:.3f}±{v[1]:.3f}" for k, v in out[name].items()}, flush=True)
out["n_same_class_entries"] = int((same & np.isfinite(T)).sum())
json.dump(out, open("../results/shuffle_sameclass.json", "w"), indent=1)
