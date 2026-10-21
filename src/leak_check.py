"""Leak check: mask the last K burn-in frames (all neurons) at evaluation for config A and the shuffled model.
    python leak_check.py -> results/leak_check.json
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
T = nf.TA["T_auth_ours"]; c = lambda a: float(met.nan_corr(np.where(nf.eye, np.nan, a), np.where(nf.eye, np.nan, T))[0])
_, te_n, _ = tw.get_split(wd.recording_names())
_crop = wm.crop_batch
out = {}
for name, pat, scheme, h in (("real A", "../results/worm_final/M1_class_s[0-2]_FA_s[0-2].pt", "M1", 128),
                             ("shuffled A", "../results/worm_final_shuf/M2_class_s*_FAshuf_s*.pt", "M2", 128)):
    for K in (0, 1, 2, 3, 5):
        def masked(*a, K=K, **k):
            x, m, u, keys = _crop(*a, **k)
            if K:
                m[:, wm.PRE - K:wm.PRE] = 0; x[:, wm.PRE - K:wm.PRE] = 0
            return x, m, u, keys
        wm.crop_batch = masked
        rs = []
        for p in sorted(glob.glob(pat)):
            ck = torch.load(p, map_location=wm.DEVICE, weights_only=False)
            mdl = wm.WormNet(ck["n_keys"], hidden=h, core="gru").to(wm.DEVICE); mdl.load_state_dict(ck["state"]); mdl.openloop = True
            te = tw.load_recs(te_n); wm.assign_keys(te, scheme, "class", ck["vocab"], np.random.default_rng(7)); ev = wm.event_list(te)
            rs.append(c(wm.pooled_stam(te, ev, wm.predict_events(mdl, te, ev), nf.cell)))
        wm.crop_batch = _crop
        out[f"{name} K={K}"] = [float(np.mean(rs)), float(np.std(rs))]
        print(f"{name:12s} mask last {K} burn-in frames: {np.mean(rs):.3f} ± {np.std(rs):.3f}", flush=True)
json.dump(out, open("../results/leak_check.json", "w"), indent=1)
