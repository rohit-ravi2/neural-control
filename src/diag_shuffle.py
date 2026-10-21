"""What does the shuffled-identity final model predict? (diagnostic; no claims)
    python diag_shuffle.py -> results/diag_shuffle.json
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
T = nf.TA["T_auth_ours"]; offd = ~nf.eye
c = lambda a, b: float(met.nan_corr(np.where(offd, a, np.nan), np.where(offd, b, np.nan))[0])
_, te_n, _ = tw.get_split(wd.recording_names())
_crop = wm.crop_batch


def stam(p, scheme, blind=False, rng_seed=7):
    def f(*a, **k):
        x, m, u, keys = _crop(*a, **k)
        if blind:
            m[:, :wm.PRE] = 0; x[:, :wm.PRE] = 0
        return x, m, u, keys
    wm.crop_batch = f
    ck = torch.load(p, map_location=wm.DEVICE, weights_only=False)
    mdl = wm.WormNet(ck["n_keys"], hidden=128, core="gru").to(wm.DEVICE); mdl.load_state_dict(ck["state"]); mdl.openloop = True
    te = tw.load_recs(te_n); wm.assign_keys(te, scheme, "class", ck["vocab"], np.random.default_rng(rng_seed)); ev = wm.event_list(te)
    S = wm.pooled_stam(te, ev, wm.predict_events(mdl, te, ev), nf.cell); wm.crop_batch = _crop
    return S


ps = sorted(glob.glob("../results/worm_final_shuf/M2_class_s*_FAshuf_s*.pt"))[:1]
pa = sorted(glob.glob("../results/worm_final/M1_class_s0_FA_s0.pt"))[:1]
Ssh, Ssh_blind, Ssh_rng2 = stam(ps[0], "M2"), stam(ps[0], "M2", blind=True), stam(ps[0], "M2", rng_seed=99)
SA = stam(pa[0], "M1")
add = lambda S: np.nanmean(S, 1)[:, None] + np.nanmean(S, 0)[None, :] - np.nanmean(S)
out = {"shuffled vs target": c(Ssh, T), "shuffled blind burn-in vs target": c(Ssh_blind, T),
       "shuffled (different test permutation) vs target": c(Ssh_rng2, T), "shuffled perm1 vs perm2": c(Ssh, Ssh_rng2),
       "shuffled vs its own additive (row+col)": c(Ssh, add(Ssh)), "shuffled additive vs target": c(add(Ssh), T),
       "real A vs target": c(SA, T), "real A vs shuffled": c(SA, Ssh), "real A additive vs target": c(add(SA), T),
       "target own additive vs target": c(add(T), T)}
for k, v in out.items():
    print(f"{k:50s} {v:.3f}")
json.dump(out, open("../results/diag_shuffle.json", "w"), indent=1)
