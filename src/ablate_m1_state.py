"""Eval-only ablation: is M1's advantage over Creamer the state-dependence of its prediction?
M1 v2 checkpoints, zero-shot on our test events, scored on the authors' estimator (results/phase0/targets_test.npz).
  real    burn-in on the animal's real pre-stimulus activity (as in all reported numbers)
  blind   burn-in with every observation masked (m = 0): the core starts from its learned rest dynamics, so the
          prediction is an impulse response from rest, as in Creamer's STAM
    python ablate_m1_state.py -> results/m1_state_ablation.json
"""
import json

import numpy as np
import torch

import phase0_ceiling as p0
import train_worm as tw
import worm_data as wd
import worm_models as wm

wm.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cell = wd.model_cell_ids()
T = dict(np.load(p0.OUT / "targets_test.npz"))
names = wd.recording_names(); _, te_n, _ = tw.get_split(names)
import sys; sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402
c = lambda a, b: float(met.nan_corr(a, b)[0])

_crop = wm.crop_batch
def blind_crop(*a, **k):
    x, m, u, keys = _crop(*a, **k)
    m[:, : wm.PRE] = 0
    x[:, : wm.PRE] = 0
    return x, m, u, keys

out = {}
for level in ("neuron", "class"):
    for s in range(5):
        ck = torch.load(f"../results/worm_v2/M1_{level}_s{s}.pt", map_location=wm.DEVICE, weights_only=False)
        model = wm.WormNet(ck["n_keys"]).to(wm.DEVICE); model.load_state_dict(ck["state"])
        te = tw.load_recs(te_n); wm.assign_keys(te, "M1", level, ck["vocab"], np.random.default_rng(s))
        ev = wm.event_list(te)
        row = {}
        for mode in ("real", "blind"):
            wm.crop_batch = blind_crop if mode == "blind" else _crop
            S = wm.pooled_stam(te, ev, wm.predict_events(model, te, ev), cell)
            row[mode] = {"T_auth_ours": c(S, T["T_auth_ours"])}
        wm.crop_batch = _crop
        out.setdefault(level, []).append(row); print(level, s, row, flush=True)
summ = {lvl: {mode: [float(np.mean([r[mode]["T_auth_ours"] for r in v])), float(np.std([r[mode]["T_auth_ours"] for r in v], ddof=1))]
              for mode in ("real", "blind")} for lvl, v in out.items()}
json.dump({"per_seed": out, "mean_sd": summ}, open(p0.OUT.parent / "m1_state_ablation.json", "w"), indent=1)
print(json.dumps(summ))
