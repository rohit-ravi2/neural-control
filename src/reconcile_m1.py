"""Score M1 (v2 checkpoints, zero-shot) against the SAME targets as reconcile_creamer.py, so M1 and Creamer's
published models are compared on one estimator and one split.
  T_auth_ours  the authors' estimator on our 30 test animals (clean comparison: M1 never saw these animals;
               Creamer's models may have, if any of them were in the paper's training set -> optimistic for Creamer)
  T_ship       the shipped test STAMs (CONTAMINATED for M1: its 80 training animals include some of the paper's
               test animals; reported for completeness only)
    python reconcile_m1.py  -> results/creamer_reconcile_m1.json, results/phase0/targets_test.npz
"""
import json, pickle, sys, warnings

import numpy as np
import torch

import phase0_ceiling as p0
import train_worm as tw
import worm_data as wd
import worm_models as wm

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402

wm.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cell = wd.model_cell_ids(); n = len(cell); eye = np.eye(n, dtype=bool)
c = lambda a, b: float(met.nan_corr(a, b)[0])
names = wd.recording_names(); tr_n, te_n, _ = tw.get_split(names)
shipped = pickle.load(open(wd.REPO / "data" / "measured_stams.pkl", "rb"))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    T = {"T_ship": shipped["test"], "T_auth_ours": p0.authors_stams(te_n, cell)}
for k in T:
    T[k][eye] = np.nan
np.savez(p0.OUT / "targets_test.npz", **T)

tr = tw.load_recs(tr_n)
out = {}
for level in ("neuron", "class"):
    for s in range(5):
        ck = torch.load(f"../results/worm_v2/M1_{level}_s{s}.pt", map_location=wm.DEVICE, weights_only=False)
        model = wm.WormNet(ck["n_keys"]).to(wm.DEVICE)
        model.load_state_dict(ck["state"])
        te = tw.load_recs(te_n)
        wm.assign_keys(te, "M1", level, ck["vocab"], np.random.default_rng(s))
        ev = wm.event_list(te)
        S = wm.pooled_stam(te, ev, wm.predict_events(model, te, ev), cell)
        out.setdefault(level, []).append({k: c(S, T[k]) for k in T})
        print(level, s, out[level][-1], flush=True)
summ = {lvl: {k: [float(np.mean([r[k] for r in v])), float(np.std([r[k] for r in v], ddof=1))] for k in T}
        for lvl, v in out.items()}
json.dump({"per_seed": out, "mean_sd": summ}, open(p0.OUT.parent / "creamer_reconcile_m1.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
