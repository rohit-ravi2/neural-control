"""L2: residual target with the train-fitted alpha; ceilings and model scores on raw vs residual targets (same events, same pooling).
    python residual_eval.py -> results/residual_eval.json (+ results/residual_maps.npz)
"""
import glob, json, pickle

import numpy as np
import torch

import background_layer as bl
import lgssm_utilities as ssmu
import worm_data as wd
import worm_models as wm

alpha = json.load(open("../results/background_recal.json"))["alpha_fit_on_train"]
real, recs = bl.real, bl.recs
mr = [bl.measured(recs[ri], t) for ri, t, _ in real]; br = bl.background(real, bl.max_n)
for (ri, t, j), v in zip(real, mr):
    v[j] = np.nan
res = [m - alpha * b for m, b in zip(mr, br)]
M, B = np.concatenate(mr), np.concatenate(br); ok = np.isfinite(M) & np.isfinite(B)
out = {"alpha": alpha, "real_var_removed": float(1 - np.var(M[ok] - alpha * B[ok]) / np.var(M[ok]))}
T_raw, T_res = bl.pool(real, mr), bl.pool(real, res)
out["ceiling_raw"], out["ceiling_res"] = bl.halves(mr), bl.halves(res)
_crop = wm.crop_batch


def blind(*a, **k):
    x, m, u, keys = _crop(*a, **k); x[:, :wm.PRE] = 0; m[:, :wm.PRE] = 0
    return x, m, u, keys


def model_map(pattern, scheme, lvl, h, core):
    Ss = []
    for p in sorted(glob.glob(pattern)):
        ck = torch.load(p, map_location=wm.DEVICE, weights_only=False)
        m = wm.WormNet(ck["n_keys"], hidden=h, core=core).to(wm.DEVICE); m.load_state_dict(ck["state"]); m.openloop = True
        te = [wm.Rec(r.name, r.x, r.u, r.ids) for r in recs]; wm.assign_keys(te, scheme, lvl, ck["vocab"], np.random.default_rng(7))
        wm.crop_batch = blind; pr = wm.predict_events(m, te, real); wm.crop_batch = _crop
        S = bl.pool(real, pr); Ss.append((S - np.nanmean(S)) / np.nanstd(S))
    return np.nanmean(Ss, 0)


P = {"C state-free ensemble": model_map("../results/worm_causal_final/M1_class_s[0-9]_CF_C_s[0-9].pt", "M1", "class", 128, "gru"),
     "shuffled identity (v2 M2 class)": model_map("../results/worm_v2/M2_class_s[0-9].pt", "M2", "class", 64, "gru")}
for mn in ("connectome_constrained", "fully_connected"):
    mdl = pickle.load(open(wd.REPO / "models" / f"{mn}.pkl", "rb")); P[f"Creamer {mn}"] = ssmu.calculate_irfs(mdl, window=(15, 30))[15:].sum(0)
for k, S in P.items():
    out[k] = {"raw": bl.c(S, T_raw), "res": bl.c(S, T_res),
              "raw_frac_ceiling": bl.c(S, T_raw) / out["ceiling_raw"]["ceiling"], "res_frac_ceiling": bl.c(S, T_res) / out["ceiling_res"]["ceiling"]}
np.savez("../results/residual_maps.npz", T_raw=T_raw, T_res=T_res)
print(json.dumps(out, indent=1)); json.dump(out, open("../results/residual_eval.json", "w"), indent=1)
