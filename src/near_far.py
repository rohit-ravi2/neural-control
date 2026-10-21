"""Near/far split of atlas reproducibility and model scores.
    python near_far.py -> results/near_far.json
"""
import collections, json, pickle, sys

import numpy as np
import torch

import phase0_ceiling as p0
import train_worm as tw
import worm_data as wd
import worm_models as wm

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402
import lgssm_utilities as ssmu  # noqa: E402

cell = wd.model_cell_ids(); n = len(cell); col = {c: i for i, c in enumerate(cell)}; eye = np.eye(n, dtype=bool)
POS = dict(np.load(__import__("pathlib").Path(__file__).resolve().parents[1].joinpath("work/roi_pos.npz").expanduser()))
dist = collections.defaultdict(list)
for rec, C in POS.items():
    ids = [str(x) for x in wd.load_recording(rec)["cell_ids"]]
    named = [(k, col[s]) for k, s in enumerate(ids) if s in col]
    for a, ia in named:
        for b, ib in named:
            if ia != ib:
                dist[(ia, ib)].append(np.linalg.norm(C[a] - C[b]))
D = np.full((n, n), np.nan)
for (i, j), v in dist.items():
    if len(v) >= 3:
        D[i, j] = np.median(v)
near, far = (D < 10) & ~eye, (D >= 15) & ~eye
TA = np.load(p0.OUT / "targets_test.npz"); T = TA["T_auth_ours"]
c = lambda a, b, m: float(met.nan_corr(np.where(m, a, np.nan), np.where(m, b, np.nan))[0])
cnt = lambda m: int((m & np.isfinite(T)).sum())
if __name__ == "__main__":
    preds = {"train->test (rho)": TA["T_auth_train"]}
    for mn in ("connectome_constrained", "fully_connected"):
        mdl = pickle.load(open(wd.REPO / "models" / f"{mn}.pkl", "rb")); irfs = ssmu.calculate_irfs(mdl, window=(15, 30))
        preds[f"Creamer {mn}"] = irfs[15:].sum(0)
    wm.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _, te_n, _ = tw.get_split(wd.recording_names())
    for tag, path, lvl, core in (("M1 neuron (GRU v2)", "../results/worm_v2/M1_neuron_s0.pt", "neuron", "gru"),
                                 ("M1 class (GRU v2)", "../results/worm_v2/M1_class_s0.pt", "class", "gru"),
                                 ("M1 linear-diag OL", "../results/worm_block2/M1_neuron_s0_LINdiagol_s0.pt", "neuron", "lineardiag")):
        ck = torch.load(path, map_location=wm.DEVICE, weights_only=False)
        mdl = wm.WormNet(ck["n_keys"], core=core).to(wm.DEVICE); mdl.load_state_dict(ck["state"]); mdl.openloop = core != "gru"
        te = tw.load_recs(te_n); wm.assign_keys(te, "M1", lvl, ck["vocab"], np.random.default_rng(0)); ev = wm.event_list(te)
        preds[tag] = wm.pooled_stam(te, ev, wm.predict_events(mdl, te, ev), cell)
    out = {"n_entries": {"near": cnt(near), "far": cnt(far), "all": cnt(~eye)}}
    for k, P in preds.items():
        out[k] = {"all": c(P, T, ~eye), "near": c(P, T, near), "far": c(P, T, far)}
        print(f"{k:32s} all {out[k]['all']:.3f}  near {out[k]['near']:.3f}  far {out[k]['far']:.3f}", flush=True)
    print(out["n_entries"])
    json.dump(out, open("../results/near_far.json", "w"), indent=1)
