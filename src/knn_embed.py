"""Model-based imputation for never-trained classes: in the p3b M1h checkpoints (GRU M1, held-out classes untrained),
replace each held-out key's enc/stim/dec rows by the similarity-weighted mean of TRAINED keys' rows (similarity = the banked
config: CeNGEN + connectome z-concat, softmax T=5, k=40; selected on training animals in knn_nested.py). Compared with the
mean-row fallback (M1h) and the measured-map kNN (0.147). Eval-only (CPU); authors' estimator; held-out-class entries.
    python knn_embed.py -> results/knn_embed.json
"""
import json, sys

import numpy as np
import torch

import connectome as cx
import expression as ex
import knn_prior as kp
import phase0_ceiling as p0
import train_worm as tw
import worm_data as wd
import worm_models as wm

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402

torch.set_num_threads(6); wm.DEVICE = "cpu"
cell = wd.model_cell_ids(); TA = np.load(p0.OUT / "targets_test.npz"); T = TA["T_auth_ours"]
names = wd.recording_names(); _, te_n, _ = tw.get_split(names)
all_cls = kp.all_cls
zn = lambda G: G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
out = {"mean_row": [], "knn_rows": []}
for s in (0, 1):
    for k in range(5):
        held = set(np.random.default_rng(12345).permutation(all_cls)[k::5])
        ck = torch.load(f"../results/worm_p3b/M1_neuron_s{s}_M1h_f{k}_s{s}.pt", map_location="cpu", weights_only=False)
        vocab = ck["vocab"]; vn = sorted(vocab, key=vocab.get)
        Ge, _, g_cls = ex.features(vn); Gc, _ = cx.features(vn)
        F = zn(np.concatenate([zn(Ge), zn(Gc)], 1)); S = F @ F.T
        hk = np.array([i for i, c in enumerate(g_cls) if c in held]); tk = np.array([i for i, c in enumerate(g_cls) if c not in held and c])
        h = np.array([c in held for c in ex.features(cell)[2]]); touch = h[:, None] | h[None, :]
        for mode in ("mean_row", "knn_rows"):
            model = wm.WormNet(ck["n_keys"]); model.load_state_dict(ck["state"])
            with torch.no_grad():
                for nm in model.row_names():
                    w = getattr(model, nm).weight
                    if mode == "mean_row":
                        w[hk] = w[tk].mean(0)
                    else:
                        for i in hk:
                            nb = tk[np.argsort(-S[i, tk])[:40]]; a = np.exp(5 * S[i, nb]); a /= a.sum()
                            w[i] = (torch.from_numpy(a).float()[:, None] * w[nb]).sum(0)
            te = tw.load_recs(te_n); wm.assign_keys(te, "M1", "neuron", vocab, np.random.default_rng(s))
            ev = wm.event_list(te)
            M = wm.pooled_stam(te, ev, wm.predict_events(model, te, ev), cell)
            r = float(met.nan_corr(np.where(touch, M, np.nan), np.where(touch, T, np.nan))[0])
            out[mode].append(r); print(s, k, mode, round(r, 3), flush=True)
summ = {m: [float(np.mean(v)), float(np.std(v, ddof=1))] for m, v in out.items()}
d = np.array(out["knn_rows"]) - np.array(out["mean_row"])
summ["knn_minus_mean"] = [float(d.mean()), float(d.std(ddof=1)), int((d > 0).sum())]
print(json.dumps(summ)); json.dump({"per_run": out, "summary": summ}, open("../results/knn_embed.json", "w"), indent=1)
