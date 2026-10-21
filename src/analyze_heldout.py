"""Phase 3 mechanism split (eval-only, CPU): for never-trained neuron classes, does the prior predict how the neuron
RESPONDS (held-out rows) or what STIMULATING it does (held-out columns)? Plus a per-class map.
p3b checkpoints (5 folds x 2 seeds) for M3h (CeNGEN), Mch (connectome), M1h (fallback); target = authors' estimator.
    python analyze_heldout.py -> results/heldout_split.json
"""
import collections, json, sys

import numpy as np
import torch

import connectome as cx
import expression as ex
import phase0_ceiling as p0
import train_worm as tw
import worm_data as wd
import worm_models as wm

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402

torch.set_num_threads(6)
wm.DEVICE = "cpu"
cell = wd.model_cell_ids(); n = len(cell)
TA = np.load(p0.OUT / "targets_test.npz"); T, TR = TA["T_auth_ours"], TA["T_auth_train"]
names = wd.recording_names(); tr_n, te_n, _ = tw.get_split(names)
tr = tw.load_recs(tr_n)
cls_of = ex.features(cell)[2]
all_cls = sorted({c for c in cls_of if c})
R = "../results/worm_p3b"


def corr(a, b, m):
    return float(met.nan_corr(np.where(m, a, np.nan), np.where(m, b, np.nan))[0])


out = collections.defaultdict(list)
percls = collections.defaultdict(lambda: collections.defaultdict(list))
for s in (0, 1):
    for k in range(5):
        held = set(np.random.default_rng(12345).permutation(all_cls)[k::5])
        h = np.array([c in held for c in cls_of])
        masks = {"rows": h[:, None] & ~h[None, :], "cols": ~h[:, None] & h[None, :], "both": h[:, None] & h[None, :]}
        for m, prior in (("M3h", "cengen"), ("Mch", "connectome"), ("M1h", None)):
            ck = torch.load(f"{R}/{'M1' if m == 'M1h' else 'M3'}_neuron_s{s}_{m}_f{k}_s{s}.pt", map_location="cpu", weights_only=False)
            vocab = ck["vocab"]; vn = sorted(vocab, key=vocab.get)
            if m == "M1h":
                model = wm.WormNet(ck["n_keys"]); model.load_state_dict(ck["state"])
                g_cls = ex.features(vn)[2]
                hk = [i for i, c in enumerate(g_cls) if c in held]
                keep = torch.ones(ck["n_keys"], dtype=torch.bool); keep[hk] = False
                with torch.no_grad():
                    for nm in model.row_names():
                        w = getattr(model, nm).weight; w[hk] = w[keep].mean(0)
                ok = np.ones(len(vn), bool)
            else:
                G, ok, _ = ex.features(vn)
                if prior == "connectome":
                    G, ok = cx.features(vn)
                model = wm.ExprNet(G); model.load_state_dict(ck["state"])
            te = tw.load_recs(te_n)
            wm.assign_keys(te, "M1", "neuron", vocab, np.random.default_rng(s))
            for r in te:
                r.keys = np.where((r.keys >= 0) & ~ok[r.keys.clip(0)], -1, r.keys)
            ev = wm.event_list(te)
            S = wm.pooled_stam(te, ev, wm.predict_events(model, te, ev), cell)
            row = {mk: corr(S, T, mm) for mk, mm in masks.items()}
            row.update({f"rho_{mk}": corr(TR, T, mm) for mk, mm in masks.items()})
            out[m].append(row)
            for c in held:
                hc = np.array([x == c for x in cls_of]); mc = hc[:, None] | hc[None, :]
                if np.isfinite(np.where(mc, T, np.nan)).sum() >= 10:
                    percls[m][c].append(corr(S, T, mc))
            print(s, k, m, {a: round(b, 3) for a, b in row.items()}, flush=True)

summ = {m: {key: [float(np.nanmean([r[key] for r in rs])), float(np.nanstd([r[key] for r in rs], ddof=1))] for key in rs[0]} for m, rs in out.items()}
pc = {m: {c: float(np.mean(v)) for c, v in d.items()} for m, d in percls.items()}
json.dump({"summary": summ, "per_run": out, "per_class": pc}, open("../results/heldout_split.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
