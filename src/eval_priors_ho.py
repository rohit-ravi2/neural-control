"""STATE-FREE near/far scoring of the held-out-class strong-prior runs.
    python eval_priors_ho.py -> results/priors_ho_nearfar.json
"""
import collections, glob, json, re, sys

import numpy as np
import torch

import connectome as cx
import expression as ex
import near_far as nf
import train_worm as tw
import worm_data as wd
import worm_models as wm

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402

wm.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T = nf.TA["T_auth_ours"]; cell = nf.cell
c = lambda a, m: float(met.nan_corr(np.where(m, a, np.nan), np.where(m, T, np.nan))[0])
_, te_n, _ = tw.get_split(wd.recording_names())
cls_cell = ex.features(cell)[2]; all_cls = sorted({x for x in cls_cell if x})
R = collections.defaultdict(list)
for p in sorted(glob.glob("../results/worm_priors_ho/*.pt")):
    tag = re.search(r"_(HO_\w+?)_f(\d)\.pt$", p); name, k = tag.group(1), int(tag.group(2))
    ck = torch.load(p, map_location=wm.DEVICE, weights_only=False); vocab = ck["vocab"]; vn = sorted(vocab, key=vocab.get)
    held = set(np.random.default_rng(12345).permutation(all_cls)[k::5])
    G, ok, g_cls = ex.features(vn)
    if name == "HO_m1":
        m = wm.WormNet(ck["n_keys"], hidden=128, core="gru"); m.load_state_dict(ck["state"])
        hk = [i for i, cc in enumerate(g_cls) if cc in held]; keep = torch.ones(ck["n_keys"], dtype=torch.bool); keep[hk] = False
        with torch.no_grad():
            for nm in m.row_names():
                w = getattr(m, nm).weight; w[hk] = w[keep].mean(0)
        ok = np.ones(len(vn), bool)
    else:
        if name == "HO_connectome":
            G, ok = cx.features(vn)
        elif name == "HO_shuf":
            cl = sorted({x for x in g_cls if x}); perm = dict(zip(cl, np.random.default_rng(1000 + 0).permutation(cl)))
            Gc = {x: G[g_cls.index(x)] for x in cl}; G = np.stack([Gc[perm[x]] if x else G[i] for i, x in enumerate(g_cls)])
        m = wm.ExprNet(G, hidden=128); m.load_state_dict(ck["state"])
    m = m.to(wm.DEVICE); m.openloop = True
    te = tw.load_recs(te_n); wm.assign_keys(te, "M1", "neuron", vocab, np.random.default_rng(0))
    for r in te:
        r.keys = np.where((r.keys >= 0) & ~ok[r.keys.clip(0)], -1, r.keys)
    ev = wm.event_list(te)
    _crop = wm.crop_batch
    def _blind(*a, **k):   # state-free (causal) evaluation
        x_, m_, u_, k_ = _crop(*a, **k); x_[:, :wm.PRE] = 0; m_[:, :wm.PRE] = 0
        return x_, m_, u_, k_
    wm.crop_batch = _blind
    S = wm.pooled_stam(te, ev, wm.predict_events(m, te, ev), cell)
    wm.crop_batch = _crop
    h = np.array([x in held for x in cls_cell]); touch = (h[:, None] | h[None, :]) & ~nf.eye
    row = {"held_all": c(S, touch), "held_near": c(S, touch & nf.near), "held_far": c(S, touch & nf.far), "seen_far": c(S, ~touch & nf.far)}
    R[name].append(row); print(name, k, {a: round(b, 3) for a, b in row.items()}, flush=True)
out = {k: {m_: [float(np.mean([r[m_] for r in v])), float(np.std([r[m_] for r in v], ddof=1)) if len(v) > 1 else None, len(v)] for m_ in v[0]} for k, v in R.items()}
for a in ("HO_connectome", "HO_cengen"):
    if a in R and "HO_shuf" in R:
        d = np.array([x["held_far"] for x in R[a]]) - np.array([x["held_far"] for x in R["HO_shuf"]])
        out[f"{a} - shuf (held_far)"] = [float(d.mean()), float(d.std(ddof=1)), int((d > 0).sum())]
print(json.dumps(out, indent=1)); json.dump(out, open("../results/priors_ho_nearfar.json", "w"), indent=1)
