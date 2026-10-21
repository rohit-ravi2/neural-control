"""Residual (background-corrected) targets and scoring for any set of animals.
build(names) -> dict with events, raw/residual per-event changes, pooled maps, split-half ceilings.
score(pattern, ...) -> state-free and state-conditioned correlation with T_raw and T_res.
Background model: C seed 4 (train animals only); alpha from results/background_recal.json (fit on training animals' pseudo events).
"""
import glob, json, sys

import numpy as np
import torch

import train_worm as tw
import worm_data as wd
import worm_models as wm

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402

wm.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cell = wd.model_cell_ids(); n = len(cell); col = {c: i for i, c in enumerate(cell)}; eye = np.eye(n, dtype=bool)
PRE, POST = wm.PRE, wm.POST
ALPHA = json.load(open("../results/background_recal.json"))["alpha_fit_on_train"]
_bgc = torch.load("../results/worm_causal_final/M1_class_s4_CF_C_s4.pt", map_location=wm.DEVICE, weights_only=False)
BG = wm.WormNet(_bgc["n_keys"], hidden=128, core="gru").to(wm.DEVICE); BG.load_state_dict(_bgc["state"]); BG.openloop = True
c = lambda a, b: float(met.nan_corr(np.where(eye, np.nan, a), np.where(eye, np.nan, b))[0])
_crop = wm.crop_batch


def _blind(*a, **k):
    x, m, u, keys = _crop(*a, **k); x[:, :PRE] = 0; m[:, :PRE] = 0
    return x, m, u, keys


def measured(r, t):
    clip = r.x[t - PRE:t + POST]
    ok = (np.isfinite(clip[:PRE]).mean(0) > 0.5) & (np.isfinite(clip[PRE:]).mean(0) > 0.5)
    with np.errstate(all="ignore"):
        v = (np.nanmean(clip[PRE:], 0) - np.nanmean(clip[:PRE], 0)) * POST
    v[~ok] = np.nan
    return v


@torch.no_grad()
def background(recs, items):
    max_n = max(r.x.shape[1] for r in recs); out = []
    for i in range(0, len(items), 64):
        it = items[i:i + 64]
        x, m, u, keys = wm.crop_batch(recs, [(ri, t - PRE) for ri, t, _ in it], max_n)
        off = BG.rollout(x, m, u, keys, n_teacher=PRE, stim_on=False)
        pre = (x[:, :PRE] * m[:, :PRE]).sum(1) / m[:, :PRE].sum(1).clamp(min=1)
        b = (off[:, PRE - 1:PRE - 1 + POST].mean(1) - pre) * POST
        out += [b[k, :recs[ri].x.shape[1]].cpu().numpy() for k, (ri, _, _) in enumerate(it)]
    return out


def pool(recs, events, vals):
    acc = np.zeros((n, n)); cnt = np.zeros((n, n))
    for (ri, _, j), v in zip(events, vals):
        tj = col.get(recs[ri].ids[j])
        if tj is None:
            continue
        for i, nm in enumerate(recs[ri].ids):
            ci = col.get(nm)
            if ci is not None and i != j and np.isfinite(v[i]):
                acc[ci, tj] += v[i]; cnt[ci, tj] += 1
    S = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan); S[eye] = np.nan
    return S


def build(names, ceiling=True, seed=0):
    recs = tw.load_recs(names); wm.assign_keys(recs, "M1", "class", _bgc["vocab"], np.random.default_rng(0))
    ev = wm.event_list(recs); mr = [measured(recs[ri], t) for ri, t, _ in ev]; br = background(recs, ev)
    for (ri, t, j), v in zip(ev, mr):
        v[j] = np.nan
    rs = [m - ALPHA * b for m, b in zip(mr, br)]
    D = {"recs": recs, "events": ev, "raw": mr, "res": rs, "T_raw": pool(recs, ev, mr), "T_res": pool(recs, ev, rs)}
    if ceiling:
        rng = np.random.default_rng(seed); animals = np.array([ri for ri, _, _ in ev])
        for key in ("raw", "res"):
            hs = []
            for _ in range(20):
                A = set(rng.permutation(len(recs))[:len(recs) // 2])
                ia = [k for k, a in enumerate(animals) if a in A]; ib = [k for k, a in enumerate(animals) if a not in A]
                hs.append(c(pool(recs, [ev[k] for k in ia], [D[key][k] for k in ia]), pool(recs, [ev[k] for k in ib], [D[key][k] for k in ib])))
            r = float(np.mean(hs)); rtt = 2 * r / (1 + r); D[f"ceiling_{key}"] = float(np.sqrt(max(rtt, 0)))
    return D


def score(D, pattern, scheme="M1", lvl="class", h=128, core="gru"):
    out = {}
    for mode in ("state-free", "state-conditioned"):
        Ss = []
        for p in sorted(glob.glob(pattern)):
            ck = torch.load(p, map_location=wm.DEVICE, weights_only=False)
            m = wm.WormNet(ck["n_keys"], hidden=h, core=core).to(wm.DEVICE); m.load_state_dict(ck["state"]); m.openloop = True
            te = [wm.Rec(r.name, r.x, r.u, r.ids) for r in D["recs"]]; wm.assign_keys(te, scheme, lvl, ck["vocab"], np.random.default_rng(7))
            if mode == "state-free":
                wm.crop_batch = _blind
            pr = wm.predict_events(m, te, D["events"]); wm.crop_batch = _crop
            S = pool(D["recs"], D["events"], pr); Ss.append((S - np.nanmean(S)) / np.nanstd(S))
        E = np.nanmean(Ss, 0) if Ss else None
        out[mode] = {"n_models": len(Ss), "raw": c(E, D["T_raw"]) if Ss else None, "res": c(E, D["T_res"]) if Ss else None}
    return out
