"""LAYER 2: background (no-stimulation) forecast, sham gate, residual target and its ceiling.
    python background_layer.py -> results/background_layer.json (+ results/background_maps.npz)
"""
import glob, json, pickle, sys

import numpy as np
import torch

import phase0_ceiling as p0
import train_worm as tw
import worm_data as wd
import worm_models as wm

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402
import lgssm_utilities as ssmu  # noqa: E402

wm.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cell = wd.model_cell_ids(); n = len(cell); col = {c: i for i, c in enumerate(cell)}; eye = np.eye(n, dtype=bool)
TA = np.load(p0.OUT / "targets_test.npz")
c = lambda a, b: float(met.nan_corr(np.where(eye, np.nan, a), np.where(eye, np.nan, b))[0])
_, te_n, _ = tw.get_split(wd.recording_names())
rng = np.random.default_rng(0)
PRE, POST = wm.PRE, wm.POST

ck = torch.load("../results/worm_causal_final/M1_class_s4_CF_C_s4.pt", map_location=wm.DEVICE, weights_only=False)
bg = wm.WormNet(ck["n_keys"], hidden=128, core="gru").to(wm.DEVICE); bg.load_state_dict(ck["state"]); bg.openloop = True
recs = tw.load_recs(te_n); wm.assign_keys(recs, "M1", "class", ck["vocab"], np.random.default_rng(0))


def measured(r, t):
    clip = r.x[t - PRE:t + POST]
    ok = (np.isfinite(clip[:PRE]).mean(0) > 0.5) & (np.isfinite(clip[PRE:]).mean(0) > 0.5)
    with np.errstate(all="ignore"):
        v = (np.nanmean(clip[PRE:], 0) - np.nanmean(clip[:PRE], 0)) * POST
    v[~ok] = np.nan
    return v


@torch.no_grad()
def background(items, max_n):
    out = []
    for i in range(0, len(items), 64):
        it = items[i:i + 64]
        x, m, u, keys = wm.crop_batch(recs, [(ri, t - PRE) for ri, t, _ in it], max_n)
        off = bg.rollout(x, m, u, keys, n_teacher=PRE, stim_on=False)     # predictions of frames 1..T-1
        post = off[:, PRE - 1:PRE - 1 + POST]                               # frames PRE..PRE+POST-1
        mm = m[:, :PRE]; pre = (x[:, :PRE] * mm).sum(1) / mm.sum(1).clamp(min=1)
        b = (post.mean(1) - pre) * POST
        for k, (ri, _, _) in enumerate(it):
            out.append(b[k, :recs[ri].x.shape[1]].cpu().numpy())
    return out


max_n = max(r.x.shape[1] for r in recs)
real = wm.event_list(recs)
pseudo = []
for ri, r in enumerate(recs):
    st = np.where(r.u.any(1))[0]
    named = [k for k, s in enumerate(r.ids) if s in col]
    for t in range(PRE + 40, r.x.shape[0] - POST, 15):
        if not ((st >= t - PRE - 40) & (st < t + POST)).any():   # no stimulus in the window and >= 20 s before it
            pseudo.append((ri, t, int(rng.choice(named))))
print("real events", len(real), "pseudo events", len(pseudo), flush=True)


def pool(events, vals):
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


out = {"n_real": len(real), "n_pseudo": len(pseudo)}
# GATE
mp = [measured(recs[ri], t) for ri, t, _ in pseudo]; bp = background(pseudo, max_n)
M = np.concatenate([v for v in mp]); B = np.concatenate([b for b in bp]); ok = np.isfinite(M) & np.isfinite(B)
out["gate_var_reduction"] = float(1 - np.var(M[ok] - B[ok]) / np.var(M[ok]))
Spr, Spres = pool(pseudo, mp), pool(pseudo, [m - b for m, b in zip(mp, bp)])
out["gate_pseudo_raw_map_vs_causal"] = c(Spr, TA["T_auth_ours"]); out["gate_pseudo_residual_map_vs_causal"] = c(Spres, TA["T_auth_ours"])
out["gate_pass"] = bool(out["gate_var_reduction"] >= 0.10 and abs(out["gate_pseudo_residual_map_vs_causal"]) < 0.03)
print("GATE", json.dumps({k: v for k, v in out.items() if k.startswith("gate")}), flush=True)
# real events: raw and residual targets
mr = [measured(recs[ri], t) for ri, t, _ in real]; br = background(real, max_n)
for (ri, t, j), v in zip(real, mr):
    v[j] = np.nan
res = [m - b for m, b in zip(mr, br)]
M = np.concatenate(mr); B = np.concatenate(br); ok = np.isfinite(M) & np.isfinite(B)
out["real_var_reduction"] = float(1 - np.var(M[ok] - B[ok]) / np.var(M[ok]))
T_raw, T_res = pool(real, mr), pool(real, res)
# split-half ceilings by animal
def halves(vals):
    rs = []
    animals = np.array([ri for ri, _, _ in real])
    for _ in range(20):
        A = set(rng.permutation(len(recs))[:len(recs) // 2])
        ia = [k for k, a in enumerate(animals) if a in A]; ib = [k for k, a in enumerate(animals) if a not in A]
        rs.append(c(pool([real[k] for k in ia], [vals[k] for k in ia]), pool([real[k] for k in ib], [vals[k] for k in ib])))
    r = float(np.mean(rs)); rtt = 2 * r / (1 + r)
    return {"r_half": r, "r_tt": rtt, "ceiling": float(np.sqrt(max(rtt, 0)))}
out["ceiling_raw"], out["ceiling_res"] = halves(mr), halves(res)
out["T_raw_vs_T_res"] = c(T_raw, T_res); out["T_raw_vs_authors"] = c(T_raw, TA["T_auth_ours"])
np.savez("../results/background_maps.npz", T_raw=T_raw, T_res=T_res)
print(json.dumps(out, indent=1)); json.dump(out, open("../results/background_layer.json", "w"), indent=1)
