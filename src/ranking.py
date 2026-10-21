"""LAYER 1: retrospective intervention ranking on held-out animals.
    python ranking.py -> results/ranking.json
"""
import glob, json, pickle, sys, warnings

import numpy as np
import torch
from scipy.stats import spearmanr

import phase0_ceiling as p0
import train_worm as tw
import worm_data as wd
import worm_models as wm

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import lgssm_utilities as ssmu  # noqa: E402

wm.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cell = wd.model_cell_ids(); n = len(cell); eye = np.eye(n, dtype=bool)
TA = np.load(p0.OUT / "targets_test.npz"); T = TA["T_auth_ours"]
_, te_n, _ = tw.get_split(wd.recording_names())
_crop = wm.crop_batch


def blind(*a, **k):
    x, m, u, keys = _crop(*a, **k); x[:, :wm.PRE] = 0; m[:, :wm.PRE] = 0
    return x, m, u, keys


seeds = []
for p in sorted(glob.glob("../results/worm_causal_final/M1_class_s[0-9]_CF_C_s[0-9].pt")):
    ck = torch.load(p, map_location=wm.DEVICE, weights_only=False)
    m = wm.WormNet(ck["n_keys"], hidden=128, core="gru").to(wm.DEVICE); m.load_state_dict(ck["state"]); m.openloop = True
    te = tw.load_recs(te_n); wm.assign_keys(te, "M1", "class", ck["vocab"], np.random.default_rng(0)); ev = wm.event_list(te)
    wm.crop_batch = blind; S = wm.pooled_stam(te, ev, wm.predict_events(m, te, ev), cell); wm.crop_batch = _crop
    seeds.append((S - np.nanmean(S)) / np.nanstd(S))
P = {"C (state-free ensemble)": np.nanmean(seeds, 0), "atlas lookup (train animals)": TA["T_auth_train"]}
for mn in ("connectome_constrained", "fully_connected"):
    mdl = pickle.load(open(wd.REPO / "models" / f"{mn}.pkl", "rb")); P[f"Creamer {mn}"] = ssmu.calculate_irfs(mdl, window=(15, 30))[15:].sum(0)
P["random"] = np.random.default_rng(0).normal(size=(n, n))
SD = np.nanstd(seeds, 0)


def scores(Pm, Tm):
    act, sel, top1, top3 = [], [], [], []
    for r in range(n):
        cand = np.where(np.isfinite(Tm[r]) & np.isfinite(Pm[r]) & ~eye[r])[0]
        if len(cand) < 8:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            off_p = np.array([np.nanmean(np.abs(np.delete(Pm[:, j], r))) for j in cand])
            off_t = np.array([np.nanmean(np.abs(np.delete(Tm[:, j], r))) for j in cand])
        pa, ta = Pm[r, cand], Tm[r, cand]
        ps, ts = pa / np.nanstd(Pm) - off_p / np.nanstd(Pm), ta / np.nanstd(Tm) - off_t / np.nanstd(Tm)
        act.append(spearmanr(pa, ta).correlation); sel.append(spearmanr(ps, ts).correlation)
        ch = np.argmax(pa); top1.append(float((ta < ta[ch]).mean())); top3.append(float(ch in np.argsort(-ta)[:3]))
    return {"activate_spearman": float(np.nanmean(act)), "selective_spearman": float(np.nanmean(sel)),
            "top1_percentile": float(np.mean(top1)), "top3_hit": float(np.mean(top3)), "n_goals": len(act)}


out = {k: scores(v, T) for k, v in P.items()}
# split-half reference: rank on half A of the test animals, score on half B
rng = np.random.default_rng(1); ref = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _ in range(10):
        p_ = rng.permutation(te_n); A, B = p0.authors_stams(sorted(p_[:15]), cell), p0.authors_stams(sorted(p_[15:]), cell)
        A[eye] = np.nan; B[eye] = np.nan; ref.append(scores(A, B))
out["split-half reference (15 animals -> 15 animals)"] = {k: float(np.mean([r[k] for r in ref])) for k in ref[0]}
# abstention curve for C: margin of the top choice / seed SD
Pc = P["C (state-free ensemble)"]; rows = []
for r in range(n):
    cand = np.where(np.isfinite(T[r]) & np.isfinite(Pc[r]) & ~eye[r])[0]
    if len(cand) < 8:
        continue
    pa, ta = Pc[r, cand], T[r, cand]; o = np.argsort(-pa)
    conf = (pa[o[0]] - pa[o[1]]) / (SD[r, cand[o[0]]] + 1e-9); rows.append((conf, float((ta < ta[o[0]]).mean())))
rows = np.array(sorted(rows, key=lambda x: -x[0]))
out["C abstention (top-1 percentile at coverage)"] = {f"{int(cv*100)}%": float(rows[:max(1, int(cv * len(rows))), 1].mean()) for cv in (0.25, 0.5, 0.75, 1.0)}
for k, v in out.items():
    print(f"{k:48s} {json.dumps({a: round(b, 3) for a, b in v.items()})}")
json.dump(out, open("../results/ranking.json", "w"), indent=1)
