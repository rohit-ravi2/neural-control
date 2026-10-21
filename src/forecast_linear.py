"""Angle 1 comparator (exploratory, CPU): class-indexed ridge forecasters on the same split/windows as forecast.py.
  self : per target class, ridge on own last L frames                (generalises the AR(1) baseline)
  pop  : per target class, ridge on own last L frames + last frame of every common class (class-mean, missing = 0)
Scored exactly like forecast.py: change from the last context frame at horizons h, test windows every 15 frames.
    python forecast_linear.py --n-train 28 --seed 0   -> results/forecast/linear_n28_s0.json
"""
import argparse, json, pathlib

import numpy as np

import atanas_data as ad
import worm_models as wm

ap = argparse.ArgumentParser()
ap.add_argument("--n-train", type=int, default=28); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lags", type=int, default=10)
ap.add_argument("--pcs", type=int, default=0, help="if >0, population features = top-k PCs of the class-mean vector (fit on training animals)")
ap.add_argument("--basis-ckpt", default="", help="project population onto a pretrained model's class table ('random' = Gaussian control)")
ap.add_argument("--basis-table", default="enc", choices=["enc", "dec", "stim"])
ap.add_argument("--drive", choices=["none", "causal", "causalT", "anat", "shuf"], default="none")
args = ap.parse_args()
OUT = pathlib.Path(__file__).resolve().parents[1].joinpath("results/forecast").expanduser(); OUT.mkdir(parents=True, exist_ok=True)
H, L, PRE, POST = [2, 10, 30, 60], args.lags, wm.PRE, wm.POST
ALPHAS = [1, 10, 100, 1000, 1e4]

A = ad.load()
perm = np.random.default_rng(0).permutation(len(A))
test_idx, pool = perm[:10], perm[10:]
train_idx = np.random.default_rng(100 + args.seed).choice(pool, args.n_train, replace=False)
cls = [np.array([wm.neuron_class(x) if x else "" for x in A[i][2]]) for i in range(len(A))]
cnt = {}
for i in train_idx:
    for c in set(cls[i]) - {""}:
        cnt[c] = cnt.get(c, 0) + 1
common = sorted(c for c, k in cnt.items() if k >= max(1, 0.5 * len(train_idx)))
cidx = {c: j for j, c in enumerate(common)}


def pop(i):
    """(T, K) class-mean activity of the common classes; 0 where absent."""
    x = A[i][1]; P = np.zeros((x.shape[0], len(common)), np.float32); n = np.zeros(len(common))
    for k, c in enumerate(cls[i]):
        if c in cidx:
            P[:, cidx[c]] += x[:, k]; n[cidx[c]] += 1
    return P / np.maximum(n, 1)


POP = {i: pop(i) for i in list(train_idx) + list(test_idx)}
if args.basis_ckpt:
    import torch
    if args.basis_ckpt == "random":
        E = np.random.default_rng(args.seed).standard_normal((len(common), 32))
    else:
        ck = torch.load(args.basis_ckpt, map_location="cpu", weights_only=False)
        W = ck["state"][f"{args.basis_table}.weight"].numpy(); W = W[:, :-1] if args.basis_table == "dec" else W
        E = np.stack([W[ck["vocab"][c]] if c in ck["vocab"] else np.zeros(W.shape[1]) for c in common])
    POP = {i: P @ E for i, P in POP.items()}
elif args.pcs:
    Ztr = np.concatenate([POP[i] for i in train_idx]); V = np.linalg.svd(Ztr - Ztr.mean(0), full_matrices=False)[2][:args.pcs]
    POP = {i: P @ V.T for i, P in POP.items()}

DRIVE = {}
if args.drive != "none":   # prior-weighted input: d_c(t) = sum_s K[c, s] * classmean_s(t) over classes present in the animal
    import wormneuroatlas as wa
    wa.WormBase.assert_db_version_consistency = lambda self: None
    na = wa.NeuroAtlas(verbose=False)
    if args.drive == "anat":
        M = np.log1p(na.get_chemical_synapses().astype(float)) + np.log1p(na.get_gap_junctions().astype(float))   # [post, pre]
    else:
        M = na.get_signal_propagation_map("wt").astype(float)       # [responder, stimulated]
        if args.drive == "causalT":
            M = M.T
    ac = np.array([wm.neuron_class(str(n)) for n in na.neuron_ids]); U = sorted(set(ac)); ui = {c: j for j, c in enumerate(U)}
    S, C = np.zeros((len(U), len(U))), np.zeros((len(U), len(U)))
    ok = np.isfinite(M); r, c = np.nonzero(ok)
    np.add.at(S, (np.vectorize(ui.get)(ac[r]), np.vectorize(ui.get)(ac[c])), M[r, c]); np.add.at(C, (np.vectorize(ui.get)(ac[r]), np.vectorize(ui.get)(ac[c])), 1)
    K = np.where(C > 0, S / np.maximum(C, 1), 0.0); np.fill_diagonal(K, 0)
    if args.drive == "shuf":   # same matrix, class labels permuted (fixed seed)
        K = K[np.ix_(*(2 * [np.random.default_rng(7).permutation(len(U))]))]
    for i in list(train_idx) + list(test_idx):
        x = A[i][1]; present = sorted(set(cls[i]) - {""} & set(U)); cm = np.zeros((x.shape[0], len(present)), np.float32)
        for j, cc in enumerate(present):
            cm[:, j] = x[:, cls[i] == cc].mean(1)
        Kp = K[np.ix_([ui[cc] for cc in present], [ui[cc] for cc in present])]
        Dm = cm @ Kp.T / (np.abs(Kp).sum(1) + 1e-6)              # (T, classes present)
        DRIVE[i] = {cc: Dm[:, j] for j, cc in enumerate(present)}


def rows(i, k, ts, h, use_pop):
    x = A[i][1][:, k]
    lag = np.stack([x[ts - 1 - j] for j in range(L)], 1)
    F = np.concatenate([lag, POP[i][ts - 1]], 1) if use_pop else lag
    if DRIVE:
        d = DRIVE[i].get(cls[i][k]); d = np.zeros(len(x)) if d is None else d
        F = np.concatenate([F, np.stack([d[ts - 1], d[ts - 3], d[ts - 6]], 1)], 1)
    return F, x[ts - 1 + h] - x[ts - 1]


def collect(idx, h, use_pop, stride):
    D = {}
    for i in idx:
        T = A[i][1].shape[0]; ts = np.arange(PRE, T - POST, stride)
        for k, c in enumerate(cls[i]):
            if c:
                F, y = rows(i, k, ts, h, use_pop); D.setdefault(c, []).append((F, y))
    return {c: (np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])) for c, v in D.items()}


def ridge(F, y, a):
    mu, sd = F.mean(0), F.std(0) + 1e-6; Z = (F - mu) / sd
    w = np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda G: ((G - mu) / sd) @ w + y.mean()


def fit_predict(tr_idx, te_idx, h, use_pop, a):
    Dtr, Dte = collect(tr_idx, h, use_pop, 1), collect(te_idx, h, use_pop, 15)
    P, Y = [], []
    for c, (G, y) in Dte.items():
        if c in Dtr and len(Dtr[c][1]) > 50:
            P.append(ridge(*Dtr[c], a)(G)); Y.append(y)
        else:
            P.append(np.zeros_like(y)); Y.append(y)
    return np.concatenate(P), np.concatenate(Y)


def score(p, y):
    return {"corr_change": float(np.corrcoef(p, y)[0, 1]), "skill_vs_persistence": float(1 - np.mean((y - p) ** 2) / np.mean(y ** 2))}


res = {"args": vars(args), "n_common": len(common)}
for name, use_pop in [("self", False), ("pop", True)]:
    res[name] = {}
    for h in H:
        if len(train_idx) >= 5:   # alpha by an inner split of the training animals only
            k = max(1, len(train_idx) // 5); inner_te, inner_tr = train_idx[:k], train_idx[k:]
            a = max(ALPHAS, key=lambda a: score(*fit_predict(inner_tr, inner_te, h, use_pop, a))["skill_vs_persistence"])
        else:
            a = 100
        r = score(*fit_predict(train_idx, test_idx, h, use_pop, a)); r["alpha"] = a
        res[name][f"h{h}"] = r
json.dump(res, open(OUT / f"linear{'' if args.drive == 'none' else '_' + args.drive}{f'_pc{args.pcs}' if args.pcs and not args.basis_ckpt else ''}{('_B' + (pathlib.Path(args.basis_ckpt).stem if args.basis_ckpt != 'random' else 'random') + '_' + args.basis_table + f'_s{args.seed}') if args.basis_ckpt else ''}_n{args.n_train}_s{args.seed}.json", "w"), indent=1)
print(args.drive, args.n_train, args.seed, json.dumps({m: {h: round(v["skill_vs_persistence"], 3) for h, v in res[m].items()} for m in ["self", "pop"]}))
