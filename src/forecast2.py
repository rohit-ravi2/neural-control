"""Angle 1 (exploratory): does a perturbation-trained causal model help forecast natural whole-brain dynamics in NEW animals?
Atanas 2023 (freely moving, NeuroPAL). Name(class)-indexed forecaster; init = scratch / Randi-pretrained (with stim, without stim, shuffled names);
fine-tuned on N Atanas training animals; evaluated zero-shot on held-out Atanas animals.
    python forecast.py --init randi --pre-ckpt <path> --n-train 28 --seed 0 [--steps 3000]  -> results/forecast/<tag>.json
"""
import argparse, json, os, pathlib

import numpy as np
import torch

import atanas_data as ad
import worm_models as wm

ap = argparse.ArgumentParser()
ap.add_argument("--init", choices=["scratch", "randi"], default="scratch"); ap.add_argument("--pre-ckpt", default="")
ap.add_argument("--n-train", type=int, default=28); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--tag", default=""); ap.add_argument("--lr", type=float, default=3e-3)
args = ap.parse_args()
wm.DEVICE = os.environ.get("WORM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(int(os.environ.get("WORM_THREADS", "2")))
torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
OUT = pathlib.Path(__file__).resolve().parents[1].joinpath("results/forecast2").expanduser(); OUT.mkdir(parents=True, exist_ok=True)

A = ad.load()
perm = np.random.default_rng(0).permutation(len(A))            # fixed split: 10 held-out animals, never trained on
test_idx, pool = perm[:10], perm[10:]
train_idx = np.random.default_rng(100 + args.seed).choice(pool, args.n_train, replace=False)
cls = lambda ids: np.array([wm.neuron_class(x) if x else "" for x in ids])
recs = {i: wm.Rec(A[i][0], A[i][1], np.zeros_like(A[i][1], dtype=np.int8), cls(A[i][2])) for i in range(len(A))}

# vocabulary: Randi checkpoint classes first (so pretrained rows keep their keys), then new Atanas classes
base_vocab = {}
if args.init == "randi":
    ck = torch.load(args.pre_ckpt, map_location=wm.DEVICE, weights_only=False); base_vocab = dict(ck["vocab"])
vocab = dict(base_vocab)
for i in list(train_idx) + list(test_idx):
    for nm in recs[i].ids:
        if nm and nm not in vocab:
            vocab[nm] = len(vocab)
for r in recs.values():
    r.keys = np.array([vocab.get(nm, -1) if nm else -1 for nm in r.ids])
model = wm.WormNet(len(vocab), hidden=args.hidden, core="gru").to(wm.DEVICE); model.openloop = True
if args.init == "randi":
    st = ck["state"]; own = model.state_dict()
    for k, v in st.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
        elif k in own and own[k].shape[1:] == v.shape[1:]:      # embedding tables: copy the shared rows
            own[k][:v.shape[0]] = v
    model.load_state_dict(own)
tr = [recs[i] for i in train_idx]; te = [recs[i] for i in test_idx]
losses = wm.train(model, tr, args.steps, rng, lr=args.lr, events=[], log_every=0)


H = [2, 10, 30, 60]


def ar1_coefs():
    a = {}
    for h in H:
        num, den = {}, {}
        for r in tr:
            for k, c in enumerate(r.ids):
                if c:
                    x0, x1 = r.x[:-h, k], r.x[h:, k]
                    num[c] = num.get(c, 0) + (x0 * x1).sum(); den[c] = den.get(c, 0) + (x0 ** 2).sum()
        a[h] = {c: num[c] / den[c] for c in num if den[c] > 0}
    return a


AR = ar1_coefs()


@torch.no_grad()
def windows(recs_, stride):
    """per horizon: actual change Y, model raw change, model delta change (vs its own last-frame estimate), AR(1) change."""
    max_n = max(r.x.shape[1] for r in recs_); items = []
    for ri, r in enumerate(recs_):
        for t in range(wm.PRE, r.x.shape[0] - wm.POST, stride):
            items.append((ri, t - wm.PRE))
    out = {h: {k: [] for k in ("y", "raw", "delta", "ar")} for h in H}
    for i in range(0, len(items), 64):
        it = items[i:i + 64]
        x, m, u, keys = wm.crop_batch(recs_, it, max_n)
        pred = model.rollout(x.to(wm.DEVICE), m.to(wm.DEVICE), u.to(wm.DEVICE), keys.to(wm.DEVICE), n_teacher=wm.PRE).cpu()
        x, m, keys = x.cpu(), m.cpu(), keys.cpu()
        last, own = x[:, wm.PRE - 1], pred[:, wm.PRE - 2]
        for h in H:
            A1 = torch.zeros_like(last)
            for b, (ri, _) in enumerate(it):
                for k, c in enumerate(recs_[ri].ids):
                    if c:
                        A1[b, k] = float(AR[h].get(c, 1.0)) - 1
            sel = (m[:, wm.PRE - 1 + h] * m[:, wm.PRE - 1] * (keys >= 0)) > 0
            ph = pred[:, wm.PRE - 2 + h]
            o = out[h]
            o["y"].append((x[:, wm.PRE - 1 + h] - last)[sel].numpy()); o["raw"].append((ph - last)[sel].numpy())
            o["delta"].append((ph - own)[sel].numpy()); o["ar"].append((A1 * last)[sel].numpy())
    return {h: {k: np.concatenate(v) for k, v in d.items()} for h, d in out.items()}


def sc(p, y):
    return {"corr_change": float(np.corrcoef(p, y)[0, 1]), "skill_vs_persistence": float(1 - np.mean((y - p) ** 2) / np.mean(y ** 2))}


def evaluate():
    Wtr, Wte = windows(tr, 15), windows(te, 15); res = {}
    for h in H:
        a, b = Wtr[h], Wte[h]
        beta = float(np.dot(a["delta"], a["y"] - a["ar"]) / np.dot(a["delta"], a["delta"]))     # fitted on TRAINING animals
        bd = float(np.dot(a["delta"], a["y"]) / np.dot(a["delta"], a["delta"]))
        res[f"h{h}"] = {"raw": sc(b["raw"], b["y"]), "delta": sc(b["delta"], b["y"]), "delta_scaled": sc(bd * b["delta"], b["y"]),
                        "ar1": sc(b["ar"], b["y"]), "ar1_plus_delta": sc(b["ar"] + beta * b["delta"], b["y"]), "beta": beta, "beta_delta": bd}
    return res


tag = args.tag or f"{args.init}_{pathlib.Path(args.pre_ckpt).stem if args.pre_ckpt else 'none'}_n{args.n_train}_s{args.seed}"
(OUT / "ckpt").mkdir(exist_ok=True); torch.save({"state": model.state_dict(), "vocab": vocab}, OUT / "ckpt" / f"{tag}.pt")
res = {"args": vars(args), "n_vocab": len(vocab), "train_loss_last500": float(np.mean(losses[-500:])), "test": evaluate()}
tag = args.tag or f"{args.init}_{pathlib.Path(args.pre_ckpt).stem if args.pre_ckpt else 'none'}_n{args.n_train}_s{args.seed}"
json.dump(res, open(OUT / f"{tag}.json", "w"), indent=1)
print(tag, json.dumps({h: {k: round(v["skill_vs_persistence"], 3) for k, v in d.items() if isinstance(v, dict)} for h, d in res["test"].items()}))
