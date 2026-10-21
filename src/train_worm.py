"""Phase 1/2 driver: train one worm model and evaluate zero-shot + few-shot.

    python train_worm.py --scheme M1 --level neuron --seed 0

Split: results/phase0/split.json if present (written once Phase 0 settles which split
reproduces the paper), else the positional 80/30 split of load_data().
Outputs results/worm/<scheme>_<level>_s<seed>.json with raw correlation, paper-normalised
score (divided by rho of measured train STAM vs measured eval STAM on the SAME eval events)
and parameter counts.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

import phase0_ceiling as p0
import worm_data as wd
import worm_models as wm
import expression as ex
import connectome as cx

torch.set_num_threads(int(os.environ.get("WORM_THREADS", "4")))
wm.DEVICE = os.environ.get("WORM_DEVICE", "cpu")
OUT = pathlib.Path(os.environ.get("WORM_OUT", str(pathlib.Path(__file__).resolve().parents[1] / "results/worm"))).expanduser()
sys.path.insert(0, str(wd.STUBS))
sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402


CLEAN = os.environ.get("WORM_CLEAN", "1") == "1"


def load_recs(names):
    """v2 cleaning (WORM_CLEAN=1, default):
    - |x| > 5 -> NaN for every column. The authors' pipeline drops only x > 5, so large NEGATIVE
      artifacts survive (unnamed columns reach |x| = 2395), and they dominated M0's loss in v1.
    - For each stimulation event at frame s on column j, x[s-2 : s+3, j] -> NaN. lu.preprocess_data
      overwrites those frames by interpolating toward x[s+3, j], so the stimulated neuron's last two
      pre-stimulus values carry its post-stimulus response into the burn-in (a leak in v1)."""
    recs = []
    for n in names:
        r = wd.load_recording(n)
        x = r["emissions"].copy()
        if CLEAN:
            x[np.abs(x) > 5] = np.nan
            for t, j in zip(*np.where(r["inputs"] == 1)):
                x[max(t - 2, 0) : t + 3, j] = np.nan
        recs.append(wm.Rec(n, x, r["inputs"], np.array([str(s) for s in r["cell_ids"]])))
    return recs


def get_split(names):
    f = p0.OUT / "split.json"
    if f.exists():
        d = json.load(open(f))
        return d["train"], d["test"], d.get("source", "split.json")
    tr, te = p0.positional_split(names)
    return tr, te, "positional"


def score(recs, events, responses, names, measured_train_stam, entry_mask=None):
    """corr(model pooled STAM, measured pooled STAM on the same events); rho on the same events.
    entry_mask (bool, len(names) x len(names)): score only those (responder, target) entries."""
    model_s = wm.pooled_stam(recs, events, responses, names)
    meas_s = wm.pooled_stam(recs, events, wm.measured_event_responses(recs, events), names)
    if entry_mask is not None:
        model_s, meas_s = np.where(entry_mask, model_s, np.nan), np.where(entry_mask, meas_s, np.nan)
        measured_train_stam = np.where(entry_mask, measured_train_stam, np.nan)
    r = float(met.nan_corr(model_s, meas_s)[0])
    rho = float(met.nan_corr(measured_train_stam, meas_s)[0])
    return {"r": r, "rho_same_events": rho, "paper_relative": r / rho if rho else None,
            "n_events": len(events)}


def few_shot(base: wm.WormNet, scheme: str, test_recs, n_cal: int, steps: int, rng, shared_keys=None,
             lr: float = 3e-3, init: str | None = None, anchor: float = 0.0):
    """Fit animal-specific table rows for every test animal on its first n_cal events, core frozen.
    init: 'random' (M0 default) | 'popmean' (mean of trained rows; identity-free prior) |
    'shared' (M1/M2 default: rows copied from the shared name rows).
    anchor: L2 pull of the new rows toward their initial values (lambda * mean squared deviation)."""
    init = init or ("random" if scheme == "M0" else "shared")
    model = copy.deepcopy(base)
    recs = [copy.copy(r) for r in test_recs]
    n_new = sum(len(r.ids) for r in recs)
    start = model.enc.num_embeddings
    model.grow(n_new, init="popmean" if init == "popmean" else "random")
    k = start
    cal_range, eval_range = {}, {}
    for i, r in enumerate(recs):
        ev_frames = sorted(t for t, _ in zip(*np.where(r.u == 1)))
        cut = ev_frames[n_cal - 1] + 1 if len(ev_frames) > n_cal else None
        new_keys = np.arange(k, k + len(r.ids))
        if init == "shared":
            old = shared_keys[i]
            with torch.no_grad():
                for name in model.row_names():
                    w = getattr(model, name).weight
                    ok = old >= 0
                    w[torch.from_numpy(new_keys[ok]).to(w.device)] = w[torch.from_numpy(old[ok]).to(w.device)]
            new_keys = np.where(shared_keys[i] >= 0, new_keys, -1)
        r.keys = new_keys
        k += len(r.ids)
        cal_range[i] = (0, cut if cut else 0)
        eval_range[i] = (cut if cut else r.x.shape[0], r.x.shape[0])
    for name, p in model.named_parameters():
        if name.split(".")[0] not in model.row_names():
            p.requires_grad_(False)
    cal_events = wm.event_list(recs, frame_range=cal_range)
    if cal_events and steps > 0:
        # restrict random crops to the calibration segment by training on truncated copies
        cal_recs = [wm.Rec(r.name, r.x[: max(cal_range[i][1] + wm.POST, wm.PRE + wm.POST + 1)],
                           r.u[: max(cal_range[i][1] + wm.POST, wm.PRE + wm.POST + 1)], r.ids, r.keys)
                    for i, r in enumerate(recs)]
        cal_events = [e for e in wm.event_list(cal_recs, frame_range=cal_range)]
        rows = [getattr(model, n).weight for n in model.row_names()]
        mask = torch.zeros(model.enc.num_embeddings, dtype=torch.bool, device=wm.DEVICE)
        mask[start:] = True
        for w in rows:
            w.register_hook(lambda g, m=mask: g * m[:, None])
        reg = None
        if anchor > 0:
            init_rows = [w.detach()[start:].clone() for w in rows]
            reg = lambda: anchor * sum(((w[start:] - w0) ** 2).mean() for w, w0 in zip(rows, init_rows))
        wm.train(model, cal_recs, steps, rng, batch=32, lr=lr, params=rows, events=cal_events,
                 log_every=0, reg=reg)
    return model, recs, eval_range


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["M0", "M1", "M2", "M3", "M4"], default="M1")
    ap.add_argument("--level", choices=["neuron", "class"], default="neuron")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--fewshot-steps", type=int, default=400)
    ap.add_argument("--n-cal", type=int, nargs="*", default=[5, 10, 20])
    ap.add_argument("--tag", default="")
    ap.add_argument("--tie", action="store_true", help="stim vector = shared map of the encoder row")
    ap.add_argument("--rank", type=int, default=0, help="low-rank per-key tables (0 = full)")
    ap.add_argument("--core", choices=["gru", "linear", "lineardiag", "linearsn"], default="gru")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--openloop", action="store_true", help="no prediction feedback after burn-in (stable linear cores)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--score-auth", action="store_true",
                    help="also score zero-shot against the authors' estimator on our test animals (targets_test.npz)")
    ap.add_argument("--no-stim", action="store_true", help="pretraining control: stimulation inputs removed from the training animals")
    ap.add_argument("--bg-ckpt", default="", help="frozen background model (class-level GRU, same vocab) for residual response targets")
    ap.add_argument("--bg-alpha", type=float, default=0.0)
    ap.add_argument("--resp-blind", action="store_true", help="state-free response loss (burn-in observations hidden)")
    ap.add_argument("--eval-blind", action="store_true", help="state-free zero-shot evaluation (blind burn-in), the causal score")
    ap.add_argument("--resp-loss", type=float, default=0.0, help="weight of the event-level counterfactual response loss")
    ap.add_argument("--inner", action="store_true", help="selection split: 60/20 within the 80 training animals (seed 555)")
    ap.add_argument("--fold", type=int, default=-1, help="5-fold CV over all 110 animals (seed 777); saves pooled-prediction accumulators")
    ap.add_argument("--genes", default="all", help="Phase 3c gene group for the CeNGEN prior (expression.GROUPS)")
    ap.add_argument("--fs-lr", type=float, default=3e-3)
    ap.add_argument("--fs-init", choices=["random", "popmean", "shared"], default=None)
    ap.add_argument("--fs-anchor", type=float, default=0.0)
    ap.add_argument("--from-ckpt", default="", help="skip training; load this checkpoint (few-shot sweeps)")
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="hold out this fraction of CeNGEN classes (fixed set, seed 12345) from training entirely")
    ap.add_argument("--holdout-fold", type=int, default=-1, help="hold out CV fold k of --holdout-nfolds (overrides frac)")
    ap.add_argument("--holdout-nfolds", type=int, default=5)
    ap.add_argument("--prior", choices=["cengen", "connectome", "both"], default="cengen", help="M3/M4 feature prior")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()

    names = wd.recording_names()
    train_n, test_n, split_src = get_split(names)
    if args.inner:
        p_ = list(np.random.default_rng(555).permutation(train_n)); test_n = sorted(p_[60:]); train_n = sorted(p_[:60]); split_src = "inner 60/20 of train (seed 555)"
    if args.fold >= 0:
        perm = list(np.random.default_rng(777).permutation(names))
        test_n = sorted(perm[args.fold::5]); train_n = sorted(set(names) - set(test_n)); split_src = f"cv5 fold {args.fold} (seed 777)"
    cell_ids = wd.model_cell_ids()
    tr, te = load_recs(train_n), load_recs(test_n)
    if args.no_stim:
        for r in tr:
            r.u = np.zeros_like(r.u)
    expr = args.scheme in ("M3", "M4")
    if expr:
        assert args.level == "neuron"
        args.n_cal = []  # no per-key rows to calibrate: M3/M4 are scored zero-shot
    vocab, n_keys = wm.assign_keys(tr, "M1" if expr else args.scheme, args.level, None, rng)
    vnames = sorted(vocab, key=vocab.get)
    G, g_ok, g_cls = ex.features(vnames, genes=args.genes)   # g_cls (CeNGEN class) also defines the held-out classes
    if expr and args.prior != "cengen":
        Gc, c_ok = cx.features(vnames)
        G, g_ok = (Gc, c_ok) if args.prior == "connectome" else (np.concatenate([G, Gc], 1), g_ok & c_ok)
    if args.scheme == "M4":  # shuffle whole class profiles across classes (L/R keep sharing a profile)
        cl = sorted({c for c in g_cls if c})
        perm = dict(zip(cl, np.random.default_rng(1000 + args.seed).permutation(cl)))
        Gc = {c: G[g_cls.index(c)] for c in cl}
        G = np.stack([Gc[perm[c]] if c else G[i] for i, c in enumerate(g_cls)])
    held_keys = np.array([], int)
    if args.holdout_frac > 0 or args.holdout_fold >= 0:
        all_cls = sorted({c for c in ex.features(cell_ids)[2] if c})
        if args.holdout_fold >= 0:
            perm = np.random.default_rng(12345).permutation(all_cls)
            held_cls = set(perm[args.holdout_fold :: args.holdout_nfolds])
        else:
            held_cls = set(np.random.default_rng(12345).choice(all_cls, round(args.holdout_frac * len(all_cls)), replace=False))
        held_keys = np.array([k for k, c in enumerate(g_cls) if c in held_cls], int)
        for r in tr:
            r.keys = np.where(np.isin(r.keys, held_keys), -1, r.keys)
        print(f"  holdout: {len(held_cls)} classes, {len(held_keys)} keys never trained: {sorted(held_cls)}", flush=True)
    if expr:
        for r in tr:
            r.keys = np.where((r.keys >= 0) & ~g_ok[r.keys.clip(0)], -1, r.keys)
        model = wm.ExprNet(G, hidden=args.hidden).to(wm.DEVICE)
    else:
        model = wm.WormNet(n_keys, hidden=args.hidden, tie=args.tie, rank=args.rank, core=args.core).to(wm.DEVICE)
    model.openloop = args.openloop
    n_params_tables = sum(getattr(model, n).weight.numel() for n in model.row_names())
    n_params_core = sum(p.numel() for p in model.parameters()) - n_params_tables
    print(f"{args.scheme}/{args.level} s{args.seed}: split {split_src} {len(tr)}/{len(te)}, keys {n_keys}, "
          f"params core {n_params_core} tables {n_params_tables}", flush=True)
    bg_model = None
    if args.bg_ckpt:
        bgc = torch.load(args.bg_ckpt, map_location=wm.DEVICE, weights_only=False)
        bgv = bgc["vocab"]; inv = {k: nm for nm, k in vocab.items()}
        bg_model_keymap = torch.tensor([bgv.get(inv.get(k), -1) for k in range(n_keys)], device=wm.DEVICE)
        bg_model = wm.WormNet(bgc["n_keys"], hidden=bgc["args"].get("hidden", 128), core=bgc["args"].get("core", "gru")).to(wm.DEVICE)
        bg_model.load_state_dict(bgc["state"]); bg_model.openloop = True
        for q in bg_model.parameters():
            q.requires_grad_(False)
    if args.from_ckpt:
        ck = torch.load(args.from_ckpt, map_location=wm.DEVICE, weights_only=False)  # our own checkpoint
        assert ck["n_keys"] == n_keys, (ck["n_keys"], n_keys)
        model.load_state_dict(ck["state"])
        losses = [float("nan")]
    else:
        losses = wm.train(model, tr, args.steps, rng, lr=args.lr, tag=f"{args.scheme}/{args.level}/s{args.seed}",
                          log_every=100 if args.steps <= 500 else 500, resp_loss=args.resp_loss, resp_blind=args.resp_blind,
                          bg_model=bg_model, bg_alpha=args.bg_alpha, bg_keymap=bg_model_keymap if args.bg_ckpt else None)

    # measured train STAM (same pooling as the evaluation), used for the paper normalisation
    tr_events = wm.event_list(tr)
    train_stam = wm.pooled_stam(tr, tr_events, wm.measured_event_responses(tr, tr_events), cell_ids)

    res = {"scheme": args.scheme, "level": args.level, "seed": args.seed, "split": split_src,
           "n_keys": n_keys, "params_core": n_params_core, "params_tables": n_params_tables,
           "train_loss_last500": float(np.mean(losses[-500:])), "zero_shot": None, "few_shot": {},
           "variant": {"tie": args.tie, "rank": args.rank, "core": args.core, "lr": args.lr, "resp_loss": args.resp_loss, "fold": args.fold, "resp_blind": args.resp_blind, "eval_blind": args.eval_blind, "bg_alpha": args.bg_alpha,
                       "openloop": args.openloop, "hidden": args.hidden, "genes": args.genes, "prior": args.prior,
                       "holdout_fold": args.holdout_fold, "fs_lr": args.fs_lr, "fs_steps": args.fewshot_steps,
                       "fs_init": args.fs_init, "fs_anchor": args.fs_anchor, "from_ckpt": args.from_ckpt}}

    # zero-shot (M1-M4; M0 has no rows for unseen animals — a structural property)
    shared_keys = None
    if args.scheme != "M0":
        wm.assign_keys(te, "M1" if expr else args.scheme, args.level, vocab, rng)
        if expr:
            for r in te:
                r.keys = np.where((r.keys >= 0) & ~g_ok[r.keys.clip(0)], -1, r.keys)
        if len(held_keys) and not expr:  # never-trained rows -> mean of trained rows (identity-free fallback)
            with torch.no_grad():
                keep = torch.ones(n_keys, dtype=torch.bool)
                keep[torch.from_numpy(held_keys)] = False
                for name in model.row_names():
                    w = getattr(model, name).weight
                    w[torch.from_numpy(held_keys).to(w.device)] = w[keep.to(w.device)].mean(0)
        shared_keys = [r.keys.copy() for r in te]
        ev = wm.event_list(te)
        if args.eval_blind:
            _crop = wm.crop_batch
            def _blind(*a, **k):
                x_, m_, u_, k_ = _crop(*a, **k); x_[:, :wm.PRE] = 0; m_[:, :wm.PRE] = 0
                return x_, m_, u_, k_
            wm.crop_batch = _blind
        resp = wm.predict_events(model, te, ev)
        if args.eval_blind:
            wm.crop_batch = _crop
        res["zero_shot"] = score(te, ev, resp, cell_ids, train_stam)
        if args.fold >= 0:   # accumulators for pooling predictions across folds (responder x target over cell_ids)
            colx = {cc: k for k, cc in enumerate(cell_ids)}
            acc = np.zeros((len(cell_ids), len(cell_ids))); cnt = np.zeros_like(acc)
            for (ri, _, j), rv in zip(ev, resp):
                tj = colx.get(te[ri].ids[j])
                if tj is None:
                    continue
                for i_, nm in enumerate(te[ri].ids):
                    ci_ = colx.get(nm)
                    if ci_ is not None and np.isfinite(rv[i_]):
                        acc[ci_, tj] += rv[i_]; cnt[ci_, tj] += 1
            OUT.mkdir(parents=True, exist_ok=True)
            np.savez(OUT / f"{args.scheme}_{args.level}_s{args.seed}_{args.tag}_acc.npz", acc=acc, cnt=cnt)
        print(f"  zero-shot: {res['zero_shot']}", flush=True)
        if args.score_auth:  # same model STAM, scored against the authors' estimator on our test animals
            TA = np.load(p0.OUT / "targets_test.npz")
            S = wm.pooled_stam(te, ev, resp, cell_ids)
            cc = lambda a, b, m=None: float(met.nan_corr(a if m is None else np.where(m, a, np.nan),
                                                          b if m is None else np.where(m, b, np.nan))[0])
            res["auth"] = {"r": cc(S, TA["T_auth_ours"]), "rho": cc(TA["T_auth_train"], TA["T_auth_ours"])}
            if len(held_keys):
                hn = {vnames[k] for k in held_keys}
                h = np.array([nm in hn for nm in cell_ids]); touch = h[:, None] | h[None, :]
                res["auth"].update(held_r=cc(S, TA["T_auth_ours"], touch), held_rho=cc(TA["T_auth_train"], TA["T_auth_ours"], touch),
                                   seen_r=cc(S, TA["T_auth_ours"], ~touch))
            print(f"  authors' estimator: {res['auth']}", flush=True)
        if len(held_keys):
            held_names = {vnames[k] for k in held_keys}
            h = np.array([nm in held_names for nm in cell_ids])
            touch = h[:, None] | h[None, :]
            res["zero_shot_heldout_entries"] = score(te, ev, resp, cell_ids, train_stam, entry_mask=touch)
            res["zero_shot_seen_entries"] = score(te, ev, resp, cell_ids, train_stam, entry_mask=~touch)
            res["holdout_classes"] = sorted({g_cls[k] for k in held_keys})
            print(f"  held-out entries: {res['zero_shot_heldout_entries']}", flush=True)

    for n_cal in args.n_cal:
        fs_model, fs_recs, eval_range = few_shot(model, args.scheme, te, n_cal, args.fewshot_steps, rng, shared_keys,
                                                 lr=args.fs_lr, init=args.fs_init, anchor=args.fs_anchor)
        ev = wm.event_list(fs_recs, frame_range=eval_range)
        out = {"few_shot": score(fs_recs, ev, wm.predict_events(fs_model, fs_recs, ev), cell_ids, train_stam)}
        if args.scheme != "M0":  # zero-shot on the SAME eval events, for the P1 comparison
            zs_recs = [copy.copy(r) for r in te]
            for r, k in zip(zs_recs, shared_keys):
                r.keys = k
            out["zero_shot_same_events"] = score(zs_recs, ev, wm.predict_events(model, zs_recs, ev), cell_ids, train_stam)
        # subset: eval events whose target column was already stimulated in that animal's calibration
        # segment (the only events where a per-animal stim row has been fitted to data)
        cal_t = {(i, j) for i, r in enumerate(fs_recs) for t, j in zip(*np.where(r.u == 1)) if t < eval_range[i][0]}
        ev_seen = [e for e in ev if (e[0], e[2]) in cal_t]
        if len(ev_seen) >= 20:
            out["seen_targets"] = score(fs_recs, ev_seen, wm.predict_events(fs_model, fs_recs, ev_seen), cell_ids, train_stam)
            if args.scheme != "M0":
                out["seen_targets_zero_shot"] = score(zs_recs, ev_seen, wm.predict_events(model, zs_recs, ev_seen), cell_ids, train_stam)
        res["few_shot"][n_cal] = out
        print(f"  n_cal={n_cal}: {json.dumps(out)}", flush=True)

    res["seconds"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    json.dump(res, open(OUT / f"{args.scheme}_{args.level}_s{args.seed}{tag}.json", "w"), indent=1)
    if not args.from_ckpt:
        torch.save({"state": model.state_dict(), "vocab": vocab, "n_keys": n_keys, "args": vars(args)},
                   OUT / f"{args.scheme}_{args.level}_s{args.seed}{tag}.pt")
    print(f"DONE {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
