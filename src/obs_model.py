"""OBS->CAUSAL O4: shared-by-name model trained ONLY on Atanas spontaneous activity, then asked to
predict Randi optogenetic responses by activity injection.
    python obs_model.py --seed 0 [--core lineardiag] [--hidden 64] [--dur 4] [--amp 1]
-> results/obs_model/<core>_K<hidden>_s<seed>.json (+ .pt)
"""
import argparse, json, os, pathlib, sys

import numpy as np
import torch

import atanas_data as ad
import phase0_ceiling as p0
import train_worm as tw
import worm_data as wd
import worm_models as wm

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--core", default="lineardiag")
ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--durs", type=int, nargs="+", default=[4, 1, 10]); ap.add_argument("--amp", type=float, default=1.0)
ap.add_argument("--shuffle", action="store_true", help="control: names permuted independently per Atanas animal")
ap.add_argument("--zscore-randi", action="store_true", help="z-score Randi traces per neuron (matters for nonlinear cores)")
args = ap.parse_args()
torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
wm.DEVICE = os.environ.get("WORM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(int(os.environ.get("WORM_THREADS", "4")))
OUT = pathlib.Path(__file__).resolve().parents[1].joinpath("results/obs_model").expanduser(); OUT.mkdir(parents=True, exist_ok=True)
tag = f"{args.core}_K{args.hidden}{'_shuf' if args.shuffle else ''}{'_z' if args.zscore_randi else ''}_s{args.seed}"

# train on Atanas only (no stimulation anywhere)
at = [wm.Rec(nm, x, np.zeros_like(x, dtype=np.int8), ids) for nm, x, ids in ad.load()]
vocab, n_keys = wm.assign_keys(at, "M2" if args.shuffle else "M1", "neuron", None, rng)
model = wm.WormNet(n_keys, hidden=args.hidden, core=args.core).to(wm.DEVICE); model.openloop = True
losses = wm.train(model, at, args.steps, rng, events=[], tag=f"obs/{tag}")

# evaluate on Randi (all 110 animals; none used in training)
cell = wd.model_cell_ids(); n = len(cell); eye = np.eye(n, dtype=bool)
TA = np.load(p0.OUT / "targets_test.npz")
c = lambda a, b: float(met.nan_corr(np.where(eye, np.nan, a), np.where(eye, np.nan, b))[0])
names = wd.recording_names(); tr_n, te_n, _ = tw.get_split(names)


@torch.no_grad()
def predict_inject(recs, events, dur, amp, chunk=64):
    max_n = max(r.x.shape[1] for r in recs); out = []
    for i in range(0, len(events), chunk):
        ev = events[i:i + chunk]
        x, m, u, keys = wm.crop_batch(recs, [(ri, t - wm.PRE) for ri, t, _ in ev], max_n)
        ix, im = torch.zeros_like(x), torch.zeros_like(m)
        for b, (_, _, j) in enumerate(ev):
            ix[b, wm.PRE:wm.PRE + dur, j] = amp; im[b, wm.PRE:wm.PRE + dur, j] = 1
        on = model.rollout(x, m, u, keys, n_teacher=wm.PRE, stim_on=False, inj_x=ix, inj_m=im)
        off = model.rollout(x, m, u, keys, n_teacher=wm.PRE, stim_on=False)
        eff = (on - off)[:, wm.PRE - 1:].sum(1)
        out += [eff[b, :recs[ri].x.shape[1]].cpu().numpy() for b, (ri, _, _) in enumerate(ev)]
    return out


res = {"tag": tag, "args": vars(args), "n_keys": n_keys, "train_loss_last500": float(np.mean(losses[-500:])), "scores": {}}
for split, nm_list, tgt in (("test", te_n, "T_auth_ours"), ("all", names, "T_auth_all")):
    recs = tw.load_recs(nm_list)
    if args.zscore_randi:
        for r in recs:
            r.x = (r.x - np.nanmean(r.x, 0)) / (np.nanstd(r.x, 0) + 1e-6)
    wm.assign_keys(recs, "M1", "neuron", vocab, rng)
    ev = [e for e in wm.event_list(recs) if recs[e[0]].keys[e[2]] >= 0]   # target must be known to the model
    for dur in args.durs:
        S = wm.pooled_stam(recs, ev, predict_inject(recs, ev, dur, args.amp), cell)
        res["scores"][f"{split}_dur{dur}"] = {"r": c(S, TA[tgt]), "n_events": len(ev)}
        print(split, dur, res["scores"][f"{split}_dur{dur}"], flush=True)
json.dump(res, open(OUT / f"{tag}.json", "w"), indent=1)
torch.save({"state": model.state_dict(), "vocab": vocab, "n_keys": n_keys, "args": vars(args)}, OUT / f"{tag}.pt")
