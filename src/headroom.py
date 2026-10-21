"""HEADROOM: (a) noise ceiling of the pooled test STAM; (b) per-animal event-level pot.
    python headroom.py -> results/headroom.json
"""
import collections, json, sys, warnings

import numpy as np

import phase0_ceiling as p0
import train_worm as tw
import worm_data as wd
import worm_models as wm

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402

cell = wd.model_cell_ids(); n = len(cell); eye = np.eye(n, dtype=bool); col = {c: i for i, c in enumerate(cell)}
c = lambda a, b: float(met.nan_corr(np.where(eye, np.nan, a), np.where(eye, np.nan, b))[0])
names = wd.recording_names(); tr_n, te_n, _ = tw.get_split(names)
TA = np.load(p0.OUT / "targets_test.npz"); rng = np.random.default_rng(0)
out = {}
rh = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _ in range(20):
        p = rng.permutation(te_n); rh.append(c(p0.authors_stams(sorted(p[:15]), cell), p0.authors_stams(sorted(p[15:]), cell)))
r_half = float(np.mean(rh)); r_tt = 2 * r_half / (1 + r_half)
out["a"] = {"r_half_mean": r_half, "r_half_sd": float(np.std(rh)), "r_tt_30": r_tt, "noiseless_ceiling": float(np.sqrt(max(r_tt, 0))),
            "rho_train80_to_test30": c(TA["T_auth_train"], TA["T_auth_ours"])}
for k, v in {"Creamer CC": 0.105, "Creamer FC": 0.153, "M1 neuron": 0.223, "M1 class": 0.248}.items():
    out["a"][f"{k} fraction of ceiling"] = v / out["a"]["noiseless_ceiling"]
print("(a)", json.dumps(out["a"], indent=1), flush=True)

# (b) per-animal event level (test animals), named targets, our v2-cleaned traces, baseline-subtracted post mean
recs = tw.load_recs(te_n); TR = TA["T_auth_train"]
rows = collections.defaultdict(list)
for a, r in enumerate(recs):
    ev = []
    for t, j in zip(*np.where(r.u == 1)):
        if r.ids[j] in col and t - wm.PRE >= 0 and t + wm.POST < r.x.shape[0]:
            clip = r.x[t - wm.PRE:t + wm.POST]
            with np.errstate(all="ignore"):
                v = np.nanmean(clip[wm.PRE:], 0) - np.nanmean(clip[:wm.PRE], 0)
            ev.append((int(t), col[r.ids[j]], {col[r.ids[i]]: v[i] for i in range(len(v)) if r.ids[i] in col and i != j and np.isfinite(v[i])}))
    for k, (t, j, resp) in enumerate(ev):
        def rr(pred):
            ks = [i for i in resp if i in pred and np.isfinite(pred[i])]
            if len(ks) < 10:
                return None
            x, y = np.array([resp[i] for i in ks]), np.array([pred[i] for i in ks])
            return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
        rows["pooled train STAM (shared map)"].append(rr({i: TR[i, j] for i in range(n)}))
        same = [e for kk, e in enumerate(ev) if kk != k and e[1] == j and abs(e[0] - t) >= 120]
        if same:
            rows["same animal, same target repeat (individual ceiling proxy)"].append(rr(same[0][2]))
            rows["pooled train STAM, on events that have a repeat"].append(rr({i: TR[i, j] for i in range(n)}))
out["b"] = {k: {"mean": float(np.nanmean([x for x in v if x is not None])), "n": int(sum(x is not None for x in v))} for k, v in rows.items()}
print("(b)", json.dumps(out["b"], indent=1))
json.dump(out, open("../results/headroom.json", "w"), indent=1)
