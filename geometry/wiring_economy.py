"""Geometry test 1 (exploratory): wiring economy vs the causal atlas.

Chklovskii-style wire-length-optimal placement (quadratic cost sum_ij W_ij |x_i - x_j|^2; Chen, Hall & Chklovskii 2006)
fitted from the connectome, then compared with the measured layout and with the Randi causal atlas.

  A  1D anterior-posterior replication (whole animal). Free = interneurons (+ variant: interneurons + motor neurons);
     anchors = the other neurons at their canonical AP position and body-wall / pharyngeal muscles at their row position.
  B  3D head placement in µm (canonical atlas x 111 µm/unit, which matches per-animal median distances at r = 0.90).
       LOO   each neuron at the W-weighted centroid of its partners (the local optimum given everyone else)
       ANCH  interneurons free, sensory + motor neurons anchored at their positions (global quadratic optimum)
     Positive control: does the predicted layout beat degree-preserving rewired connectomes?
  (a) does predicted proximity (D_hat) reproduce the atlas's proximity structure, and which part of measured proximity
      (D) does the atlas follow: the part wiring economy explains, or the residual placement?
  (b) is the distance-surviving causal signal (atlas ~ anatomy | distance, and the 1b spontaneous co-activity partial)
      concentrated on pairs where the optimisation fails (|log D / D_hat| large) or on wire-expensive pairs?

    python geometry/wiring_economy.py [--null 200]  -> geometry/results/wiring_economy.json
"""
import argparse, json, os, pathlib, pickle, sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import near_far as nf  # noqa: E402  (per-animal median distances nf.D, model cell order nf.cell, targets nf.TA)
import worm_data as wd  # noqa: E402
import worm_models as wm  # noqa: E402
import wormneuroatlas as wa  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--null", type=int, default=200)
ap.add_argument("--skip-spont", action="store_true")
args = ap.parse_args()
rng = np.random.default_rng(0)
OUT = ROOT / "geometry" / "results"; OUT.mkdir(parents=True, exist_ok=True)
UM = 111.4   # µm per canonical atlas unit (median per-animal / canonical distance ratio over 20,182 head pairs)

# ---------------------------------------------------------------- connectome (Creamer/Cook 300 neurons) + muscles (Cook 2019)
AD = wd.REPO / "anatomical_data"
ids = [str(x) for x in pickle.load(open(AD / "cell_ids.pkl", "rb"))]; ix = {n: i for i, n in enumerate(ids)}
chem = np.asarray(pickle.load(open(AD / "chemical.pkl", "rb")), float)
gap = np.asarray(pickle.load(open(AD / "gap.pkl", "rb")), float)
W = chem + chem.T + gap; np.fill_diagonal(W, 0)        # undirected wiring weight (synapse size / gap count)
ntype = pd.read_csv(AD / "neuron_type.csv")
cook = pd.read_csv(AD / "cook_synapse_size_connectome.csv", header=None)
c_cols = cook.iloc[2, 3:].tolist(); c_rows = cook.iloc[3:, 2].tolist(); CM = cook.iloc[3:, 3:].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()


def norm_name(s):   # Cook pads ventral-cord numbers (VA01); the 300-id list does not (VA1)
    s = str(s)
    import re
    m = re.fullmatch(r"([A-Z]+)0(\d)", s)
    return m.group(1) + m.group(2) if m else s


L = open(os.path.join(os.path.dirname(wa.__file__), "data", "anatlas_neuron_positions.txt")).read().splitlines()
pn = L[0].lstrip("#").split(); XYZ = np.array([[float(v) for v in l.split()] for l in L[1:] if l.strip()])
POS = {n: XYZ[k] * UM for k, n in enumerate(pn)}
has = np.array([n in POS for n in ids]); P = np.array([POS.get(n, [np.nan] * 3) for n in ids])

# neuron category from Cook's row blocks (pharynx / sensory / inter / motor)
cat = {}
blk = None
for r in range(3, cook.shape[0]):
    if isinstance(cook.iloc[r, 0], str):
        blk = cook.iloc[r, 0]
    if isinstance(cook.iloc[r, 2], str):
        cat[norm_name(cook.iloc[r, 2])] = blk
CAT = np.array([cat.get(n, "?") for n in ids])
print("categories:", {k: int((CAT == k).sum()) for k in np.unique(CAT)}, "| positioned:", int(has.sum()), flush=True)

# muscles: AP row position (24 rows, nose -> tail) for body-wall; pharyngeal muscles spread over the pharynx
y_nose, y_tail = XYZ[:, 1].min() * UM, XYZ[:, 1].max() * UM
pharynx_y = np.array([POS[n][1] for n in ids if n in POS and cat.get(n) == "PHARYNX"])
MUS = {}
for c in c_cols:
    c = str(c)
    if "BWM" in c:
        k = int("".join(ch for ch in c if ch.isdigit())); MUS[c] = y_nose + (k - 0.5) / 24 * (y_tail - y_nose)
    elif c.startswith("pm"):
        k = int(c[2]); MUS[c] = pharynx_y.min() + (k - 0.5) / 8 * (pharynx_y.max() - pharynx_y.min())
mus_names = sorted(MUS); Wm = np.zeros((len(ids), len(mus_names)))
for a, rn in enumerate(c_rows):
    rn = norm_name(rn)
    if rn in ix:
        for b, cn in enumerate(c_cols):
            if str(cn) in MUS:
                Wm[ix[rn], mus_names.index(str(cn))] += CM[a, b]
print("neuron-muscle weight total", Wm.sum(), "| neuron-neuron", W.sum() / 2, flush=True)


def rewire(Wx, n_swaps=None):
    """Degree-preserving double-edge swap on the undirected weighted graph (weights travel with edges)."""
    iu = np.triu_indices(len(Wx), 1); e = np.where(Wx[iu] > 0)[0]
    a, b, w = iu[0][e].copy(), iu[1][e].copy(), Wx[iu][e].copy()
    E = set(zip(np.minimum(a, b), np.maximum(a, b)))
    for _ in range(n_swaps or 10 * len(a)):
        i, j = rng.integers(len(a), size=2)
        if len({a[i], b[i], a[j], b[j]}) < 4:
            continue
        n1, n2 = (min(a[i], b[j]), max(a[i], b[j])), (min(a[j], b[i]), max(a[j], b[i]))
        if n1 in E or n2 in E:
            continue
        E -= {(min(a[i], b[i]), max(a[i], b[i])), (min(a[j], b[j]), max(a[j], b[j]))}; E |= {n1, n2}
        b[i], b[j] = b[j], b[i]
    R = np.zeros_like(Wx); R[a, b] = w; R[b, a] = w
    return R


def solve(Wx, free, anchor_pos, Wmus=None, mus_pos=None):
    """Quadratic optimum for free nodes given anchors: (D_f - W_ff) x_f = W_fa x_a (+ muscle terms). Positions (n, d)."""
    f = np.where(free)[0]; a = np.where(~free & np.isfinite(anchor_pos).all(1))[0]
    Lff = -Wx[np.ix_(f, f)]; deg = Wx[np.ix_(f, a)].sum(1) + Wx[np.ix_(f, f)].sum(1)
    rhs = Wx[np.ix_(f, a)] @ anchor_pos[a]
    if Wmus is not None:
        deg = deg + Wmus[f].sum(1); rhs = rhs + Wmus[f] @ mus_pos
    Lff[np.diag_indices(len(f))] += deg
    ok = deg > 0
    X = np.full((len(Wx), anchor_pos.shape[1]), np.nan); X[~free] = anchor_pos[~free]
    sol = np.linalg.lstsq(Lff[np.ix_(ok, ok)], rhs[ok], rcond=None)[0]
    X[f[ok]] = sol
    return X


def loo(Wx, pos):
    """Each neuron at the weighted centroid of its partners (actual partner positions)."""
    ok = np.isfinite(pos).all(1); Wk = Wx * ok[None, :]; s = Wk.sum(1)
    X = np.where(s[:, None] > 0, (Wk @ np.nan_to_num(pos)) / np.maximum(s, 1e-12)[:, None], np.nan)
    return X


def pspear(y, x, Z):
    ry, rx = rankdata(y), rankdata(x); RZ = np.column_stack([np.ones(len(y))] + [rankdata(z) for z in Z])
    e = lambda v: v - RZ @ np.linalg.lstsq(RZ, v, rcond=None)[0]
    return float(np.corrcoef(e(ry), e(rx))[0, 1])


res = {"um_per_unit": UM, "n_neurons_positioned": int(has.sum())}

# ================================================================ A. 1D anterior-posterior replication
yA = P[:, [1]]; mus_y = np.array([[MUS[m]] for m in mus_names])
inter = (CAT == "INTERNEURONS") & has
for tag, free in (("inter_free", inter), ("inter+motor_free", ((CAT == "INTERNEURONS") | (CAT == "MOTOR NEURONS")) & has)):
    X = solve(W, free, yA, Wm, mus_y)
    ok = free & np.isfinite(X[:, 0])
    r = float(np.corrcoef(X[ok, 0], yA[ok, 0])[0, 1]); err = np.abs(X[ok, 0] - yA[ok, 0])
    null = []
    for _ in range(args.null // 4):
        Xn = solve(rewire(W), free, yA, Wm, mus_y); okn = free & np.isfinite(Xn[:, 0])
        null.append(np.corrcoef(Xn[okn, 0], yA[okn, 0])[0, 1])
    # body length for scale: fraction of free neurons placed within 5% of body length of their actual position
    blen = y_tail - y_nose
    res[f"A_{tag}"] = {"n_free": int(ok.sum()), "r_pred_actual": r, "frac_within_5pct_body": float(np.mean(err < 0.05 * blen)),
                       "median_err_um": float(np.median(err)), "null_r_mean": float(np.mean(null)), "null_r_p95": float(np.percentile(null, 95)),
                       "p": float((np.sum(np.array(null) >= r) + 1) / (len(null) + 1)),
                       "worst_placed": [ids[i] for i in np.where(ok)[0][np.argsort(-err)[:12]]]}
    print("A", tag, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[f"A_{tag}"].items()}), flush=True)

# ================================================================ B. 3D head placement
cell = nf.cell; cix = np.array([ix[c] for c in cell]); nC = len(cell); eye = np.eye(nC, dtype=bool)
D = nf.D.copy()                                       # per-animal median distance (µm); fallback canonical
Dcan = np.linalg.norm(P[cix][:, None] - P[cix][None], axis=2)
D = np.where(np.isfinite(D), D, Dcan)
A = np.log1p(chem[np.ix_(cix, cix)]) + np.log1p(gap[np.ix_(cix, cix)])   # [post? -> as in dist_shuffle: i <- j orientation]
A_sym = np.log1p(W[np.ix_(cix, cix)])
anchor = (CAT != "INTERNEURONS") & has


def predicted(Wx):
    X_loo = loo(Wx, P)
    X_anch = solve(Wx, (CAT == "INTERNEURONS") & has, P)
    return {"LOO": X_loo, "ANCH": X_anch}


def dhat(X):
    Y = X[cix]; return np.linalg.norm(Y[:, None] - Y[None], axis=2)


PRED = predicted(W)
mask_free = {"LOO": np.ones(nC, bool), "ANCH": (CAT[cix] == "INTERNEURONS")}
for k, X in PRED.items():
    Dh = dhat(X); m = ~eye & np.isfinite(Dh) & (mask_free[k][:, None] | mask_free[k][None, :])
    disp = np.linalg.norm(X[cix] - P[cix], axis=1)
    rho = spearmanr(Dh[m], D[m])[0]
    null = []
    for _ in range(args.null // 4):
        Dn = dhat(predicted(rewire(W))[k]); mn = m & np.isfinite(Dn)
        null.append(spearmanr(Dn[mn], D[mn])[0])
    res[f"B_{k}_placement"] = {"n_pairs": int(m.sum()), "rho_Dhat_D": float(rho), "null_mean": float(np.mean(null)),
                               "null_p95": float(np.percentile(null, 95)), "p": float((np.sum(np.array(null) >= rho) + 1) / (len(null) + 1)),
                               "median_displacement_um": float(np.nanmedian(disp[mask_free[k]])),
                               "most_displaced": [cell[i] for i in np.argsort(-np.nan_to_num(disp * mask_free[k]))[:12]]}
    print("B placement", k, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in res[f"B_{k}_placement"].items()}), flush=True)

# ---------------------------------------------------------------- (a) atlas vs measured / predicted proximity
na_ = None
wa.WormBase.assert_db_version_consistency = lambda self: None
na_ = wa.NeuroAtlas(verbose=False)
nid = [str(x) for x in na_.neuron_ids]; nix = {n: i for i, n in enumerate(nid)}
dff = na_.get_signal_propagation_map("wt").astype(float); occ = na_.get_signal_propagation_occurrence_matrix("wt")
dff = np.where(occ >= 3, dff, np.nan)
fun = np.full((nC, nC), np.nan)
for a_, ca in enumerate(cell):
    for b_, cb in enumerate(cell):
        if ca in nix and cb in nix:
            fun[a_, b_] = dff[nix[ca], nix[cb]]
shipped = pickle.load(open(wd.REPO / "data" / "measured_stams.pkl", "rb"))
TARGETS = {"T_auth_all": nf.TA["T_auth_all"], "T_ship": shipped["test"], "funatlas_wt_dFF": fun}
res["a"] = {}
for tn, T in TARGETS.items():
    for k, X in PRED.items():
        Dh = dhat(X)
        m = ~eye & np.isfinite(T) & np.isfinite(Dh) & np.isfinite(D) & (mask_free[k][:, None] | mask_free[k][None, :])
        t, d, dh, an = T[m], -D[m], -Dh[m], A[m]
        logres = np.log(D[m]) - np.log(np.maximum(Dh[m], 1e-3))
        r = {"n": int(m.sum()), "rho_T_minusD": float(spearmanr(t, d)[0]), "rho_T_minusDhat": float(spearmanr(t, dh)[0]),
             "rho_T_anat": float(spearmanr(t, an)[0]),
             "T~D|Dhat": pspear(t, d, [dh]), "T~Dhat|D": pspear(t, dh, [d]),
             "T~D|Dhat,anat": pspear(t, d, [dh, an]), "T~Dhat|D,anat": pspear(t, dh, [d, an]),
             "T~anat|D": pspear(t, an, [d]), "T~anat|D,Dhat": pspear(t, an, [d, dh]),
             "T~placement_residual(-logD/Dhat)": float(spearmanr(t, -logres)[0])}
        # near-field reproduction: proportion of the atlas-proximity effect carried by D_hat-near vs D-near
        nearD = D[m] < 10; nearH = Dh[m] < np.percentile(Dh[m], 100 * nearD.mean())
        r["frac_near_D"] = float(nearD.mean()); r["overlap_nearD_nearDhat"] = float((nearD & nearH).sum() / max(nearD.sum(), 1))
        r["T_mean_nearD"] = float(np.nanmean(t[nearD])); r["T_mean_nearDhat_only"] = float(np.nanmean(t[nearH & ~nearD]))
        r["T_mean_nearD_only"] = float(np.nanmean(t[nearD & ~nearH])); r["T_mean_far_both"] = float(np.nanmean(t[~nearD & ~nearH]))
        res["a"][f"{tn}|{k}"] = r
        print("a", tn, k, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()}), flush=True)

# ---------------------------------------------------------------- (b) is the distance-surviving signal on optimisation-failure pairs?
res["b"] = {}
for tn, T in TARGETS.items():
    for k, X in PRED.items():
        Dh = dhat(X)
        m = ~eye & np.isfinite(T) & np.isfinite(Dh) & (mask_free[k][:, None] | mask_free[k][None, :])
        t, d, an = T[m], -D[m], A[m]
        fail = np.abs(np.log(D[m]) - np.log(np.maximum(Dh[m], 1e-3)))
        cost = A_sym[m] * D[m] ** 2                        # wire cost carried by the pair (0 if unconnected)
        out = {}
        for sname, s in (("|log D/Dhat|", fail), ("near<10um_only:|log D/Dhat|", np.where(D[m] < 10, fail, np.nan))):
            ok = np.isfinite(s); q = np.nanpercentile(s, [33.3, 66.7])
            strata = {"low": ok & (s <= q[0]), "mid": ok & (s > q[0]) & (s <= q[1]), "high": ok & (s > q[1])}
            st = {}
            for nm_, sm in strata.items():
                st[nm_] = {"n": int(sm.sum()), "anat|D": pspear(t[sm], an[sm], [d[sm]])}
            boots = []
            lo, hi = np.where(strata["low"])[0], np.where(strata["high"])[0]
            for _ in range(500):
                bl, bh = rng.choice(lo, len(lo)), rng.choice(hi, len(hi))
                boots.append(pspear(t[bh], an[bh], [d[bh]]) - pspear(t[bl], an[bl], [d[bl]]))
            st["high_minus_low"] = float(st["high"]["anat|D"] - st["low"]["anat|D"]); st["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            out[sname] = st
        conn = an > 0
        q = np.percentile(cost[conn], [50])
        out["connected_pairs_by_wirecost"] = {"cheap": pspear(t[conn & (cost <= q[0])], an[conn & (cost <= q[0])], [d[conn & (cost <= q[0])]]),
                                              "expensive": pspear(t[conn & (cost > q[0])], an[conn & (cost > q[0])], [d[conn & (cost > q[0])]])}
        res["b"][f"{tn}|{k}"] = out
        print("b", tn, k, json.dumps(out), flush=True)

json.dump(res, open(OUT / "wiring_economy.json", "w"), indent=1)

# ---------------------------------------------------------------- (b2) 1b spontaneous co-activity partial, stratified by placement failure
if not args.skip_spont:
    import atanas_data as ad
    S0 = {}
    for name, x, ids_ in ad.load():
        cl = np.array([wm.neuron_class(i) if i else "" for i in ids_]); U = sorted(set(cl) - {""})
        Xc = np.stack([x[:, cl == c].mean(1) for c in U], 1); Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-6)
        Uv = np.linalg.svd(x - x.mean(0), full_matrices=False)[0][:, :3]; Xc = Xc - Uv @ (Uv.T @ Xc)
        G0 = np.corrcoef(Xc.T)
        for a_, c in enumerate(U):
            for b_, s in enumerate(U):
                if a_ != b_:
                    S0.setdefault((c, s), []).append(G0[a_, b_])
    gz = {p: float(np.mean(v)) for p, v in S0.items() if len(v) >= 5}
    ac = np.array([wm.neuron_class(n) for n in nid])
    def class_mean(M, rows, cols, rc, cc):
        o = {}
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if i != j and np.isfinite(M[i, j]) and rc[i] and cc[j]:
                    o.setdefault((rc[i], cc[j]), []).append(M[i, j])
        return {k: float(np.mean(v)) for k, v in o.items()}
    Cc = class_mean(dff, None, None, ac, ac)
    anat300 = np.log1p(na_.get_chemical_synapses().astype(float)) + np.log1p(na_.get_gap_junctions().astype(float))
    Ac = class_mean(anat300, None, None, ac, ac)
    cc_ = np.array([wm.neuron_class(c) for c in cell])
    Dc_ = class_mean(D, None, None, cc_, cc_)
    res["b2"] = {}
    for k, X in PRED.items():
        Dh = dhat(X); F = np.abs(np.log(D) - np.log(np.maximum(Dh, 1e-3))); F[~(mask_free[k][:, None] | mask_free[k][None, :])] = np.nan
        Fc = class_mean(F, None, None, cc_, cc_)
        both = [p for p in gz if p in Cc and p in Ac and p in Dc_ and p in Fc]
        y = np.array([gz[p] for p in both]); cv = np.array([Cc[p] for p in both]); av = np.array([Ac[p] for p in both])
        dv = np.array([Dc_[p] for p in both]); fv = np.array([Fc[p] for p in both])
        q = np.percentile(fv, [33.3, 66.7]); out = {"n_class_pairs": len(both), "all": pspear(y, cv, [av, dv])}
        cls_ = sorted({c for p in Cc for c in p})
        for nm_, sm in (("low", fv <= q[0]), ("mid", (fv > q[0]) & (fv <= q[1])), ("high", fv > q[1])):
            r0 = pspear(y[sm], cv[sm], [av[sm], dv[sm]]); null = []
            for _ in range(200):
                mp = dict(zip(cls_, rng.permutation(cls_)))
                v = np.array([Cc.get((mp[a_], mp[b_]), np.nan) for a_, b_ in np.array(both, dtype=object)[sm]]); okv = np.isfinite(v)
                null.append(pspear(y[sm][okv], v[okv], [av[sm][okv], dv[sm][okv]]))
            out[nm_] = {"n": int(sm.sum()), "causal|anat,dist": r0, "z": float((r0 - np.mean(null)) / np.std(null)),
                        "p": float((np.sum(np.abs(null) >= abs(r0)) + 1) / 201)}
        res["b2"][k] = out
        print("b2", k, json.dumps(out), flush=True)
    json.dump(res, open(OUT / "wiring_economy.json", "w"), indent=1)
print("done", flush=True)
