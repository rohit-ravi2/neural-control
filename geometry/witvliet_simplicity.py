"""Test 2 replication across 8 independent EM connectomes (Witvliet 2020: L1 x4, L2, L3, adult x2; chemical synapses, head).
Same instrument as simplicity_bias.py: directed 4-node motif enrichment vs log|Aut|, partial on edges + mutual dyads, against
DEG (degree-preserving) and RECIP (degree + mutual-partner preserving) nulls. Question: does the symmetric-motif excess
replicate across animals and development, and is it absorbed by reciprocity in each?
    python geometry/witvliet_simplicity.py [--null 40] -> geometry/results/witvliet_simplicity.json
"""
import argparse, json, pathlib

import igraph as ig
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ap = argparse.ArgumentParser(); ap.add_argument("--null", type=int, default=40); ap.add_argument("--only", default="")
ap.add_argument("--stable-split", action="store_true", help="adult graphs split into stable (in >= 6 of 8 datasets) vs variable (<= 2) edges")
ap.add_argument("--control-split", action="store_true", help="adult graphs: strongest-N edges by synapse count, and random-N, N = stable count")
ap.add_argument("--dissociate", action="store_true", help="adult graphs: stable-not-strongest vs strongest-not-stable edges")
ap.add_argument("--complexity", choices=["aut", "bdm"], default="aut",
                help="motif simplicity: log|Aut| or -BDM (min over the 24 relabelings of the 4x4 adjacency; pybdm CTM)")
args = ap.parse_args()
rng = np.random.default_rng(3)
ROOT = pathlib.Path(__file__).resolve().parent.parent
E = pd.read_csv(ROOT / "data" / "connectome" / "witvliet2020" / "witvliet_combined_edges.csv")
E = E[E.type == "chemical"]


def deg_null(A):
    A = A.copy(); S = np.argwhere(A)
    for _ in range(10 * len(S)):
        i, j = rng.integers(len(S), size=2); a, b = S[i]; c, d = S[j]
        if len({a, b, c, d}) == 4 and not A[a, d] and not A[c, b]:
            A[a, b] = A[c, d] = False; A[a, d] = A[c, b] = True; S[i] = (a, d); S[j] = (c, b)
    return A


def recip_null(A):
    A = A.copy(); M = np.argwhere(np.triu(A & A.T, 1)); S = np.argwhere(A & ~A.T)
    free = lambda x, y: x != y and not A[x, y] and not A[y, x]
    for _ in range(10 * len(S)):
        i, j = rng.integers(len(S), size=2); a, b = S[i]; c, d = S[j]
        if len({a, b, c, d}) == 4 and free(a, d) and free(c, b):
            A[a, b] = A[c, d] = False; A[a, d] = A[c, b] = True; S[i] = (a, d); S[j] = (c, b)
    for _ in range(10 * len(M)):
        i, j = rng.integers(len(M), size=2); a, b = M[i]; c, d = M[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) == 4 and free(a, d) and free(c, b):
            A[a, b] = A[b, a] = A[c, d] = A[d, c] = False; A[a, d] = A[d, a] = A[c, b] = A[b, c] = True
            M[i] = (a, d); M[j] = (c, b)
    return A


census = lambda A: np.array(ig.Graph(n=len(A), edges=np.argwhere(A).tolist(), directed=True).motifs_randesu(size=4), float)
PROPS = []
for c in range(218):
    g = ig.Graph.Isoclass(4, c, directed=True)
    PROPS.append((np.log(g.count_automorphisms()), g.ecount(), sum(1 for e in g.es if g.are_adjacent(e.target, e.source)) / 2))
LA, ED, MU = (np.array(x, float) for x in zip(*PROPS))
if args.complexity == "bdm":   # replace symmetry by an algorithmic-complexity estimate (simplicity = -K)
    import itertools
    from pybdm import BDM
    bdm = BDM(ndim=2); KB = []
    for c in range(218):
        Adj = np.array(ig.Graph.Isoclass(4, c, directed=True).get_adjacency().data, int)
        KB.append(min(bdm.bdm(Adj[np.ix_(p, p)]) for p in itertools.permutations(range(4))))
    LA = -np.array(KB)
    print("BDM range", float(min(KB)), float(max(KB)), "| rho(-K, log|Aut|) =",
          round(float(np.corrcoef(rankdata(LA), rankdata([x[0] for x in PROPS]))[0, 1]), 3), flush=True)


def pspear(y, x, Z):
    ry, rx = rankdata(y), rankdata(x); RZ = np.column_stack([np.ones(len(y))] + [rankdata(z) for z in Z])
    e = lambda v: v - RZ @ np.linalg.lstsq(RZ, v, rcond=None)[0]
    return float(np.corrcoef(e(ry), e(rx))[0, 1])


SUF = "_bdm" if args.complexity == "bdm" else ""
out_path = ROOT / "geometry" / "results" / ("witvliet_stable_split.json" if args.stable_split else
                                             "witvliet_control_split.json" if args.control_split else
                                             "witvliet_dissociate.json" if args.dissociate else "witvliet_simplicity.json").replace(".json", SUF + ".json")
if args.stable_split:   # edge stability = number of the 8 datasets containing the (pre, post) chemical edge
    cnt = E.drop_duplicates(["pre", "post", "source_file"]).groupby(["pre", "post"]).size()
    parts = []
    for src in ("witvliet_2020_7.xlsx", "witvliet_2020_8.xlsx"):
        g = E[E.source_file == src]; st = np.array([cnt[(a, b)] for a, b in zip(g.pre, g.post)])
        parts.append(g[st >= 6].assign(source_file=src.replace(".xlsx", "_stable")))
        parts.append(g[st <= 2].assign(source_file=src.replace(".xlsx", "_variable")))
    E = pd.concat(parts)
if args.dissociate:
    cnt = E.drop_duplicates(["pre", "post", "source_file"]).groupby(["pre", "post"]).size(); parts = []
    for src in ("witvliet_2020_7.xlsx", "witvliet_2020_8.xlsx"):
        g = E[E.source_file == src].groupby(["pre", "post"], as_index=False)["synapses"].sum()
        st = np.array([cnt[(a, b)] for a, b in zip(g.pre, g.post)]) >= 6; nst = int(st.sum())
        strong = np.zeros(len(g), bool); strong[np.argsort(-g.synapses.to_numpy(), kind="stable")[:nst]] = True
        print(src, "stable", nst, "overlap with strongest-N", int((st & strong).sum()), flush=True)
        parts.append(g[st & ~strong].assign(source_file=src.replace(".xlsx", "_stable_not_strong"), type="chemical"))
        parts.append(g[strong & ~st].assign(source_file=src.replace(".xlsx", "_strong_not_stable"), type="chemical"))
    E = pd.concat(parts)
if args.control_split:
    cnt = E.drop_duplicates(["pre", "post", "source_file"]).groupby(["pre", "post"]).size(); parts = []
    for src in ("witvliet_2020_7.xlsx", "witvliet_2020_8.xlsx"):
        g = E[E.source_file == src].groupby(["pre", "post"], as_index=False)["synapses"].sum()
        nst = int((np.array([cnt[(a, b)] for a, b in zip(g.pre, g.post)]) >= 6).sum())
        parts.append(g.nlargest(nst, "synapses").assign(source_file=src.replace(".xlsx", f"_strongest{nst}"), type="chemical"))
        for r in range(2):
            parts.append(g.sample(nst, random_state=r).assign(source_file=src.replace(".xlsx", f"_random{nst}_r{r}"), type="chemical"))
    E = pd.concat(parts)
res = json.load(open(out_path)) if out_path.exists() else {}
for src, g in E.groupby("source_file"):
    if args.only and args.only not in src:
        continue
    names = sorted(set(g.pre) | set(g.post)); k = {n: i for i, n in enumerate(names)}
    A = np.zeros((len(names), len(names)), bool)
    for a, b in zip(g.pre, g.post):
        if a != b:
            A[k[a], k[b]] = True
    obs = census(A); r = {"n": len(names), "edges": int(A.sum()), "mutual_dyads": int((A & A.T).sum() // 2)}
    for nn, fn in (("DEG", deg_null), ("RECIP", recip_null)):
        Nl = np.array([census(fn(A)) for _ in range(args.null)]); mu = np.nanmean(Nl, 0)
        ok = np.isfinite(obs) & (mu > 0) & (obs > 0)
        enr = np.log2(obs[ok] / mu[ok]); kk = np.arange(ok.sum()); bs = []
        for _ in range(500):
            b = rng.choice(kk, len(kk)); bs.append(pspear(enr[b], LA[ok][b], [ED[ok][b], MU[ok][b]]))
        r[nn] = {"n_motifs": int(ok.sum()), "partial|edges,mutual": pspear(enr, LA[ok], [ED[ok], MU[ok]]),
                 "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res[src] = r
    print(src, json.dumps(r, default=lambda x: round(x, 3)), flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
print("done")
