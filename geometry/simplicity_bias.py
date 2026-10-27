"""Geometry test 2 (exploratory): simplicity bias in the worm connectome (Johnston et al. PNAS 2022 framing).

Prediction (algorithmic simplicity bias): across motif types, over-representation relative to a null rises with motif symmetry
(log |Aut|; K ~ log(n!/|Aut|)), beyond edge count and mutual (reciprocal) dyads.
Three nulls, each preserving every neuron's in- and out-degree (double-edge swaps):
  DEG     plain degree-preserving rewiring                                   -> is there a simplicity bias at all?
  LEN     swaps accepted only if both new edges keep their wire length (±15%; canonical 3D positions)
                                                                              -> is it geometry / wiring economy?
  RECIP   reciprocity-preserving (Milo-style): mutual dyads swapped among themselves, one-way edges among themselves;
          each neuron keeps its in/out degree AND its number of mutual partners (the Bénichou et al. 2024 bar)
  MIRROR  left/right mirror-paired edges rewired jointly (bilateral symmetry preserved exactly)
                                                                              -> is it duplication of identical parts
                                                                                 (driver 2) rather than simplicity (driver 3)?
Graphs: chemical synapses (directed, 300 neurons; 3- and 4-node directed census) and the nerve-ring contact graph (adult;
undirected 4-node census, which has 6 motif types: descriptive only).
    python geometry/simplicity_bias.py [--null 100] -> geometry/results/simplicity_bias.json
"""
import argparse, json, os, pathlib, pickle, re, sys

import igraph as ig
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import worm_data as wd  # noqa: E402
import wormneuroatlas as wa  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--null", type=int, default=100)
ap.add_argument("--nulls", default="DEG,LEN,MIRROR", help="comma list from DEG, LEN, MIRROR, RECIP")
ap.add_argument("--out", default="simplicity_bias.json"); ap.add_argument("--skip-contact", action="store_true")
args = ap.parse_args()
rng = np.random.default_rng(0)
AD = wd.REPO / "anatomical_data"
ids = [str(x) for x in pickle.load(open(AD / "cell_ids.pkl", "rb"))]; ix = {n: i for i, n in enumerate(ids)}
chem = np.asarray(pickle.load(open(AD / "chemical.pkl", "rb")), float) > 0; np.fill_diagonal(chem, False)
L = open(os.path.join(os.path.dirname(wa.__file__), "data", "anatlas_neuron_positions.txt")).read().splitlines()
pn = L[0].lstrip("#").split(); XYZ = np.array([[float(v) for v in l.split()] for l in L[1:] if l.strip()]) * 111.4
POS = dict(zip(pn, XYZ)); DIST = np.linalg.norm(np.array([POS[n] for n in ids])[:, None] - np.array([POS[n] for n in ids])[None], axis=2)


def mirror_of(name):
    m = re.fullmatch(r"(.+?)([LR])", name)
    if m and len(name) > 2:
        other = m.group(1) + ("R" if m.group(2) == "L" else "L")
        if other in ix:
            return ix[other]
    return ix[name]


MIR = np.array([mirror_of(n) for n in ids])


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


def swap_null(A, mode, tol=0.15):
    """Directed double-edge swaps (a->b, c->d) -> (a->d, c->b), degree-preserving; mode in {DEG, LEN, MIRROR}."""
    if mode == "RECIP":
        return recip_null(A)
    A = A.copy(); E = np.argwhere(A); n_prop = 10 * len(E)
    for _ in range(n_prop):
        i, j = rng.integers(len(E), size=2); a, b = E[i]; c, d = E[j]
        if len({a, b, c, d}) < 4 or A[a, d] or A[c, b]:
            continue
        if mode == "LEN":
            if abs(DIST[a, d] - DIST[a, b]) > tol * DIST[a, b] or abs(DIST[c, b] - DIST[c, d]) > tol * DIST[c, d]:
                continue
        if mode == "MIRROR":
            ma, mb, mc, md = MIR[a], MIR[b], MIR[c], MIR[d]
            olds = {(a, b), (c, d), (ma, mb), (mc, md)}; news = {(a, d), (c, b), (ma, md), (mc, mb)}
            if any(not A[x, y] for x, y in olds) or any(A[x, y] for x, y in news) or len(olds) != len(news) \
                    or any(x == y for x, y in news) or olds & news:
                continue
            dout, din = {}, {}                       # exact degree bookkeeping (self-mirrored neurons can collide)
            for x, y in olds:
                dout[x] = dout.get(x, 0) - 1; din[y] = din.get(y, 0) - 1
            for x, y in news:
                dout[x] = dout.get(x, 0) + 1; din[y] = din.get(y, 0) + 1
            if any(dout.values()) or any(din.values()):
                continue
            for x, y in olds:
                A[x, y] = False
            for x, y in news:
                A[x, y] = True
            E = np.argwhere(A)   # edge list changed in 4 places
            continue
        A[a, b] = A[c, d] = False; A[a, d] = A[c, b] = True; E[i] = (a, d); E[j] = (c, b)
    return A


def census(A, size, directed=True):
    e = np.argwhere(A) if directed else np.argwhere(np.triu(A, 1))
    return np.array(ig.Graph(n=len(A), edges=e.tolist(), directed=directed).motifs_randesu(size=size), float)


def pspear(y, x, Z):
    ry, rx = rankdata(y), rankdata(x); RZ = np.column_stack([np.ones(len(y))] + [rankdata(z) for z in Z])
    e = lambda v: v - RZ @ np.linalg.lstsq(RZ, v, rcond=None)[0]
    return float(np.corrcoef(e(ry), e(rx))[0, 1])


def motif_props(size, directed, ncls):
    aut, edges, mutual = [], [], []
    for c in range(ncls):
        g = ig.Graph.Isoclass(size, c, directed=directed); aut.append(g.count_automorphisms()); edges.append(g.ecount())
        mutual.append(sum(1 for e in g.es if directed and g.are_adjacent(e.target, e.source)) / 2 if directed else 0)
    return np.log(np.array(aut, float)), np.array(edges, float), np.array(mutual, float)


def analyse(obs, nulls, size, directed):
    la, ed, mu_d = motif_props(size, directed, len(obs)); out = {}
    for nn, Nl in nulls.items():
        Nl = np.array(Nl); mu, sd = np.nanmean(Nl, 0), np.nanstd(Nl, 0)
        ok = np.isfinite(obs) & (mu > 0) & (obs > 0)
        enr = np.log2(obs[ok] / mu[ok])
        r = {"n_motifs": int(ok.sum()), "rho_enrich_logAut": float(spearmanr(enr, la[ok])[0]),
             "partial|edges": pspear(enr, la[ok], [ed[ok]]),
             "partial|edges,mutual": pspear(enr, la[ok], [ed[ok], mu_d[ok]]) if directed else None}
        bs = []
        k = np.arange(ok.sum())
        for _ in range(500):
            b = rng.choice(k, len(k))
            if len(np.unique(la[ok][b])) > 2:
                bs.append(pspear(enr[b], la[ok][b], [ed[ok][b]] + ([mu_d[ok][b]] if directed else [])))
        r["partial_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        r["per_motif"] = [{"cls": int(c), "logAut": float(la[c]), "edges": int(ed[c]), "mutual": float(mu_d[c]),
                           "log2_enrich": float(np.log2(obs[c] / mu[c]))} for c in np.where(ok)[0]]
        out[nn] = r
    return out


res = {"mirror_pairs": int((MIR != np.arange(len(ids))).sum() // 2)}
A = chem
for size in (3, 4):
    obs = census(A, size); nulls = {k: [] for k in args.nulls.split(",")}
    for t in range(args.null):
        for nn in nulls:
            nulls[nn].append(census(swap_null(A, nn), size))
    res[f"chem_size{size}"] = analyse(obs, nulls, size, True)
    print("chem", size, json.dumps({nn: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if k != "per_motif"}
                                   for nn, o in res[f"chem_size{size}"].items()}), flush=True)
    json.dump(res, open(ROOT / "geometry" / "results" / args.out, "w"), indent=1)

if args.skip_contact:
    print("done"); sys.exit()
# contact graph, undirected 4-node census vs DEG null only (descriptive: 6 motif types)
XL = ROOT / "data" / "connectome" / "nerve_ring_neighbors" / "Adult and L4 nerve ring neighbors.xlsx"
d = pd.read_excel(XL, sheet_name="adult nerve ring neighbors", header=None)
names = [str(x).strip() for x in d.iloc[0, 1:]]; rows = [str(x).strip() for x in d.iloc[1:, 0]]
M = d.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
keep = [i for i, n in enumerate(rows) if n != "nan" and i < len(names) and names[i] == n]
CON = (M[np.ix_(keep, keep)] + M[np.ix_(keep, keep)].T) > 0; np.fill_diagonal(CON, False)
G = ig.Graph(n=len(CON), edges=np.argwhere(np.triu(CON, 1)).tolist())
obs = np.array(G.motifs_randesu(size=4), float); nl = []
for _ in range(args.null):
    Gr = G.copy(); Gr.rewire(n=10 * G.ecount(), mode="simple"); nl.append(Gr.motifs_randesu(size=4))
res["contact_size4"] = analyse(obs, {"DEG": nl}, 4, False)
print("contact 4", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["contact_size4"]["DEG"].items() if k != "per_motif"}))
json.dump(res, open(ROOT / "geometry" / "results" / args.out, "w"), indent=1)
print("done")
