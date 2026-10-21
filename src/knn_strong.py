"""Does a strength-filtered connectome embedding close the expression ≈ connectome tie? (Currier & Clandinin 2025 motivation)
Variants fixed before running: all synapses (as in connectome.py); weights < 3 synapses dropped; each neuron's top-5
in/out partners only. Same PCA-32 pipeline, kNN k=40, T=5, same folds; scored on held-out entries (authors' estimator).
    python knn_strong.py -> results/knn_strong.json
"""
import json, pickle

import numpy as np

import connectome as cx
import knn_prior as kp

ids = [str(c) for c in pickle.load(open(cx.AD / "cell_ids.pkl", "rb"))]
chem = np.asarray(pickle.load(open(cx.AD / "chemical.pkl", "rb")), float)
gap = np.asarray(pickle.load(open(cx.AD / "gap.pkl", "rb")), float)


def topk(M, k):
    out = np.zeros_like(M)
    for i in range(len(M)):
        j = np.argsort(-M[i])[:k]
        out[i, j] = M[i, j]
    return out


def emb(c, g):
    X = np.concatenate([np.log1p(c), np.log1p(c.T), np.log1p(g)], 1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    F = U[:, :32] * S[:32]
    idx = {c: i for i, c in enumerate(ids)}
    return np.stack([F[idx[n]] for n in kp.cell])


V = {"all": emb(chem, gap), "ge3": emb(np.where(chem >= 3, chem, 0), np.where(gap >= 3, gap, 0)),
     "top5": emb(np.maximum(topk(chem, 5), topk(chem.T, 5).T), np.maximum(topk(gap, 5), topk(gap, 5).T))}
zn = lambda G: G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
out = {}
for name, G in V.items():
    for label, F in ((f"connectome_{name}", G), (f"both_{name}", np.concatenate([zn(kp.Ge), zn(G)], 1))):
        sc = []
        for f in range(5):
            held = set(np.random.default_rng(12345).permutation(kp.all_cls)[f::5])
            h = np.array([c in held for c in kp.cls_of])
            sc.append(kp.corr(kp.predict(F, h, 40), kp.T, h[:, None] | h[None, :]))
        out[label] = {"mean": float(np.mean(sc)), "per_fold": sc}
        print(f"{label:16s} {np.mean(sc):.3f}  {np.round(sc, 3)}", flush=True)
json.dump(out, open("../results/knn_strong.json", "w"), indent=1)
