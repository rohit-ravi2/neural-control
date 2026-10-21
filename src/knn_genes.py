"""Phase 3c preview on the non-parametric instrument (pre-registered: k = 20, temperature 5).
Per gene group: kNN prior from that group's CeNGEN features vs 20 size-matched random gene sets.
    python knn_genes.py -> results/knn_genes.json
"""
import json

import numpy as np

import expression as ex
import knn_prior as kp  # reuses predict(), T, cls_of, folds

K = 20


def score(G):
    vals = []
    for f in range(5):
        held = set(np.random.default_rng(12345).permutation(kp.all_cls)[f::5])
        h = np.array([c in held for c in kp.cls_of])
        vals.append(kp.corr(kp.predict(G, h, K), kp.T, h[:, None] | h[None, :]))
    return float(np.mean(vals))


def feats(idx):
    _, classes, X = ex._raw()
    Z = X[idx].T
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    F = U[:, :min(32, len(idx) - 1)] * S[:min(32, len(idx) - 1)]
    ci = {c: i for i, c in enumerate(classes)}
    return np.stack([F[ci[c]] for c in kp.cls_of])


names, _, X = ex._raw()
expressed = np.where(X.var(1) > 0)[0]
out = {"all_genes_k20": score(kp.Ge)}
print("all genes (top-2000 PCA):", round(out["all_genes_k20"], 3))
for g in ex.GROUPS:
    gi = ex.gene_index(g)
    r = score(feats(gi))
    pool = np.setdiff1d(expressed, gi)
    null = [score(feats(np.sort(np.random.default_rng(s).choice(pool, len(gi), replace=False)))) for s in range(20)]
    out[g] = {"n_genes": int(len(gi)), "r": r, "null_mean": float(np.mean(null)), "null_p95": float(np.percentile(null, 95)),
              "null": null, "beats_p95": bool(r > np.percentile(null, 95))}
    print(f"{g:13s} n={len(gi):3d}  r={r:.3f}  random: {np.mean(null):.3f} (p95 {np.percentile(null, 95):.3f})  "
          f"{'BEATS' if r > np.percentile(null, 95) else 'within null'}", flush=True)
json.dump(out, open("../results/knn_genes.json", "w"), indent=1)
