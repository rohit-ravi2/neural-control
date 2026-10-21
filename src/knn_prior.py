"""Non-parametric prior baseline for never-trained neuron classes (CPU, no model).
For a held-out neuron i, predict its STAM row (and column) as the similarity-weighted mean of the MEASURED training-animal
rows (columns) of the k most similar non-held neurons; similarity = cosine on CeNGEN or connectome features (same
features as M3h/Mch). Scored against the authors' estimator on our test animals, same folds and masks as
analyze_heldout.py. Entries where both neurons are held out are predicted by the row rule.
    python knn_prior.py -> results/knn_prior.json
"""
import json, sys

import numpy as np

import connectome as cx
import expression as ex
import phase0_ceiling as p0
import worm_data as wd

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import metrics as met  # noqa: E402

cell = wd.model_cell_ids(); n = len(cell)
TA = np.load(p0.OUT / "targets_test.npz"); T, TR = TA["T_auth_ours"], TA["T_auth_train"]
Ge, _, cls_of = ex.features(cell)
Gc, _ = cx.features(cell)
all_cls = sorted({c for c in cls_of if c})
corr = lambda a, b, m: float(met.nan_corr(np.where(m, a, np.nan), np.where(m, b, np.nan))[0])


def predict(G, h, k):
    Z = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    S = Z @ Z.T
    P = np.full((n, n), np.nan)
    src = np.where(~h)[0]
    for i in np.where(h)[0]:
        nb = src[np.argsort(-S[i, src])[:k]]
        w = np.exp(S[i, nb] * 5); w /= w.sum()
        row = np.nansum(TR[nb] * w[:, None], 0) / np.nansum(np.isfinite(TR[nb]) * w[:, None], 0)
        col = np.nansum(TR[:, nb] * w[None, :], 1) / np.nansum(np.isfinite(TR[:, nb]) * w[None, :], 1)
        P[i, :] = row
        P[~h, i] = col[~h]
    return P


if __name__ == "__main__":
    out = {}
    for name, G in (("knn_cengen", Ge), ("knn_connectome", Gc)):
        for k in (1, 3, 5, 10):
            rs = []
            for f in range(5):
                held = set(np.random.default_rng(12345).permutation(all_cls)[f::5])
                h = np.array([c in held for c in cls_of])
                P = predict(G, h, k)
                m = {"rows": h[:, None] & ~h[None, :], "cols": ~h[:, None] & h[None, :], "held": h[:, None] | h[None, :]}
                rs.append({kk: corr(P, T, mm) for kk, mm in m.items()})
            out[f"{name}_k{k}"] = {kk: [float(np.mean([r[kk] for r in rs])), float(np.std([r[kk] for r in rs], ddof=1))] for kk in rs[0]}
            print(f"{name} k={k}: " + "  ".join(f"{kk} {v[0]:.3f}±{v[1]:.3f}" for kk, v in out[f'{name}_k{k}'].items()), flush=True)
    json.dump(out, open("../results/knn_prior.json", "w"), indent=1)

    # controls: random neighbours (same k, 20 draws) and the uniform population mean (k = all non-held neurons)
    rng = np.random.default_rng(0)
    for k in (1, 3, 5, 10, 20, 40, 1000):
        for name, G in (("knn_cengen", Ge), ("knn_connectome", Gc), ("knn_random", None)):
            if name != "knn_random" and k not in (20, 40, 1000):
                continue
            rs = []
            for f in range(5):
                held = set(np.random.default_rng(12345).permutation(all_cls)[f::5])
                h = np.array([c in held for c in cls_of])
                m = {"rows": h[:, None] & ~h[None, :], "cols": ~h[:, None] & h[None, :], "held": h[:, None] | h[None, :]}
                for _ in range(20 if name == "knn_random" and k < 1000 else 1):
                    Gr = rng.normal(size=(n, 32)) if name == "knn_random" else G
                    P = predict(Gr, h, min(k, (~h).sum()))
                    rs.append({kk: corr(P, T, mm) for kk, mm in m.items()})
            key = f"{name}_k{k}"
            out[key] = {kk: [float(np.mean([r[kk] for r in rs])), float(np.std([r[kk] for r in rs], ddof=1))] for kk in rs[0]}
            print(f"{key}: " + "  ".join(f"{kk} {v[0]:.3f}±{v[1]:.3f}" for kk, v in out[key].items()), flush=True)
    json.dump(out, open("../results/knn_prior.json", "w"), indent=1)

    # combined similarity: features z-normalised per source and concatenated (equal weight)
    zn = lambda G: G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    Gb = np.concatenate([zn(Ge), zn(Gc)], 1)
    for k in (10, 40, 1000):
        rs = []
        for f in range(5):
            held = set(np.random.default_rng(12345).permutation(all_cls)[f::5])
            h = np.array([c in held for c in cls_of])
            m = {"rows": h[:, None] & ~h[None, :], "cols": ~h[:, None] & h[None, :], "held": h[:, None] | h[None, :]}
            rs.append({kk: corr(predict(Gb, h, min(k, (~h).sum())), T, mm) for kk, mm in m.items()})
        out[f"knn_both_k{k}"] = {kk: [float(np.mean([r[kk] for r in rs])), float(np.std([r[kk] for r in rs], ddof=1))] for kk in rs[0]}
        out[f"knn_both_k{k}"]["per_fold_held"] = [r["held"] for r in rs]
        print(f"knn_both_k{k}: " + "  ".join(f"{kk} {v[0]:.3f}±{v[1]:.3f}" for kk, v in out[f'knn_both_k{k}'].items() if kk != "per_fold_held"), flush=True)
    json.dump(out, open("../results/knn_prior.json", "w"), indent=1)
