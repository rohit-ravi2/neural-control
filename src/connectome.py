"""Connectome features: the competitor prior to CeNGEN expression for never-trained neurons (Phase 3b).

Source: the anatomical connectome shipped with Creamer et al. (work/Creamer_LDS_2024/anatomical_data:
chemical.pkl, gap.pkl, 300 x 300, cell_ids.pkl). Per neuron: [log1p chemical out-row, log1p chemical in-column,
log1p gap row] (900 dims) -> z-score per column -> PCA to 32 dims, fitted on the 300 neurons alone. Fixed before
any worm model sees it. Same interface as expression.features().
"""
from __future__ import annotations

import functools
import pickle

import numpy as np

import worm_data as wd

AD = wd.REPO / "anatomical_data"


@functools.lru_cache
def neuron_matrix(n_pc: int = 32):
    ids = [str(c) for c in pickle.load(open(AD / "cell_ids.pkl", "rb"))]
    chem = np.log1p(np.asarray(pickle.load(open(AD / "chemical.pkl", "rb")), float))
    gap = np.log1p(np.asarray(pickle.load(open(AD / "gap.pkl", "rb")), float))
    X = np.concatenate([chem, chem.T, gap], 1)                       # neuron x 900
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    F = U[:, :n_pc] * S[:n_pc]
    return ids, (F / F[:, 0].std()).astype(np.float32)


def features(names):
    ids, F = neuron_matrix()
    idx = {c: i for i, c in enumerate(ids)}
    G = np.zeros((len(names), F.shape[1]), np.float32)
    ok = np.array([bool(n) and n in idx for n in names])
    for i, n in enumerate(names):
        if ok[i]:
            G[i] = F[idx[n]]
    return G, ok


if __name__ == "__main__":
    import train_worm as tw, worm_models as wm
    names = wd.recording_names(); tr_n, _, _ = tw.get_split(names)
    tr = tw.load_recs(tr_n)
    vocab, _ = wm.assign_keys(tr, "M1", "neuron", None, np.random.default_rng(0))
    v = sorted(vocab, key=vocab.get)
    G, ok = features(v)
    print(f"{ok.sum()}/{len(v)} vocab names in the connectome; missing: {[n for n, o in zip(v, ok) if not o]}")
    print(f"scored names: {features(wd.model_cell_ids())[1].sum()}/154")
