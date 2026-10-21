"""CeNGEN expression features for the worm models (Phase 3).

Source: CeNGEN 021821 thresholded release, medium threshold 2 (Taylor et al. 2021), 13,669 genes x 128 neuron
classes, at data/cengen/021821_medium_threshold2.csv (see README).
Features are fixed before any worm model sees them: log1p(TPM) -> the 2,000 genes most variable across
classes -> z-score per gene -> PCA to 32 dims, fitted on the 128 CeNGEN classes alone.
"""
from __future__ import annotations

import functools
import re
import pathlib

import numpy as np
import pandas as pd

CSV = pathlib.Path(__file__).resolve().parents[1] / "data/cengen/021821_medium_threshold2.csv"
EXPLICIT = {"IL2DL": "IL2_DV", "IL2DR": "IL2_DV", "IL2VL": "IL2_DV", "IL2VR": "IL2_DV", "IL2L": "IL2_LR", "IL2R": "IL2_LR",
            "RMDDL": "RMD_DV", "RMDDR": "RMD_DV", "RMDVL": "RMD_DV", "RMDVR": "RMD_DV", "RMDL": "RMD_LR", "RMDR": "RMD_LR",
            "RMED": "RME_DV", "RMEV": "RME_DV", "RMEL": "RME_LR", "RMER": "RME_LR",
            "DB1": "DB01", "VB1": "VB01", "VB2": "VB02", "DD1": "VD_DD", "DA9": "DA9", "VA12": "VA12",
            "AWCL": None, "AWCR": None}  # AWC ON/OFF identity is not known per animal


# Phase 3c gene groups, by WormBase gene-name prefix (fixed 2026-10-04 before any 3c run). Crude but declared.
_FAM = {  # family prefixes: the gene name must be "<prefix>-<digit>..."
    "channels": "kcnl shk shl slo kvs twk irk nca cca kqt cng ocr trp trpa gtl lov pkd deg del asic inx clh best egas unc".split(),
    "receptors": "glr nmr acr lgc gar dop ser avr glc lev gbb ggr tyra octr gur".split(),
    "neuropeptide": "flp nlp ins ntc pdf npr frpr dmsr ckr tkr pdfr seb nmur trhr gnrr".split(),
    "tf": "ceh nhr hlh lim mls zag sox fkh mab ets zfh cnd ttx".split(),
}
_EXACT = {  # single genes (exact names)
    "channels": {"egl-2", "egl-19", "egl-36", "unc-2", "unc-103", "exp-2", "tax-2", "tax-4", "osm-9", "unc-8", "mec-4",
                 "mec-10", "unc-7", "unc-9", "nca-1", "nca-2", "unc-77", "unc-79", "unc-80"},
    "receptors": {"mod-1", "unc-29", "unc-38", "unc-63", "unc-49", "exp-1", "eat-2"},
    "neuropeptide": {"egl-6", "aex-2", "daf-37", "daf-38", "egl-3", "egl-21"},
    "tf": {"unc-86", "unc-3", "unc-42", "unc-4", "unc-30", "unc-55", "die-1", "lin-11", "lin-39", "egl-13", "egl-44",
           "egl-46", "che-1", "ast-1", "pag-3", "vab-3", "vab-7", "unc-39", "unc-62", "egl-5", "php-3", "nob-1", "tab-1",
           "sem-4", "cfi-1", "mec-3", "egl-43"},
}
# "unc-<digit>" is not a channel family: unc genes enter only through _EXACT
_FAM["channels"].remove("unc")
GROUPS = {k: re.compile(r"^(" + "|".join(_FAM[k]) + r")-\d") for k in _FAM}


def _in_group(name: str, key: str) -> bool:
    return bool(GROUPS[key].match(name)) or name in _EXACT[key]


@functools.lru_cache
def _raw():
    df = pd.read_csv(CSV, index_col=0)
    X = np.log1p(df.drop(columns=["gene_name", "Wormbase_ID"]).astype(float))   # genes x classes
    return df["gene_name"].astype(str).to_numpy(), list(X.columns), X.to_numpy()


def gene_index(genes: str):
    """'all' -> None (top-2000 variable genes); a GROUPS key -> its genes; 'random_<key>' -> same count of random
    expressed genes (seed 7) drawn outside that group."""
    names, _, X = _raw()
    expressed = X.var(1) > 0
    if genes == "all":
        return None
    key = genes.removeprefix("random_")
    hit = np.array([_in_group(n, key) for n in names]) & expressed
    if genes.startswith("random_"):
        pool = np.where(expressed & ~hit)[0]
        return np.sort(np.random.default_rng(7).choice(pool, hit.sum(), replace=False))
    return np.where(hit)[0]


@functools.lru_cache
def class_matrix(n_genes: int = 2000, n_pc: int = 32, genes: str = "all"):
    _, classes, X = _raw()
    gi = gene_index(genes)
    top = np.argsort(X.var(1))[::-1][:n_genes] if gi is None else gi
    n_pc = min(n_pc, len(top) - 1)
    Z = X[top].T                                                               # classes x genes
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    F = U[:, :n_pc] * S[:n_pc]
    F = F / F[:, 0].std()
    return classes, F.astype(np.float32)


def cengen_class(name: str, classes) -> str | None:
    if name in EXPLICIT:
        return EXPLICIT[name]
    m = re.fullmatch(r"(VD|DD)\d+", name) and "VD_DD" or re.fullmatch(r"VC0?[45]", name) and "VC_4_5"
    if m:
        return m
    for k in range(0, 4):
        cand = name[: len(name) - k] if k else name
        if cand in classes:
            return cand
    return None


def features(names, genes: str = "all") -> tuple[np.ndarray, np.ndarray, list]:
    """(len(names), n_pc) features, a mask of names with a CeNGEN class, and the class per name."""
    classes, F = class_matrix(genes=genes)
    idx = {c: i for i, c in enumerate(classes)}
    cls = [cengen_class(n, classes) if n else None for n in names]
    G = np.zeros((len(names), F.shape[1]), np.float32)
    ok = np.array([c is not None for c in cls])
    for i, c in enumerate(cls):
        if c is not None:
            G[i] = F[idx[c]]
    return G, ok, cls


if __name__ == "__main__":
    import worm_data as wd
    names = wd.model_cell_ids()
    for g in GROUPS:
        print(g, len(gene_index(g)), "genes; e.g.", list(_raw()[0][gene_index(g)][:12]))
    G, ok, cls = features(names)
    print(f"{ok.sum()}/{len(names)} names mapped to {len({c for c in cls if c})} CeNGEN classes; unmapped:",
          [n for n, o in zip(names, ok) if not o])
    print("examples:", [(n, c) for n, c in zip(names, cls)][::12])
