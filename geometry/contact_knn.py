"""Does nerve-ring contact help predict NEVER-SEEN neuron classes? (exploratory)
The reported unseen-neuron prior (src/knn_prior.py, knn_nested.py): similarity-weighted measured maps, CeNGEN + connectome
features z-normalised and concatenated, k = 40, T = 5 (selected on training animals only) -> 0.147 on held-out-class entries.
Here, with k and T fixed at the reported values (nothing re-tuned), contact profiles (log1p nerve-ring contact to every
nerve-ring neuron, adult and L4) are added as a third similarity source, and tested alone. Same 5 class folds, same target
(authors' estimator, test animals). Neurons absent from the contact sheet get a zero contact vector (contact-neutral).
    python geometry/contact_knn.py -> geometry/results/contact_knn.json
"""
import json, pathlib, sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import knn_prior as kp  # noqa: E402

XL = ROOT / "data" / "connectome" / "nerve_ring_neighbors" / "Adult and L4 nerve ring neighbors.xlsx"


def contact_features(sheet):
    d = pd.read_excel(XL, sheet_name=sheet, header=None)
    names = [str(x).strip() for x in d.iloc[0, 1:]]; rows = [str(x).strip() for x in d.iloc[1:, 0]]
    M = d.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    keep = [i for i, n in enumerate(rows) if n != "nan" and i < len(names) and names[i] == n]
    K = np.log1p(M[np.ix_(keep, keep)]); K = (K + K.T) / 2; lab = [rows[i] for i in keep]; k = {n: i for i, n in enumerate(lab)}
    G = np.zeros((kp.n, len(lab))); cov = np.zeros(kp.n, bool)
    for a, c in enumerate(kp.cell):
        if c in k:
            G[a] = K[k[c]]; cov[a] = True
    mu, sd = G[cov].mean(0), G[cov].std(0) + 1e-9
    G = np.where(cov[:, None], (G - mu) / sd, 0.0)
    return G, cov


zn = lambda G: G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
Gb = np.concatenate([zn(kp.Ge), zn(kp.Gc)], 1)
res = {}
for sheet in ("adult nerve ring neighbors", "L4 nerve ring neighbors"):
    Gk, cov = contact_features(sheet)
    variants = {"both (banked)": Gb, "contact alone": Gk, "both+contact": np.concatenate([zn(kp.Ge), zn(kp.Gc), zn(Gk)], 1),
                "cengen+contact": np.concatenate([zn(kp.Ge), zn(Gk)], 1), "connectome+contact": np.concatenate([zn(kp.Gc), zn(Gk)], 1)}
    out = {}
    for vn, G in variants.items():
        per = {"held": [], "held_covered": []}
        for f in range(5):
            held = set(np.random.default_rng(12345).permutation(kp.all_cls)[f::5])
            h = np.array([c in held for c in kp.cls_of])
            P = kp.predict(G, h, 40)
            hm = h[:, None] | h[None, :]
            per["held"].append(kp.corr(P, kp.T, hm))
            hc = (h & cov)[:, None] & cov[None, :] | cov[:, None] & (h & cov)[None, :]
            per["held_covered"].append(kp.corr(P, kp.T, hc))
        out[vn] = {k: [float(np.mean(v)), [round(x, 3) for x in v]] for k, v in per.items()}
        print(sheet.split()[0], f"{vn:20s}", {k: round(v[0], 3) for k, v in out[vn].items()}, out[vn]["held"][1], flush=True)
    res[sheet.split()[0]] = out
json.dump(res, open(ROOT / "geometry" / "results" / "contact_knn.json", "w"), indent=1)
