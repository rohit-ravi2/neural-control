"""Nested selection for the kNN prior: choose (similarity source, k) on TRAINING animals only, then score once on test.
Selection split: our 80 training animals -> halves A/B (seed 0); maps from A predict B's held-out entries (same class folds).
Candidates: source in {cengen, connectome, both} x k in {5, 10, 20, 40, all}; temperature fixed at 5.
    python knn_nested.py -> results/knn_nested.json
"""
import json, warnings

import numpy as np

import knn_prior as kp
import phase0_ceiling as p0
import train_worm as tw
import worm_data as wd

tr_n, _, _ = tw.get_split(wd.recording_names())
perm = np.random.default_rng(0).permutation(tr_n)
A, B = sorted(perm[:40]), sorted(perm[40:])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    SA, SB = p0.authors_stams(A, kp.cell), p0.authors_stams(B, kp.cell)
zn = lambda G: G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
SRC = {"cengen": kp.Ge, "connectome": kp.Gc, "both": np.concatenate([zn(kp.Ge), zn(kp.Gc)], 1)}
KS = (5, 10, 20, 40, 1000)


def fold_scores(G, k, src_maps, tgt):
    out = []
    for f in range(5):
        held = set(np.random.default_rng(12345).permutation(kp.all_cls)[f::5])
        h = np.array([c in held for c in kp.cls_of])
        kp.TR = src_maps  # predict() reads kp.TR
        out.append(kp.corr(kp.predict(G, h, min(k, (~h).sum())), tgt, h[:, None] | h[None, :]))
    return out


sel = {f"{s}_k{k}": float(np.mean(fold_scores(G, k, SA, SB))) for s, G in SRC.items() for k in KS}
best = max(sel, key=sel.get)
print("selection (train halves A->B):", {k: round(v, 3) for k, v in sel.items()}, "\nchosen:", best, flush=True)
TR_full = np.load(p0.OUT / "targets_test.npz")["T_auth_train"]
res = {}
for s, G in SRC.items():  # test scores of each source at its own train-selected k, plus the overall winner
    ks = max(KS, key=lambda k: sel[f"{s}_k{k}"])
    sc = fold_scores(G, ks, TR_full, kp.T)
    res[s] = {"k_selected_on_train": ks, "test_held": float(np.mean(sc)), "per_fold": sc}
    print(f"test {s:10s} k={ks}: {np.mean(sc):.3f}  folds {np.round(sc, 3)}", flush=True)
d = np.array(res["both"]["per_fold"]) - np.maximum(res["cengen"]["per_fold"], res["connectome"]["per_fold"])
print("both - best single, per fold:", np.round(d, 3))
json.dump({"selection": sel, "chosen": best, "test": res, "both_minus_best_single": d.tolist()},
          open("../results/knn_nested.json", "w"), indent=1)
