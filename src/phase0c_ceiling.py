"""Phase 0c: phase0b with the (tau, sigma) grid extended past the 0b edge (tau=3 was the max).

Phase 0 showed: (a) the positional 80/30 split does not recover the shipped test STAMs;
(b) per-pair STAMs pool a median of 6 (train) / 2 (test) EVENTS, not 80 / 30 animals;
(c) split-half reliability rises far more slowly in N animals than Spearman-Brown predicts.

This script:
  A2. Tries the split in submission_params/exp_test.yml: its 10 held_out_data recordings go
      to test first, then load_data's positional rule picks the rest with num_data_sets=80.
  C2. Fits the real reliability curve with a generative model on the REAL coverage:
        event response = s_ij + a_{animal,ij} + e,   var(s)=1, a ~ N(0, tau^2), e ~ N(0, sigma^2)
      grid over (sigma, tau); the synthetic split-half curve is computed with exactly the same
      animal draws and pooling as the real one. tau > 0 = variability that does not average
      away over events within an animal.
  D2. At the fitted (sigma, tau), compares against the known truth:
        true ceiling      corr(s, test STAM)                     (s = population mean)
        estimator 1       sqrt(rho_30) via noise_ceiling.solve_k(rho, 80, 30)  (animal-SB)
        estimator 2       count-based: solve (1+q a)(1+q b) = 1/rho^2 with a,b = mean 1/n per
                          pair on each side; ceiling = 1/sqrt(1 + q b)
      and reports the model scores against each.
"""
from __future__ import annotations

import json
import pickle

import numpy as np

import noise_ceiling as nc
import phase0_ceiling as p0
import worm_data as wd


def split_with_holdout(names, held, n_train=80, start_frac=0.9):
    rest = [n for n in names if n not in held]
    n_test = min(len(rest) - n_train, n_train)
    start = int(start_frac * len(rest))
    ti = set(np.mod(np.arange(start, start + n_test), len(rest)).tolist())
    test = [n for n in names if n in held] + [n for i, n in enumerate(rest) if i in ti]
    train = [n for i, n in enumerate(rest) if i not in ti]
    return train, test


def synth_vecs(s, tau, sigma, animals, targets, obs, n_animals, rng):
    n = s.shape[0]
    v = s[:, targets].T.copy()
    if tau > 0:
        # animal-specific deviation per (animal, responder, target), generated lazily per event
        keys = animals * n + targets
        uniq, inv = np.unique(keys, return_inverse=True)
        dev = rng.normal(0, tau, (len(uniq), n))
        v += dev[inv]
    v += rng.normal(0, sigma, v.shape)
    v[~obs] = np.nan
    return v


def halves_curve(animals, targets, vecs, n, Ns, reps, seed):
    rng = np.random.default_rng(seed)  # same draws for real and synthetic
    all_a = np.unique(animals)
    out = {}
    for N in Ns:
        rs = []
        for _ in range(reps):
            p = rng.permutation(all_a)
            rs.append(p0.corr(p0.stam_from_events(animals, targets, vecs, p[:N], n),
                              p0.stam_from_events(animals, targets, vecs, p[N : 2 * N], n)))
        out[N] = float(np.mean(rs))
    return out


def count_ceiling(rho, a, b):
    t = 1 / rho**2
    A, B, C = a * b, a + b, 1 - t
    q = (-B + np.sqrt(B * B - 4 * A * C)) / (2 * A)
    return float(1 / np.sqrt(1 + q * b)), float(q)


def main():
    names = wd.recording_names()
    cell_ids = wd.model_cell_ids()
    n = len(cell_ids)
    shipped = pickle.load(open(wd.REPO / "data" / "measured_stams.pkl", "rb"))
    for k in shipped:
        shipped[k][np.eye(n, dtype=bool)] = np.nan
    rep = {}

    # ---- A2 ----
    import yaml
    held = yaml.safe_load(open(wd.REPO / "submission_params" / "exp_test.yml"))["held_out_data"]
    train, test = split_with_holdout(names, held)
    s_tr, s_te = p0.authors_stams(train, cell_ids), p0.authors_stams(test, cell_ids)
    rep["A2_split_with_holdout_list"] = {
        "n_train": len(train), "n_test": len(test),
        "corr_vs_shipped_train": p0.corr(s_tr, shipped["train"]),
        "corr_vs_shipped_test": p0.corr(s_te, shipped["test"]),
        "rho": p0.corr(s_tr, s_te)}
    print("A2", json.dumps(rep["A2_split_with_holdout_list"], indent=1), flush=True)
    # use whichever split recovers the shipped test STAMs better for D2
    ptr, pte = p0.positional_split(names)
    pos_te = p0.corr(p0.authors_stams(pte, cell_ids), shipped["test"])
    if rep["A2_split_with_holdout_list"]["corr_vs_shipped_test"] < pos_te:
        train, test = ptr, pte
    rep["split_used"] = "holdout_list" if train is not ptr else "positional"

    animals, targets, vecs = p0.event_table(names, cell_ids)
    obs = ~np.isnan(vecs)
    idx = {nm: i for i, nm in enumerate(names)}
    tr_set, te_set = [idx[x] for x in train], [idx[x] for x in test]

    # ---- C2 ----
    Ns, reps = (5, 10, 20, 40, 55), 12
    real = halves_curve(animals, targets, vecs, n, Ns, reps, seed=1)
    # scale of real per-event vectors, so sigma grid is in data units relative to var(s)=1 later
    print("C2 real curve", {k: round(v, 4) for k, v in real.items()}, flush=True)
    rng = np.random.default_rng(2)
    s = rng.normal(0, 1, (n, n))
    best, grid = None, []
    for tau in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        for sigma in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            sv = synth_vecs(s, tau, sigma, animals, targets, obs, len(names), rng)
            syn = halves_curve(animals, targets, sv, n, Ns, reps, seed=1)
            err = float(np.sqrt(np.mean([(syn[N] - real[N]) ** 2 for N in Ns])))
            grid.append({"tau": tau, "sigma": sigma, "rmse": err, "curve": syn})
            if best is None or err < best["rmse"]:
                best = grid[-1]
            print(f"  tau {tau:.1f} sigma {sigma:.1f}  rmse {err:.4f}  " +
                  " ".join(f"{syn[N]:.3f}" for N in Ns), flush=True)
    rep["C2_fit"] = {"real_curve": real, "best": best, "grid": grid}
    print("C2 best", best["tau"], best["sigma"], round(best["rmse"], 4), flush=True)

    # ---- D2 ----
    rows = []
    for r in range(10):
        s = rng.normal(0, 1, (n, n))
        sv = synth_vecs(s, best["tau"], best["sigma"], animals, targets, obs, len(names), rng)
        st = p0.stam_from_events(animals, targets, sv, tr_set, n)
        se = p0.stam_from_events(animals, targets, sv, te_set, n)
        m = ~np.isnan(st) & ~np.isnan(se)
        rho = p0.corr(np.where(m, st, np.nan), np.where(m, se, np.nan))
        s0 = s.copy(); s0[np.eye(n, dtype=bool)] = np.nan
        true_c = p0.corr(np.where(m, s0, np.nan), np.where(m, se, np.nan))
        # counts on scored pairs
        ntr = np.zeros((n, n)); nte = np.zeros((n, n))
        for side, aset, arr in (("tr", tr_set, ntr), ("te", te_set, nte)):
            sel = np.isin(animals, aset)
            for j in range(n):
                arr[:, j] = obs[sel & (targets == j)].sum(0)
        a, b = float(np.mean(1 / ntr[m])), float(np.mean(1 / nte[m]))
        est1 = float(np.sqrt(1 / (1 + nc.solve_k(rho, 80, 30) / 30)))
        est2, _ = count_ceiling(rho, a, b)
        rows.append((rho, true_c, est1, est2))
    rho, true_c, est1, est2 = np.array(rows).mean(0)
    sd = np.array(rows).std(0)
    rep["D2_synthetic_at_fit"] = {"rho_obs": rho, "true_ceiling": true_c, "true_ceiling_sd": sd[1],
                                  "est_animal_SB_sqrt_rho30": est1, "est_count_based": est2,
                                  "perfect_model_paper_relative": true_c / rho}
    print("D2", json.dumps(rep["D2_synthetic_at_fit"], indent=1), flush=True)

    # real-data estimates with both estimators, on the shipped STAMs and on the recomputed split
    real_rho_shipped = p0.corr(shipped["train"], shipped["test"])
    st = p0.stam_from_events(animals, targets, vecs, tr_set, n)
    se = p0.stam_from_events(animals, targets, vecs, te_set, n)
    m = ~np.isnan(st) & ~np.isnan(se)
    ntr = np.zeros((n, n)); nte = np.zeros((n, n))
    for aset, arr in ((tr_set, ntr), (te_set, nte)):
        sel = np.isin(animals, aset)
        for j in range(n):
            arr[:, j] = obs[sel & (targets == j)].sum(0)
    a, b = float(np.mean(1 / ntr[m])), float(np.mean(1 / nte[m]))
    ratio = rep["D2_synthetic_at_fit"]["true_ceiling"] / rep["D2_synthetic_at_fit"]["rho_obs"]
    rep["real_estimates"] = {
        "rho_shipped": real_rho_shipped,
        "mean_inv_count_train": a, "mean_inv_count_test": b,
        "ceiling_animal_SB": float(np.sqrt(1 / (1 + nc.solve_k(real_rho_shipped, 80, 30) / 30))),
        "ceiling_count_based": count_ceiling(real_rho_shipped, a, b)[0],
        "ceiling_generative_fit_scaled": float(real_rho_shipped * ratio),
        "connectome_constrained_r": 0.14032,
    }
    for k in ("ceiling_animal_SB", "ceiling_count_based", "ceiling_generative_fit_scaled"):
        rep["real_estimates"]["model_pct_of_" + k] = 0.14032 / rep["real_estimates"][k] * 100
    print("REAL", json.dumps(rep["real_estimates"], indent=1), flush=True)
    json.dump(rep, open(p0.OUT / "phase0c_report.json", "w"), indent=1, default=float)
    print("wrote", p0.OUT / "phase0c_report.json")


if __name__ == "__main__":
    main()
