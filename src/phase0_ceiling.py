"""Phase 0 — noise-ceiling verification.

Four checks, all written to results/phase0/:

  A. Split recovery: recompute the measured STAMs from the raw OSF recordings with the
     authors' preprocessing and IRF estimator, under the positional 80/30 split that
     load_data(hold_out='worm', num_data_sets=80, hold_out_start=0.9) produces, and
     correlate with the shipped data/measured_stams.pkl.
  B. Event table: every stimulation event -> (animal, target, 154-vector of integrated
     post-stimulus response). STAM(set of animals) = mean over that set's events.
  C. Reliability vs number of animals: disjoint random animal halves of size N, and
     date-blocked halves, to test whether split reliability follows Spearman-Brown in N.
  D. Synthetic ground truth on the REAL coverage (which pairs each animal measured, and
     how many events): known true STAM s, event noise sigma chosen so rho_obs matches the
     data. Compare the true achievable ceiling corr(s, test STAM) with the noise_ceiling.py
     estimate sqrt(rho_30) and with the paper's normalisation.
"""
from __future__ import annotations

import json
import pathlib
import pickle
import sys
import warnings

import numpy as np

import worm_data as wd

sys.path.insert(0, str(wd.STUBS))
sys.path.insert(0, str(wd.REPO))
import lgssm_utilities as ssmu  # noqa: E402
import metrics as met  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1].joinpath("results/phase0").expanduser()
WINDOW = (15, 30)  # seconds, as in the paper
PRE, POST = WINDOW[0] * wd.SAMPLE_RATE, WINDOW[1] * wd.SAMPLE_RATE


def positional_split(names: list[str], n_train: int = 80, start_frac: float = 0.9):
    names = list(names)
    n_test = min(len(names) - n_train, n_train)
    start = int(start_frac * len(names))
    test_idx = sorted(np.mod(np.arange(start, start + n_test), len(names)))
    test = [names[i] for i in test_idx]
    train = [n for i, n in enumerate(names) if i not in set(test_idx)]
    return train, test


def authors_stams(names, cell_ids):
    ems, ins, _ = wd.load_aligned(names, cell_ids)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        irfs = ssmu.get_impulse_response_functions(ems, ins, sample_rate=wd.SAMPLE_RATE,
                                                   window=WINDOW, sub_pre_stim=True)[0]
    stams = np.sum(irfs[PRE:], axis=0)
    stams[np.eye(len(cell_ids), dtype=bool)] = np.nan
    return stams


def event_table(names, cell_ids):
    """(animal index, target column, integrated response vector) for every usable event."""
    ems, ins, _ = wd.load_aligned(names, cell_ids)
    animals, targets, vecs = [], [], []
    for a, (e, i) in enumerate(zip(ems, ins)):
        for t, j in zip(*np.where(i == 1)):
            if t - PRE >= 0 and t + POST < e.shape[0]:
                clip = e[t - PRE : t + POST]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clip = clip - np.nanmean(clip[:PRE], axis=0)
                    v = np.nansum(clip[PRE:], axis=0)
                v[np.all(np.isnan(clip[PRE:]), axis=0)] = np.nan
                animals.append(a)
                targets.append(j)
                vecs.append(v)
    return np.array(animals), np.array(targets), np.array(vecs, dtype=np.float64)


def stam_from_events(animals, targets, vecs, animal_set, n):
    sel = np.isin(animals, list(animal_set))
    out = np.full((n, n), np.nan)
    for j in np.unique(targets[sel]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[:, j] = np.nanmean(vecs[sel & (targets == j)], axis=0)
    out[np.eye(n, dtype=bool)] = np.nan
    return out


def corr(a, b) -> float:
    return float(met.nan_corr(a, b)[0])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    names = wd.recording_names()
    cell_ids = wd.model_cell_ids()
    n = len(cell_ids)
    shipped = pickle.load(open(wd.REPO / "data" / "measured_stams.pkl", "rb"))
    for k in shipped:
        shipped[k][np.eye(n, dtype=bool)] = np.nan
    report: dict = {"n_recordings": len(names), "n_neurons": n}

    # ---- A. split recovery ----
    train, test = positional_split(names)
    s_tr, s_te = authors_stams(train, cell_ids), authors_stams(test, cell_ids)
    report["A_split_recovery"] = {
        "test_names_first_last": [test[0], test[-1]],
        "corr_recomputed_vs_shipped_train": corr(s_tr, shipped["train"]),
        "corr_recomputed_vs_shipped_test": corr(s_te, shipped["test"]),
        "rho_recomputed": corr(s_tr, s_te),
        "rho_shipped": corr(shipped["train"], shipped["test"]),
    }
    print("A.", json.dumps(report["A_split_recovery"], indent=1), flush=True)

    # ---- B. event table ----
    animals, targets, vecs = event_table(names, cell_ids)
    idx = {nm: i for i, nm in enumerate(names)}
    tr_set, te_set = {idx[x] for x in train}, {idx[x] for x in test}
    e_tr = stam_from_events(animals, targets, vecs, tr_set, n)
    e_te = stam_from_events(animals, targets, vecs, te_set, n)
    report["B_event_table"] = {
        "n_events": int(len(animals)),
        "event_stam_vs_authors_train": corr(e_tr, s_tr),
        "event_stam_vs_authors_test": corr(e_te, s_te),
        "rho_event": corr(e_tr, e_te),
    }
    # per-pair event counts on each side (for the synthetic model)
    obs = ~np.isnan(vecs)
    cnt = {}
    for side, aset in (("train", tr_set), ("test", te_set)):
        c = np.zeros((n, n))
        sel = np.isin(animals, list(aset))
        for j in range(n):
            m = sel & (targets == j)
            c[:, j] = obs[m].sum(axis=0)
        c[np.eye(n, dtype=bool)] = 0
        cnt[side] = c
    both = (cnt["train"] > 0) & (cnt["test"] > 0)
    report["B_event_table"]["pairs_scored"] = int(both.sum())
    report["B_event_table"]["events_per_pair_train_median"] = float(np.median(cnt["train"][both]))
    report["B_event_table"]["events_per_pair_test_median"] = float(np.median(cnt["test"][both]))
    report["B_event_table"]["mean_inv_count_train"] = float(np.mean(1 / cnt["train"][both]))
    report["B_event_table"]["mean_inv_count_test"] = float(np.mean(1 / cnt["test"][both]))
    print("B.", json.dumps(report["B_event_table"], indent=1), flush=True)

    # ---- C. reliability vs number of animals ----
    all_a = np.arange(len(names))
    curve = {}
    for N in (5, 10, 20, 30, 40, 55):
        rs = []
        for _ in range(40 if N < 40 else 20):
            p = rng.permutation(all_a)
            rs.append(corr(stam_from_events(animals, targets, vecs, p[:N], n),
                           stam_from_events(animals, targets, vecs, p[N : 2 * N], n)))
        curve[N] = (float(np.mean(rs)), float(np.std(rs)))
        print(f"C. random halves N={N:2d}: r = {curve[N][0]:.4f} ± {curve[N][1]:.4f}", flush=True)
    # Spearman-Brown in N: r(N) = N / (N + k). Fit k by least squares on 1/r - 1 = k/N.
    Ns = np.array(list(curve))
    rv = np.array([curve[N][0] for N in Ns])
    k_fit = float(np.sum((1 / rv - 1) * (1 / Ns)) / np.sum((1 / Ns) ** 2))
    pred = Ns / (Ns + k_fit)
    # date-blocked halves (names are date-sorted): early 55 vs late 55
    half = len(names) // 2
    r_date = corr(stam_from_events(animals, targets, vecs, all_a[:half], n),
                  stam_from_events(animals, targets, vecs, all_a[half : 2 * half], n))
    report["C_reliability_curve"] = {
        "random_halves": {int(k): v for k, v in curve.items()},
        "spearman_brown_k": k_fit,
        "sb_predicted": dict(zip(map(int, Ns), map(float, pred))),
        "sb_residuals": dict(zip(map(int, Ns), map(float, rv - pred))),
        "date_blocked_55v55": r_date,
        "random_55v55": curve[55][0],
    }
    print(f"C. SB k={k_fit:.1f}; residuals {np.round(rv - pred, 4)}; date-blocked 55v55 r={r_date:.4f} vs random {curve[55][0]:.4f}", flush=True)

    # ---- D. synthetic ground truth on real coverage ----
    # event response = s[:, j] + animal-specific deviation (tau) + event noise (sigma), on the
    # real observation mask; STAMs pooled over events as the authors do.
    real_te = e_te
    var_s = 1.0
    synth = {}
    for tau in (0.0, 1.0):
        for sigma in (5.0, 10.0, 15.0, 20.0, 30.0):
            vals = []
            for rep in range(5):
                s = rng.normal(0, np.sqrt(var_s), (n, n))
                dev = rng.normal(0, tau, (len(names), n, n)) if tau > 0 else None
                sv = s[:, targets].T.copy()                       # (events, n)
                if dev is not None:
                    sv += dev[animals, :, targets]
                sv += rng.normal(0, sigma, sv.shape)
                sv[~obs] = np.nan
                str_ = stam_from_events(animals, targets, sv, tr_set, n)
                ste = stam_from_events(animals, targets, sv, te_set, n)
                strue = s.copy()
                strue[np.eye(n, dtype=bool)] = np.nan
                m = both & ~np.eye(n, dtype=bool)
                rho = corr(np.where(m, str_, np.nan), np.where(m, ste, np.nan))
                true_ceiling = corr(np.where(m, strue, np.nan), np.where(m, ste, np.nan))
                k = None
                try:
                    import noise_ceiling as nc
                    k = nc.solve_k(rho, 80, 30)
                    est = float(np.sqrt(1 / (1 + k / 30)))
                except Exception:
                    est = float("nan")
                vals.append((rho, true_ceiling, est, true_ceiling / rho))
            v = np.array(vals).mean(0)
            synth[f"tau{tau}_sigma{sigma}"] = {"rho_obs": v[0], "true_ceiling": v[1],
                                               "noise_ceiling_py_estimate": v[2],
                                               "perfect_model_paper_relative": v[3]}
            print(f"D. tau={tau} sigma={sigma:4.1f}: rho_obs {v[0]:.4f}  true ceiling {v[1]:.4f}  "
                  f"sqrt(rho30) est {v[2]:.4f}  perfect-model paper-relative {v[3]:.2f}", flush=True)
    report["D_synthetic"] = synth
    json.dump(report, open(OUT / "phase0_report.json", "w"), indent=1)
    print("wrote", OUT / "phase0_report.json")


if __name__ == "__main__":
    main()
