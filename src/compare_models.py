"""Are the differences between the published models statistically real?

Caches model predictions, then uses the authors' own paired bootstrap
(metrics.two_sample_boostrap_corr_p) to compare, against the held-out measured
data: connectome-constrained vs fully-connected, and each vs the shuffled
control.
"""
from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1].joinpath("work/Creamer_LDS_2024").expanduser()
STUBS = pathlib.Path(__file__).resolve().parents[1].joinpath("src/stubs").expanduser()
OUT = pathlib.Path(__file__).resolve().parents[1].joinpath("results").expanduser()
CACHE = OUT / "model_predictions.npz"

sys.path.insert(0, str(STUBS))
sys.path.insert(0, str(REPO))

import lgssm_utilities as ssmu  # noqa: E402
import metrics as met  # noqa: E402

MODEL_NAMES = ["connectome_constrained", "fully_connected", "shuffled_constrained"]
WINDOW = [15, 30]


def load_measured():
    measured = {}
    with open(REPO / "data" / "measured_stams.pkl", "rb") as fh:
        measured["stams"] = pickle.load(fh)
    with open(REPO / "data" / "measured_corr.pkl", "rb") as fh:
        measured["corr"] = pickle.load(fh)
    n = measured["stams"]["train"].shape[0]
    diag = np.eye(n, dtype=bool)
    for metric in measured:
        for split in ["train", "test"]:
            measured[metric][split][diag] = np.nan
    return measured, diag


def get_predictions(diag):
    if CACHE.exists():
        z = np.load(CACHE)
        return {m: {"stams": z[f"{m}__stams"], "corr": z[f"{m}__corr"]} for m in MODEL_NAMES}
    preds = {}
    for name in MODEL_NAMES:
        with open(REPO / "models" / f"{name}.pkl", "rb") as fh:
            model = pickle.load(fh)
        preds[name] = {
            "stams": ssmu.calculate_stams(model, window=WINDOW),
            "corr": ssmu.predict_model_corr_coef(model),
        }
        for metric in preds[name]:
            preds[name][metric][diag] = np.nan
    np.savez(
        CACHE,
        **{f"{n}__{m}": preds[n][m] for n in MODEL_NAMES for m in ["stams", "corr"]},
    )
    return preds


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    measured, diag = load_measured()
    preds = get_predictions(diag)
    rng = np.random.default_rng(0)

    results = {}
    pairs = [
        ("connectome_constrained", "fully_connected"),
        ("connectome_constrained", "shuffled_constrained"),
        ("fully_connected", "shuffled_constrained"),
    ]
    for metric in ["stams", "corr"]:
        target = measured[metric]["test"]
        n_pairs = int((np.isfinite(target) & ~diag).sum())
        results[metric] = {"n_scored_pairs": n_pairs, "comparisons": {}}
        print(f"\n=== {metric} === scored neuron pairs: {n_pairs}")
        for a, b in pairs:
            # returns (p, delta, ci) where delta = corr(target,a) - corr(target,b)
            p, delta, ci = met.two_sample_boostrap_corr_p(
                target, preds[a][metric], preds[b][metric], n_boot=2000, rng=rng
            )
            ca = met.nan_corr(preds[a][metric], target)[0]
            cb = met.nan_corr(preds[b][metric], target)[0]
            results[metric]["comparisons"][f"{a}_vs_{b}"] = {
                "corr_a": float(ca),
                "corr_b": float(cb),
                "delta": float(delta),
                "ci": [float(c) for c in np.asarray(ci).reshape(-1)],
                "p": float(p),
            }
            print(
                f"  {a} ({ca:.4f}) vs {b} ({cb:.4f})  "
                f"delta {delta:+.4f} [-{ci[0]:.4f},+{ci[1]:.4f}]  p={p:.4f}"
            )

    (OUT / "model_comparisons.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT / 'model_comparisons.json'}")


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
