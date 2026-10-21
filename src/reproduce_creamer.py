"""Reproduce the headline metric of Creamer/Leifer/Pillow 2025.

Loads the three published models and the measured STAMs / correlation matrices
that ship with the authors' repo, and recomputes, exactly as
`quick_start_examples/predict_stams.py` does:

    absolute score = corr(model prediction, measured TEST)
    ceiling        = corr(measured TRAIN, measured TEST)
    relative score = absolute / ceiling      <- the "92% of the ceiling" number

Diagonals are masked, so only neuron-to-neuron interactions are scored.
No model fitting: fitting needs MPI, evaluation does not.
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

# Import-only mpi4py shim must precede the repo imports; see stubs/mpi4py.
sys.path.insert(0, str(STUBS))
sys.path.insert(0, str(REPO))

import lgssm_utilities as ssmu  # noqa: E402
import metrics as met  # noqa: E402

MODEL_NAMES = ["connectome_constrained", "fully_connected", "shuffled_constrained"]
WINDOW = [15, 30]  # seconds [before, after], as in the paper's quick-start


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    models = {}
    for mn in MODEL_NAMES:
        with open(REPO / "models" / f"{mn}.pkl", "rb") as fh:
            models[mn] = pickle.load(fh)

    measured = {}
    with open(REPO / "data" / "measured_stams.pkl", "rb") as fh:
        measured["stams"] = pickle.load(fh)
    with open(REPO / "data" / "measured_corr.pkl", "rb") as fh:
        measured["corr"] = pickle.load(fh)

    n_neurons = measured["stams"]["train"].shape[0]
    diag = np.eye(n_neurons, dtype=bool)
    for metric in measured:
        for split in ["train", "test"]:
            measured[metric][split][diag] = np.nan

    print(f"neurons in model/metric matrices: {n_neurons}")
    for metric in ["stams", "corr"]:
        tr, te = measured[metric]["train"], measured[metric]["test"]
        print(
            f"  measured {metric}: train finite {np.isfinite(tr).sum():6d} / {tr.size}, "
            f"test finite {np.isfinite(te).sum():6d} / {te.size}"
        )

    preds = {}
    for name, model in models.items():
        preds[name] = {
            "stams": ssmu.calculate_stams(model, window=WINDOW),
            "corr": ssmu.predict_model_corr_coef(model),
        }
        for metric in preds[name]:
            preds[name][metric][diag] = np.nan

    ceiling = {
        m: met.nan_corr(measured[m]["train"], measured[m]["test"])[0]
        for m in ["stams", "corr"]
    }

    results = {"n_neurons": int(n_neurons), "window_s": WINDOW, "ceiling": {}, "models": {}}
    for m in ["stams", "corr"]:
        results["ceiling"][m] = float(ceiling[m])

    print("\nreproducibility ceiling  corr(measured train, measured test)")
    for m in ["stams", "corr"]:
        print(f"  {m:6s} {ceiling[m]:.4f}")

    print("\nmodel scores  [absolute = corr(prediction, measured test)]")
    for name in MODEL_NAMES:
        results["models"][name] = {}
        for m in ["stams", "corr"]:
            score, ci = met.nan_corr(preds[name][m], measured[m]["test"])
            rel = score / ceiling[m]
            results["models"][name][m] = {
                "absolute": float(score),
                "ci": [float(c) for c in np.asarray(ci).reshape(-1)],
                "relative_to_ceiling": float(rel),
            }
            print(
                f"  {name:24s} {m:6s} absolute {score:.4f}   "
                f"relative {rel:.3f}  ({rel * 100:.1f}% of ceiling)"
            )

    (OUT / "creamer_reproduction.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT / 'creamer_reproduction.json'}")


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
