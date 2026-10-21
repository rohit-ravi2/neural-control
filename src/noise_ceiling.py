"""Correct noise-ceiling normalization for the Creamer/Leifer/Pillow STAM metric.

The paper normalizes model performance by the train-test correlation rho and
states in Methods that "a perfect model ... will achieve a relative correlation
of 1". That is not right. With measurements = signal + independent noise, a
perfect model predicting the NOISELESS signal correlates with a noisy
measurement at sqrt(reliability), not reliability.

Here the two sides are averages over unequal numbers of animals (80 train,
30 test), so the correction needs care:

    reliability of an N-animal average:  rho_N = 1 / (1 + k/N),  k = var_noise/var_signal
    observed cross correlation:          rho_obs = corr(avg_80, avg_30) = sqrt(rho_80 * rho_30)
    achievable ceiling for a model
    evaluated against the 30-animal avg: sqrt(rho_30)

Solve the quadratic in k from rho_obs, then report the ceiling.

Caveat recorded honestly: this assumes animal-to-animal variability is iid noise
that averages down as 1/N. Any genuine, non-diminishing biological variability
between animals raises the true ceiling.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1].joinpath("results").expanduser()


def solve_k(rho_obs: float, n_a: int, n_b: int) -> float:
    """var_noise/var_signal implied by corr(avg_n_a, avg_n_b) = rho_obs."""
    # (1 + k/n_a)(1 + k/n_b) = 1/rho_obs^2
    target = 1.0 / rho_obs**2
    a = 1.0 / (n_a * n_b)
    b = 1.0 / n_a + 1.0 / n_b
    c = 1.0 - target
    disc = b**2 - 4 * a * c
    return (-b + np.sqrt(disc)) / (2 * a)


def main() -> None:
    rho_obs = 0.17336  # reproduced from the authors' shipped measured_stams.pkl
    n_train, n_test = 80, 30
    models = {
        "connectome_constrained": 0.14032,
        "fully_connected": 0.11964,
        "shuffled_constrained": 0.04032,
    }

    k = solve_k(rho_obs, n_train, n_test)
    rho_train = 1.0 / (1.0 + k / n_train)
    rho_test = 1.0 / (1.0 + k / n_test)
    ceiling_test = np.sqrt(rho_test)
    ceiling_naive = np.sqrt(rho_obs)

    print(f"observed corr(train80, test30)      rho_obs      = {rho_obs:.4f}")
    print(f"implied noise/signal variance ratio k            = {k:.1f}")
    print(f"reliability of the 80-animal average rho_80      = {rho_train:.4f}")
    print(f"reliability of the 30-animal average rho_30      = {rho_test:.4f}")
    print(f"check sqrt(rho_80*rho_30)                        = {np.sqrt(rho_train*rho_test):.4f}")
    print()
    print(f"ceiling, naive sqrt(rho_obs)                     = {ceiling_naive:.4f}")
    print(f"ceiling, correct for 30-animal target sqrt(rho_30)= {ceiling_test:.4f}")
    print()
    rows = {}
    print(f"{'model':24s} {'r':>7s} {'/rho (paper)':>13s} {'/sqrt(rho_obs)':>15s} {'/sqrt(rho_30)':>14s}")
    for name, r in models.items():
        rows[name] = {
            "r": r,
            "paper_relative": r / rho_obs,
            "vs_naive_ceiling": r / ceiling_naive,
            "vs_correct_ceiling": r / ceiling_test,
        }
        print(
            f"{name:24s} {r:7.4f} {r/rho_obs*100:12.1f}% {r/ceiling_naive*100:14.1f}% "
            f"{r/ceiling_test*100:13.1f}%"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "noise_ceiling.json").write_text(
        json.dumps(
            {
                "rho_obs": rho_obs,
                "n_train": n_train,
                "n_test": n_test,
                "k_noise_over_signal": float(k),
                "rho_train": float(rho_train),
                "rho_test": float(rho_test),
                "ceiling_naive_sqrt_rho_obs": float(ceiling_naive),
                "ceiling_correct_sqrt_rho_test": float(ceiling_test),
                "models": rows,
                "assumption": "animal-to-animal variability is iid noise averaging as 1/N; "
                "non-diminishing biological variability would raise the true ceiling",
            },
            indent=1,
        )
    )
    print(f"\nwrote {OUT / 'noise_ceiling.json'}")


if __name__ == "__main__":
    main()
