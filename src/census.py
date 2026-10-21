"""Corpus census of the Creamer/Leifer 2025 dataset.

Reports the quantities that determine whether a held-out-perturbation-target
split is feasible: how many stimulation events exist, how many resolve to a
named neuron, and how often each target neuron is stimulated across recordings.
"""
from __future__ import annotations

import collections
import json
import pathlib

import numpy as np

from load_creamer import load_recording, recording_paths

OUT = pathlib.Path(__file__).resolve().parents[1].joinpath("results").expanduser()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    target_counts: collections.Counter[str] = collections.Counter()
    target_recordings: dict[str, set[str]] = collections.defaultdict(set)
    nan_fracs = []

    for path in recording_paths():
        rec = load_recording(path)
        named = rec.named_mask.sum()
        resolved = int(rec.resolved_stims.sum())
        named_targets = [t for t in rec.stim_targets() if t]
        for t in named_targets:
            target_counts[t] += 1
            target_recordings[t].add(rec.name)
        nan_frac = float(np.isnan(rec.traces).mean())
        nan_fracs.append(nan_frac)
        rows.append(
            dict(
                name=rec.name,
                n_frames=rec.n_frames,
                n_neurons=rec.n_neurons,
                n_named=int(named),
                n_stims=int(rec.stim_cells.size),
                n_stims_resolved=resolved,
                n_stims_named=len(named_targets),
                nan_frac=round(nan_frac, 4),
            )
        )

    n_rec = len(rows)
    tot_stim = sum(r["n_stims"] for r in rows)
    tot_named = sum(r["n_stims_named"] for r in rows)

    print(f"recordings                {n_rec}")
    print(
        "frames per recording      "
        f"min {min(r['n_frames'] for r in rows)}  "
        f"median {int(np.median([r['n_frames'] for r in rows]))}  "
        f"max {max(r['n_frames'] for r in rows)}"
    )
    print(
        "neurons per recording     "
        f"min {min(r['n_neurons'] for r in rows)}  "
        f"median {int(np.median([r['n_neurons'] for r in rows]))}  "
        f"max {max(r['n_neurons'] for r in rows)}"
    )
    print(
        "named neurons per rec     "
        f"min {min(r['n_named'] for r in rows)}  "
        f"median {int(np.median([r['n_named'] for r in rows]))}  "
        f"max {max(r['n_named'] for r in rows)}"
    )
    print(f"NaN fraction of traces    mean {np.mean(nan_fracs):.3f}")
    print(f"stimulation events        {tot_stim} total, {tot_named} resolve to a named neuron")
    print(f"distinct target neurons   {len(target_counts)}")
    print()
    print("targets stimulated in >= 5 distinct recordings:")
    multi = sorted(
        ((t, len(r)) for t, r in target_recordings.items() if len(r) >= 5),
        key=lambda x: -x[1],
    )
    print(f"  {len(multi)} neurons")
    for t, c in multi[:25]:
        print(f"    {t:8s} {c:3d} recordings, {target_counts[t]:3d} events")

    singletons = [t for t, r in target_recordings.items() if len(r) == 1]
    print(f"\ntargets stimulated in exactly 1 recording: {len(singletons)}")

    (OUT / "census_recordings.json").write_text(json.dumps(rows, indent=1))
    (OUT / "census_targets.json").write_text(
        json.dumps(
            {
                t: dict(events=target_counts[t], recordings=sorted(r))
                for t, r in sorted(target_recordings.items())
            },
            indent=1,
        )
    )
    print(f"\nwrote {OUT/'census_recordings.json'} and {OUT/'census_targets.json'}")


if __name__ == "__main__":
    main()
