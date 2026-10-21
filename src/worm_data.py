"""Preprocess the Creamer/Leifer 2025 OSF recordings exactly as the authors' pipeline did.

The OSF release (osf.io/qxhjd) ships RAW green-channel traces (`francesco_green`
with nan_mask applied), not the `funcon_preprocessed_data.pkl` that load_data()
consumes. The txt->pkl step is commented out in the released create_data_set.py,
so it is replicated here by calling the authors' own lu.preprocess_data with the
parameters in submission_params/create_data_set.yml:
    start_index=120, correct_photobleach=True, filter_size=2, upsample_factor=1
Inputs: inputs[stim_frame, stim_cell] = 1 for stim events with a resolved
(non-negative) column index, as in the commented-out original.

Per recording, the cache stores
    emissions   (T, N_rec)  preprocessed, recording's own columns (named + unnamed)
    inputs      (T, N_rec)
    cell_ids    (N_rec,)    "" for unidentified
and load_aligned() maps named columns onto a common identity axis (default: the
154 cell_ids of the published models, AWCL/AWCR blacklisted as in load_data).
"""
from __future__ import annotations

import multiprocessing as mp
import pathlib
import pickle
import sys
import warnings

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1].joinpath("work/Creamer_LDS_2024").expanduser()
STUBS = pathlib.Path(__file__).resolve().parents[1].joinpath("src/stubs").expanduser()
RAW = pathlib.Path(__file__).resolve().parents[1].joinpath("work/creamer_lds/2025_creamer_lds").expanduser()
CACHE = pathlib.Path(__file__).resolve().parents[1].joinpath("work/worm_cache").expanduser()
BLACKLIST = ("AWCL", "AWCR")
SAMPLE_RATE = 2

sys.path.insert(0, str(STUBS))
sys.path.insert(0, str(REPO))


def recording_names() -> list[str]:
    return sorted(p.name for p in RAW.iterdir() if p.is_dir())


def _preprocess_one(name: str) -> str:
    import loading_utilities as lu  # noqa: E402  (after sys.path setup)

    out = CACHE / f"{name}.npz"
    if out.exists():
        return name
    d = RAW / name
    e = np.loadtxt(d / "neural_data.txt", delimiter=",", ndmin=2)
    mask = np.loadtxt(d / "nan_mask.txt", delimiter=",", ndmin=2).astype(bool)
    e[mask] = np.nan
    ids = [s.strip().strip('"') for s in open(d / "cell_ids.txt").read().splitlines()]
    cells = np.atleast_1d(np.loadtxt(d / "stim_cell_indicies.txt", dtype=int))
    frames = np.atleast_1d(np.loadtxt(d / "stim_frame_indicies.txt", dtype=int))
    inputs = np.zeros_like(e)
    ok = cells >= 0
    inputs[frames[ok], cells[ok]] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        em, inp = lu.preprocess_data(e, inputs, start_index=120, correct_photobleach=True,
                                     filter_size=2, upsample_factor=1)
    keep = [i for i, c in enumerate(ids) if c not in BLACKLIST]
    np.savez_compressed(out, emissions=em[:, keep].astype(np.float32), inputs=inp[:, keep].astype(np.int8),
                        cell_ids=np.array([ids[i] for i in keep]),
                        n_stim_total=len(cells), n_stim_resolved=int(ok.sum()))
    return name


def build_cache(workers: int = 8) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    names = recording_names()
    with mp.Pool(workers) as pool:
        for i, n in enumerate(pool.imap_unordered(_preprocess_one, names), 1):
            print(f"  {i}/{len(names)} {n}", flush=True)


def model_cell_ids() -> list[str]:
    with open(REPO / "models" / "connectome_constrained.pkl", "rb") as fh:
        return [str(c) for c in pickle.load(fh).cell_ids]


def load_recording(name: str) -> dict:
    z = np.load(CACHE / f"{name}.npz")
    return {k: z[k] for k in z.files}


def load_aligned(names: list[str], cell_ids: list[str] | None = None):
    """Emissions/inputs on a common identity axis (NaN / 0 where a neuron is absent)."""
    cell_ids = cell_ids or model_cell_ids()
    col = {c: j for j, c in enumerate(cell_ids)}
    ems, ins = [], []
    for n in names:
        r = load_recording(n)
        e = np.full((r["emissions"].shape[0], len(cell_ids)), np.nan, dtype=np.float32)
        u = np.zeros_like(e, dtype=np.int8)
        for i, c in enumerate(r["cell_ids"]):
            if c and c in col:
                e[:, col[c]] = r["emissions"][:, i]
                u[:, col[c]] = r["inputs"][:, i]
        ems.append(e)
        ins.append(u)
    return ems, ins, cell_ids


if __name__ == "__main__":
    build_cache()
