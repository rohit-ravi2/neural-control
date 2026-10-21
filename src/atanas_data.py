"""Atanas et al. 2023 (Flavell lab; DANDI:000776) spontaneous whole-brain recordings, freely moving, NeuroPAL-identified.
Source: data/atanas2023_nwb_extract/*.h5 (38 animals, ~1600 frames at 0.6 s).
Prepared for the observational->causal test: traces linearly resampled to Randi's 2 Hz grid, z-scored per neuron per
animal; labels kept only if exact (labels with '?' or blank -> unnamed); no stimulation (u = 0).
    python atanas_data.py   -> work/atanas_cache.npz (lists of arrays) + summary
"""
import glob, pathlib

import h5py
import numpy as np

SRC = str(pathlib.Path(__file__).resolve().parents[1] / "data/atanas2023_nwb_extract") + "/*.h5"
CACHE = pathlib.Path(__file__).resolve().parents[1].joinpath("work/atanas_cache.npz").expanduser()
RATE = 2.0


def build():
    X, IDS, NAMES = [], [], []
    for p in sorted(glob.glob(SRC)):
        f = h5py.File(p, "r")
        lab = [x.decode() if isinstance(x, bytes) else str(x) for x in f["neuron_labels"][:]]
        lab = ["" if ("?" in l or not l.strip()) else l.strip() for l in lab]
        ts = f["trace_timestamps"][:]; tr = f["traces"][:]
        grid = np.arange(ts[0], ts[-1], 1.0 / RATE)
        x = np.stack([np.interp(grid, ts, tr[:, i]) for i in range(tr.shape[1])], 1)
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        X.append(x.astype(np.float32)); IDS.append(np.array(lab)); NAMES.append(pathlib.Path(p).stem)
    np.savez(CACHE, **{f"x{i}": x for i, x in enumerate(X)}, **{f"id{i}": d for i, d in enumerate(IDS)}, names=np.array(NAMES))


def load():
    z = np.load(CACHE, allow_pickle=False)
    n = len(z["names"])
    return [(str(z["names"][i]), z[f"x{i}"], z[f"id{i}"]) for i in range(n)]


if __name__ == "__main__":
    build()
    recs = load()
    print(len(recs), "animals; frames at 2 Hz: median", int(np.median([r[1].shape[0] for r in recs])),
          "; named per animal median", int(np.median([(r[2] != "").sum() for r in recs])))
