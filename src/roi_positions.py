"""Per-animal ROI centroids in µm for every OSF recording column (DANDI 001075 green segmentation; 1:1 with OSF columns).
    python roi_positions.py -> work/roi_pos.npz  (key = recording name, value = (n_cols, 3) µm)
"""
import json, pathlib, re

import h5py
import numpy as np

import worm_data as wd

SEG = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data/randi2023_dandi001075_seg")
SCALE = np.array([0.42, 0.42, 1.22])
out, bad = {}, []
for a in json.load(open(f"{SEG}/seg_assets.json")):
    m = re.search(r"(sub-\d+)_ses-(\d{8})-(\d{6})", a["path"]); rec = f"pumpprobe_{m.group(2)}_{m.group(3)}"
    f = pathlib.Path(f"{SEG}/{m.group(1)}_seg.nwb")
    if not f.exists() or rec not in wd.recording_names():
        continue
    g = h5py.File(f, "r")["processing/ophys/PumpProbeGreenSegmentations/PumpProbeGreenPlaneSegmentation"]
    C = g["centroids"][:].astype(float) * SCALE
    npl = h5py.File(f, "r")["processing/ophys/NeuroPALSegmentations/NeuroPALPlaneSegmentation"]
    lab = [x.decode().strip() if isinstance(x, bytes) else str(x).strip() for x in npl["labels"][:]]; nid = list(npl["id"][:])
    def to_lab(v):
        try:
            k = int(float(v.decode() if isinstance(v, bytes) else v)); return lab[nid.index(k)] if k in nid else ""
        except ValueError:
            return ""
    roi_lab = [to_lab(v) for v in g["neuropal_ids"][:]]
    keep = [i for i, l in enumerate(roi_lab) if l not in wd.BLACKLIST]   # the cache drops AWCL/AWCR columns
    C = C[keep]
    osf = [str(x) for x in wd.load_recording(rec)["cell_ids"]]
    kl = [roi_lab[i] for i in keep]
    if len(C) == len(osf) and sum(a == b for a, b in zip(kl, osf) if b) != sum(1 for b in osf if b):
        bad.append((rec, "label mismatch")); continue
    ncol = len(osf)
    if len(C) != ncol:
        bad.append((rec, len(C), ncol)); continue
    out[rec] = C
np.savez(pathlib.Path(__file__).resolve().parents[1].joinpath("work/roi_pos.npz").expanduser(), **out)
print(len(out), "recordings with 1:1 positions; mismatched:", bad)
