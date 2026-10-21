"""Per-animal NeuroPAL label confidences (DANDI 001075 segmentation NWBs), keyed by OSF recording name.
    python id_confidence.py -> work/id_conf.json  {recording: {label: confidence}}
"""
import glob, json, pathlib, re

import h5py

SEG = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data/randi2023_dandi001075_seg")
out = {}
for a in json.load(open(f"{SEG}/seg_assets.json")):
    m = re.search(r"(sub-\d+)_ses-(\d{8})-(\d{6})", a["path"])
    f = pathlib.Path(f"{SEG}/{m.group(1)}_seg.nwb")
    if not f.exists():
        continue
    g = h5py.File(f, "r")["processing/ophys/NeuroPALSegmentations/NeuroPALPlaneSegmentation"]
    lab = [x.decode().strip() if isinstance(x, bytes) else str(x).strip() for x in g["labels"][:]]
    conf = {}
    for l, c in zip(lab, g["labels_confidences"][:]):
        if l:
            conf[l] = max(conf.get(l, -9), float(c))
    out[f"pumpprobe_{m.group(2)}_{m.group(3)}"] = conf
pathlib.Path(__file__).resolve().parents[1].joinpath("work/id_conf.json").expanduser().write_text(json.dumps(out))
print(len(out), "recordings with confidences")
