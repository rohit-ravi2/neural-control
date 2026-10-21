"""STEREOTYPY vs DISTANCE. All controls stacked.
    python stereo_distance.py -> results/stereo_distance_relaxed.json
"""
import collections, json, pathlib

import numpy as np

import train_worm as tw
import worm_data as wd
import worm_models as wm

rng = np.random.default_rng(0)
names = wd.recording_names()
POS = dict(np.load(pathlib.Path(__file__).resolve().parents[1].joinpath("work/roi_pos.npz").expanduser()))
IDC = json.loads(pathlib.Path(__file__).resolve().parents[1].joinpath("work/id_conf.json").expanduser().read_text())
BINS = [(0, 10), (10, 20), (20, 1e9)]
recs = tw.load_recs(names)
E = []   # (animal, frame, target, {bin: {responder: value}})
for a, r in enumerate(recs):
    if names[a] not in POS or names[a] not in IDC:
        continue
    x = (r.x - np.nanmean(r.x, 0)) / (np.nanstd(r.x, 0) + 1e-6)
    C = POS[names[a]]; cf = IDC[names[a]]
    for t, j in zip(*np.where(r.u == 1)):
        tn = str(r.ids[j])
        if not tn or t - wm.PRE < 0 or t + wm.POST >= x.shape[0]:
            continue
        clip = x[t - wm.PRE:t + wm.POST]
        with np.errstate(all="ignore"):
            v = np.nanmean(clip[wm.PRE:], 0) - np.nanmean(clip[:wm.PRE], 0)
        if False:
            continue
        byb = {b: {} for b in range(len(BINS))}
        for i in range(len(v)):
            nm = str(r.ids[i])
            if not nm or i == j or not np.isfinite(v[i]) or np.isfinite(clip[wm.PRE:, i]).mean() <= 0.5:
                continue
            d = np.linalg.norm(C[i] - C[j])
            for b, (lo, hi) in enumerate(BINS):
                if lo <= d < hi:
                    byb[b][nm] = float(v[i])
        E.append((a, int(t), tn, byb))
print("events:", len(E), flush=True)


import sys
SNR = "--snr" in sys.argv


def rp(d1, d2, excl, minn=5):
    common = [k for k in d1 if k in d2 and k not in excl]
    if len(common) < minn:
        return None
    if SNR:   # keep the strongest responders of this pair (top 30%, at least 5)
        mag = np.array([abs(d1[k]) + abs(d2[k]) for k in common])
        keep = max(5, int(np.ceil(0.3 * len(common))))
        common = [common[i] for i in np.argsort(-mag)[:keep]]
    x, y = np.array([d1[k] for k in common]), np.array([d2[k] for k in common])
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None


by_t = collections.defaultdict(list)
for k, e in enumerate(E):
    by_t[e[2]].append(k)
out = {"n_events": len(E), "bins": {}}
for b, (lo, hi) in enumerate(BINS):
    cls = collections.defaultdict(list); PT = collections.defaultdict(lambda: {"SA": [], "DA": []})
    for tgt, ks in by_t.items():
        for ii in range(len(ks)):
            for jj in range(ii + 1, len(ks)):
                e1, e2 = E[ks[ii]], E[ks[jj]]
                if e1[0] == e2[0] and abs(e1[1] - e2[1]) < 120:
                    continue
                r = rp(e1[3][b], e2[3][b], (e1[2], e2[2]))
                if r is None:
                    continue
                key = "SA" if e1[0] == e2[0] else "DA"
                cls["ST-" + key].append(r); PT[tgt][key].append(r)
    for _ in range(40000):
        k1, k2 = rng.integers(len(E), size=2); e1, e2 = E[k1], E[k2]
        if e1[2] == e2[2] or k1 == k2:
            continue
        r = rp(e1[3][b], e2[3][b], (e1[2], e2[2]))
        if r is not None:
            cls["DT-SA" if e1[0] == e2[0] else "DT-DA"].append(r)
    m = {k: [float(np.mean(v)), float(np.std(v) / np.sqrt(len(v))), len(v)] for k, v in cls.items()}
    tm = np.array([(np.mean(v["SA"]), np.mean(v["DA"])) for v in PT.values() if len(v["SA"]) >= 1 and len(v["DA"]) >= 3])
    dtsa, dtda = m.get("DT-SA", [0])[0], m.get("DT-DA", [0])[0]
    idx = lambda t: (t[:, 1].mean() - dtda) / (t[:, 0].mean() - dtsa)
    boots = [idx(tm[rng.integers(0, len(tm), len(tm))]) for _ in range(1000)] if len(tm) > 3 else [np.nan]
    out["bins"][f"{lo}-{hi if hi < 1e8 else 'inf'}um"] = {"classes": m, "n_targets": int(len(tm)),
        "index_target_matched": float(idx(tm)) if len(tm) > 3 else None, "ci95": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]}
    o = out["bins"][f"{lo}-{hi if hi < 1e8 else 'inf'}um"]
    fx = lambda v: "nan" if v is None or not np.isfinite(v) else f"{v:.3f}"
    print(f"{lo:>3}-{hi if hi < 1e8 else 'inf'} µm: ST-SA {fx(m.get('ST-SA', [np.nan])[0])} (n={m.get('ST-SA', [0, 0, 0])[2]})  ST-DA {fx(m.get('ST-DA', [np.nan])[0])} (n={m.get('ST-DA', [0, 0, 0])[2]})  "
          f"DT-SA {fx(dtsa)}  DT-DA {fx(dtda)}  index {fx(o['index_target_matched'])} [{fx(o['ci95'][0])}, {fx(o['ci95'][1])}] ({o['n_targets']} targets)", flush=True)
json.dump(out, open("../results/stereo_distance_relaxed" + ("_snr" if SNR else "") + ".json", "w"), indent=1)
