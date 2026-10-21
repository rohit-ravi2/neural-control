"""STEREOTYPY: how individual-specific is the C. elegans causal map?
    python stereotypy.py -> results/stereotypy.json
"""
import collections, json

import numpy as np

import train_worm as tw
import worm_data as wd
import worm_models as wm

import sys
CTRL = "--controls" in sys.argv
rng = np.random.default_rng(0)
names = wd.recording_names()
recs = tw.load_recs(names)
if CTRL:   # (a) gain control: z-score each neuron's trace per animal
    for r in recs:
        r.x = (r.x - np.nanmean(r.x, 0)) / (np.nanstd(r.x, 0) + 1e-6)
    cnt_names = collections.Counter(n for r in recs for n in set(r.ids) if n)
    COMMON = {n for n, c in cnt_names.items() if c >= 60}   # (d) frequently identified names
    print("common names (>=60 animals):", len(COMMON), flush=True)
date = [n.split("_")[1] for n in names]
CLASS = "--class-level" in sys.argv
POSONLY = "--pos-subset" in sys.argv
EXUM = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--exclude-um=")), None)
CROWD = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--uncrowded=")), None)
POS = dict(np.load(__import__("pathlib").Path(__file__).resolve().parents[1].joinpath("work/roi_pos.npz").expanduser())) if (POSONLY or EXUM or CROWD) else {}
TGTHIT = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--target-hit=")), None)
MINCONF = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--min-conf=")), None)
if MINCONF is not None:
    import json as _j, pathlib as _pl
    IDC = _j.loads(_pl.Path(__file__).resolve().parents[1].joinpath("work/id_conf.json").expanduser().read_text())
    print("recordings with ID confidences:", sum(n in IDC for n in names), "of", len(names), flush=True)
KNN = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--exclude-near=")), 0)
NEAR = {}
if KNN:   # exclude each target's KNN nearest neighbours in canonical atlas space (unit-free)
    import os, wormneuroatlas as wa_
    fpos = os.path.join(os.path.dirname(wa_.__file__), "data", "anatlas_neuron_positions.txt")
    lines = open(fpos).read().splitlines()
    pos_names = lines[0].lstrip("#").split(); XYZ = np.array([[float(v) for v in l.split()] for l in lines[1:] if l.strip()])
    for a_, nm in enumerate(pos_names):
        d = np.linalg.norm(XYZ - XYZ[a_], axis=1); NEAR[nm] = set(pos_names[b] for b in np.argsort(d)[1:KNN + 1])
    print(f"excluding each target's {KNN} nearest canonical neighbours ({len(NEAR)} positioned names)", flush=True)
E = []   # (animal, frame, target name, {responder name: value})
for a, r in enumerate(recs):
    if POS and names[a] not in POS:
        continue
    if POS:
        C = POS[names[a]]; Dm = np.linalg.norm(C[:, None] - C[None], axis=2); np.fill_diagonal(Dm, np.inf)
        crowded = (Dm.min(1) < CROWD) if CROWD else np.zeros(len(C), bool)
    for t, j in zip(*np.where(r.u == 1)):
        if not r.ids[j] or t - wm.PRE < 0 or t + wm.POST >= r.x.shape[0]:
            continue
        if MINCONF is not None:
            cf = IDC.get(names[a])
            if cf is None or cf.get(str(r.ids[j]), -9) < MINCONF:
                continue
        clip = r.x[t - wm.PRE:t + wm.POST]
        with np.errstate(all="ignore"):
            v = (np.nanmean(clip[wm.PRE:], 0) - np.nanmean(clip[:wm.PRE], 0)) * wm.POST
        if TGTHIT is not None and not (np.isfinite(v[j]) and v[j] / wm.POST >= TGTHIT):
            continue
        resp = {r.ids[i]: float(v[i]) for i in range(len(v)) if r.ids[i] and i != j and np.isfinite(v[i])
                and not (EXUM and Dm[i, j] < EXUM) and not (CROWD and crowded[i])
                and np.isfinite(clip[wm.PRE:, i]).mean() > 0.5 and (not CTRL or r.ids[i] in COMMON)}
        if KNN:
            resp = {k_: v_ for k_, v_ in resp.items() if k_ not in NEAR.get(str(r.ids[j]), set())}
        if MINCONF is not None:
            resp = {k_: v_ for k_, v_ in resp.items() if cf.get(k_, -9) >= MINCONF}
        tname = str(r.ids[j])
        if CLASS:   # collapse L/R (and D/V) to classes; average responders within class
            tname = wm.neuron_class(tname); agg = collections.defaultdict(list)
            for k_, v_ in resp.items():
                if wm.neuron_class(k_) != tname:
                    agg[wm.neuron_class(k_)].append(v_)
            resp = {k_: float(np.mean(v_)) for k_, v_ in agg.items()}
        E.append((a, int(t), tname, resp))
print("named-target events:", len(E), flush=True)
by_t = collections.defaultdict(list)
for k, e in enumerate(E):
    by_t[e[2]].append(k)


def r_pair(e1, e2):
    common = [n for n in e1[3] if n in e2[3] and n not in (e1[2], e2[2])]
    if len(common) < 10:
        return None
    x, y = np.array([e1[3][n] for n in common]), np.array([e2[3][n] for n in common])
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None


cls = collections.defaultdict(list); gap_rs = []
PT = collections.defaultdict(lambda: {"SA": [], "DA": []})
for tgt, ks in by_t.items():
    for ii in range(len(ks)):
        for jj in range(ii + 1, len(ks)):
            e1, e2 = E[ks[ii]], E[ks[jj]]
            r = r_pair(e1, e2)
            if r is None:
                continue
            if e1[0] == e2[0]:
                PT[tgt]["SA"].append(r)
                if CTRL and abs(e1[1] - e2[1]) < 120:   # (b) exclude repeats < 60 s apart
                    continue
                cls["ST-SA"].append(r); gap_rs.append((abs(e1[1] - e2[1]) / 2.0, r))
            else:
                PT[tgt]["DA"].append(r)
                cls["ST-DA"].append(r)
                cls["ST-DA same-date" if date[e1[0]] == date[e2[0]] else "ST-DA diff-date"].append(r)
# different-target pairs: sample to similar counts
by_a = collections.defaultdict(list)
for k, e in enumerate(E):
    by_a[e[0]].append(k)
for _ in range(60000):
    k1, k2 = rng.integers(len(E), size=2); e1, e2 = E[k1], E[k2]
    if e1[2] == e2[2] or k1 == k2:
        continue
    r = r_pair(e1, e2)
    if r is not None:
        cls["DT-DA" if e1[0] != e2[0] else "DT-SA"].append(r)
for a, ks in by_a.items():   # dedicated same-animal different-target sample
    if len(ks) < 2:
        continue
    for _ in range(80):
        k1, k2 = rng.choice(ks, 2, replace=False); e1, e2 = E[k1], E[k2]
        if e1[2] != e2[2]:
            r = r_pair(e1, e2)
            if r is not None:
                cls["DT-SA"].append(r)
m = {k: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v))), len(v)) for k, v in cls.items()}
within, between = m["ST-SA"][0] - m["DT-SA"][0], m["ST-DA"][0] - m["DT-DA"][0]
g = np.array(gap_rs)
bins = [0, 60, 120, 240, 480, 2000]
gap_curve = [(f"{bins[b]}-{bins[b+1]}s", float(g[(g[:, 0] >= bins[b]) & (g[:, 0] < bins[b + 1]), 1].mean()) if ((g[:, 0] >= bins[b]) & (g[:, 0] < bins[b + 1])).any() else None,
              int(((g[:, 0] >= bins[b]) & (g[:, 0] < bins[b + 1])).sum())) for b in range(len(bins) - 1)]
# bootstrap the index over targets
tg = list(by_t)
tm = [(np.mean(v["SA"]), np.mean(v["DA"])) for v in PT.values() if len(v["SA"]) >= 2 and len(v["DA"]) >= 5]
tm = np.array(tm)
boot = []
for _ in range(1000):
    b = tm[rng.integers(0, len(tm), len(tm))]
    boot.append((b[:, 1].mean() - m["DT-DA"][0]) / (b[:, 0].mean() - m["DT-SA"][0]))
target_matched = {"n_targets": int(len(tm)), "SA": float(tm[:, 0].mean()), "DA": float(tm[:, 1].mean()),
                  "index": float((tm[:, 1].mean() - m["DT-DA"][0]) / (tm[:, 0].mean() - m["DT-SA"][0])),
                  "index_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
out = {"target_matched": target_matched, "classes": m, "within": within, "between": between, "stereotypy_index": between / within, "ST-SA_vs_gap": gap_curve}
print(json.dumps(out, indent=1))
json.dump(out, open("../results/stereotypy" + ("_controls" if CTRL else "") + (f"_near{KNN}" if KNN else "") + ("_class" if CLASS else "") + (f"_conf{MINCONF}" if MINCONF is not None else "") + (f"_hit{TGTHIT}" if TGTHIT is not None else "") + ("_pos" if POS else "") + (f"_ex{EXUM}" if EXUM else "") + (f"_crowd{CROWD}" if CROWD else "") + ".json", "w"), indent=1)
