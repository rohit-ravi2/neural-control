"""Angle 3 (exploratory): individuality of SPONTANEOUS functional coupling (Atanas 2023, 38 animals, class-level).
Per animal: correlation matrices on the first and second half of the recording. For each neuron class i, its coupling profile (correlations with the
other classes). within = corr(profile_i half1, profile_i half2) same animal; across = corr(profile_i animal a half1, profile_i animal b half2).
Index = across / within (1 = fully stereotyped). Also per class, and vs time-lag between halves (gap) as a drift check.
    python spont_individuality.py -> results/spont_individuality.json
"""
import collections, itertools, json

import numpy as np

import atanas_data as ad
import worm_models as wm

import glob, h5py, sys
A = ad.load()
BEHAV = "--behav" in sys.argv
FWD = "--fwd" in sys.argv
if FWD:   # state-matched: keep only forward-locomotion frames (velocity > 0), per animal
    files = sorted(glob.glob(ad.SRC)); A2 = []
    for (nm, x, ids), f in zip(A, files):
        g = h5py.File(f, "r"); ts = g["trace_timestamps"][:]; grid = np.arange(ts[0], ts[-1], 1 / ad.RATE)[:x.shape[0]]
        v = np.interp(grid, g["behavior/velocity/timestamps"][:], np.nan_to_num(g["behavior/velocity/data"][:]))
        A2.append((nm, x[:len(grid)][v > 0], ids))
    A = A2
if BEHAV:   # regress behaviour (velocity, angular velocity, head curvature, reversal) out of every trace, per animal
    files = sorted(glob.glob(ad.SRC))
    A2 = []
    for (nm, x, ids), f in zip(A, files):
        g = h5py.File(f, "r"); ts = g["trace_timestamps"][:]; grid = np.arange(ts[0], ts[-1], 1 / ad.RATE)[:x.shape[0]]
        B = []
        for k in ("velocity", "angular_velocity", "head_curvature", "reversal_events"):
            bt = g[f"behavior/{k}/timestamps"][:]; bd = g[f"behavior/{k}/data"][:].astype(float)
            B.append(np.interp(grid, bt, np.nan_to_num(bd)))
        B = np.c_[np.ones(len(grid)), np.stack(B, 1)]; B = (B - B.mean(0)) / (B.std(0) + 1e-9); B[:, 0] = 1
        beta = np.linalg.lstsq(B, x[:len(grid)], rcond=None)[0]
        A2.append((nm, x[:len(grid)] - B @ beta, ids))
    A = A2


def halves(x, ids, gap=0):
    T = x.shape[0]; h = (T - gap) // 2
    out = []
    for seg in (x[:h], x[h + gap:]):
        cl = collections.defaultdict(list)
        for k, nm in enumerate(ids):
            if nm:
                cl[wm.neuron_class(nm)].append(seg[:, k])
        names = sorted(cl); X = np.stack([np.mean(cl[c], 0) for c in names], 1)
        C = np.corrcoef(X.T); out.append(dict(zip(names, range(len(names)))) | {"_C": C, "_names": names})
    return out


H = [halves(x, ids) for _, x, ids in A]
within, across = collections.defaultdict(list), collections.defaultdict(list)


def prof(h, c, partners):
    return np.array([h["_C"][h[c], h[p]] for p in partners])


for a, Ha in enumerate(H):
    for c in Ha[0]["_names"]:
        partners = [p for p in Ha[0]["_names"] if p != c and p in Ha[1]["_names"]]
        if len(partners) >= 10:
            within[c].append(np.corrcoef(prof(Ha[0], c, partners), prof(Ha[1], c, partners))[0, 1])
for a, b in itertools.permutations(range(len(H)), 2):
    if (a * 7 + b) % 5:            # subsample animal pairs for speed
        continue
    for c in set(H[a][0]["_names"]) & set(H[b][1]["_names"]):
        partners = [p for p in H[a][0]["_names"] if p != c and p in H[b][1]["_names"]]
        if len(partners) >= 10:
            across[c].append(np.corrcoef(prof(H[a][0], c, partners), prof(H[b][1], c, partners))[0, 1])
w = np.mean([v for vs in within.values() for v in vs]); x = np.mean([v for vs in across.values() for v in vs])
per = {c: {"within": float(np.mean(within[c])), "across": float(np.mean(across[c])), "index": float(np.mean(across[c]) / np.mean(within[c])),
           "n_within": len(within[c])} for c in within if len(within[c]) >= 8 and len(across[c]) >= 20}
out = {"within_mean": float(w), "across_mean": float(x), "stereotypy_index": float(x / w), "n_classes": len(per),
       "most_individual": sorted(per, key=lambda c: per[c]["index"])[:10], "most_stereotyped": sorted(per, key=lambda c: -per[c]["index"])[:10], "per_class": per}
print(json.dumps({k: v for k, v in out.items() if k != "per_class"}, indent=1))
json.dump(out, open("../results/spont_individuality" + ("_behav" if BEHAV else "") + ("_fwd" if FWD else "") + ".json", "w"), indent=1)
