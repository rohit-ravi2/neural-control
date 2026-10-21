"""Reconcile Creamer's 0.140 (own shipped test STAMs) with 0.088 (our held-out events).

Changes one factor at a time, for each published Creamer model:
  target side  T_ship        shipped data/measured_stams.pkl["test"] (their split, their pipeline)
               T_auth_ours   their estimator (lu preprocessing, get_impulse_response_functions) on OUR test animals
               T_evt_raw     our event-pooled STAM (wm.pooled_stam) on our test animals, uncleaned data (= v1 / the 0.088)
               T_evt_clean   same, v2-cleaned data (what M1's 0.193 is scored against)
  model side   skip15        their calculate_stams: impulse IRF summed from frame 15 (window[0] used as a frame index)
               full          impulse IRF summed over all 60 post-stimulus frames (matches the measured window)
Also prints rho (measured train STAM -> each target) for scale. Linear model: (stim on - stim off) from any
state equals the impulse response, so the counterfactual definition reduces to the window choice.

    python reconcile_creamer.py   -> results/creamer_reconcile.json
"""
import json, pickle, sys, warnings

import numpy as np

import phase0_ceiling as p0
import train_worm as tw
import worm_data as wd
import worm_models as wm

sys.path.insert(0, str(wd.STUBS)); sys.path.insert(0, str(wd.REPO))
import lgssm_utilities as ssmu  # noqa: E402
import metrics as met  # noqa: E402

cell = wd.model_cell_ids(); n = len(cell); eye = np.eye(n, dtype=bool)
c = lambda a, b: float(met.nan_corr(a, b)[0])
names = wd.recording_names(); tr_n, te_n, src = tw.get_split(names)

shipped = pickle.load(open(wd.REPO / "data" / "measured_stams.pkl", "rb"))
for k in shipped:
    shipped[k][eye] = np.nan

def evt_stams(clean):
    tw.CLEAN = clean
    tr, te = tw.load_recs(tr_n), tw.load_recs(te_n)
    a, b = wm.event_list(tr), wm.event_list(te)
    return (wm.pooled_stam(tr, a, wm.measured_event_responses(tr, a), cell),
            wm.pooled_stam(te, b, wm.measured_event_responses(te, b), cell), len(b))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    T = {"T_ship": shipped["test"], "T_auth_ours": p0.authors_stams(te_n, cell)}
    R = {"T_ship": shipped["train"], "T_auth_ours": p0.authors_stams(tr_n, cell)}
    R["T_evt_raw"], T["T_evt_raw"], n_raw = evt_stams(False)
    R["T_evt_clean"], T["T_evt_clean"], n_clean = evt_stams(True)

out = {"split": src, "n_events_raw": n_raw, "n_events_clean": n_clean, "rho": {}, "models": {},
       "target_vs_target": {}}
for k in T:
    out["rho"][k] = c(R[k], T[k])
keys = list(T)
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        out["target_vs_target"][f"{a}~{b}"] = c(T[a], T[b])
for mn in ["connectome_constrained", "fully_connected", "shuffled_constrained"]:
    m = pickle.load(open(wd.REPO / "models" / f"{mn}.pkl", "rb"))
    irfs = ssmu.calculate_irfs(m, window=(15, 30))
    S = {"skip15": irfs[15:].sum(0) / m.sample_rate, "full": irfs.sum(0) / m.sample_rate}
    for s in S.values():
        s[eye] = np.nan
    out["models"][mn] = {w: {k: c(S[w], T[k]) for k in T} for w in S}

json.dump(out, open(p0.OUT.parent / "creamer_reconcile.json", "w"), indent=1)
print(json.dumps(out, indent=1))
