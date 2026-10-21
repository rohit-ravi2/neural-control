"""Select the residual-loss lambda on the INNER validation animals (60/20 split of the training animals, seed 555), residual target, state-free.
    python resid_select.py -> results/resid_select.json
"""
import json

import numpy as np

import resid_lib as rl
import train_worm as tw
import worm_data as wd

tr_n, _, _ = tw.get_split(wd.recording_names())
val = sorted(list(np.random.default_rng(555).permutation(tr_n))[60:])
D = rl.build(val, ceiling=False)
out = {"0 (next-step reference, stage-3 inner)": rl.score(D, "../results/worm_inner3/M1_class_s0_I3_gru_class_l0.pt")}
print("ref", json.dumps(out["0 (next-step reference, stage-3 inner)"]), flush=True)
for lam in ("0.03", "0.1", "0.3"):
    out[lam] = rl.score(D, f"../results/worm_resid_inner/M1_class_s0_RI_l{lam}.pt")
    print(lam, json.dumps(out[lam]), flush=True)
json.dump(out, open("../results/resid_select.json", "w"), indent=1)
