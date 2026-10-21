"""Cache the authors'-estimator STAMs for our split: adds T_auth_train to results/phase0/targets_test.npz."""
import warnings

import numpy as np

import phase0_ceiling as p0
import train_worm as tw
import worm_data as wd

f = p0.OUT / "targets_test.npz"
T = dict(np.load(f))
tr_n, _, _ = tw.get_split(wd.recording_names())
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    T["T_auth_train"] = p0.authors_stams(tr_n, wd.model_cell_ids())
np.savez(f, **T)
print({k: v.shape for k, v in T.items()})
