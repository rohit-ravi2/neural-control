"""PEPTIDE enrichment variant: are wt connections lost in unc-31 enriched for peptide/GPCR matches?"""
import json

import numpy as np
from scipy.stats import fisher_exact
import statsmodels.api as sm

import peptide_did as pd0
import peptide_did2 as pd2

na, N = pd0.na, pd0.N
qwt = na.get_signal_propagation_q(strain="wt"); qeq = na.get_signal_propagation_q_eq(strain="unc31")
anat = (na.get_chemical_synapses() > 0) | (na.get_gap_junctions() > 0)
sel = (qwt < 0.05) & np.isfinite(qeq) & ~np.eye(N, dtype=bool)
lost = sel & (qeq < 0.05); kept = sel & ~(qeq < 0.05)
out = {"n_wt_sig_measured_in_unc31": int(sel.sum()), "n_lost": int(lost.sum()), "n_kept": int(kept.sum())}
for name, Pm in (("P", pd0.P), ("Pstrong", pd2.Pstrong)):
    has = Pm > 0
    tab = [[int((has & lost).sum()), int((~has & lost).sum())], [int((has & kept).sum()), int((~has & kept).sum())]]
    orr, p = fisher_exact(tab)
    y = lost[sel].astype(float); X = sm.add_constant(np.c_[has[sel].astype(float), anat[sel].astype(float)])
    try:
        fit = sm.Logit(y, X).fit(disp=0); lo = {"coef_pep": float(fit.params[1]), "p_pep": float(fit.pvalues[1]),
                                               "coef_anat": float(fit.params[2]), "p_anat": float(fit.pvalues[2])}
    except Exception as e:
        lo = {"error": str(e)}
    out[name] = {"table[[lost&pep, lost&no],[kept&pep, kept&no]]": tab, "odds_ratio": float(orr), "fisher_p": float(p), "logit": lo}
print(json.dumps(out, indent=1))
json.dump(out, open("../results/peptide_enrich.json", "w"), indent=1)
