# neural-control

How much of a *C. elegans* neuron's response to stimulating another neuron is set by who the neuron is, and how much by
how it is wired? I worked on this with the whole-brain optogenetic perturbation atlas of Randi et al. (2023): 110 animals,
5,618 stimulation events. A model that shares one readout per named neuron across animals predicts the causal response
map of unseen animals at r = 0.451 in 5-fold cross-validation, 77% of the noiseless ceiling (0.586). The connectome's
predictive power for the same maps turns out to be confined to neurons within about 10 µm of the stimulated one.

## Background

Creamer, Leifer and Pillow (2025) fit a connectome-constrained linear dynamical system to these recordings and report
that it reaches 92% of the data's train-to-test reproducibility on the stimulus-triggered average matrix (STAM: for each
responder/target pair, the average response of the responder after the target is stimulated). That result is the main
evidence that the anatomical wiring diagram predicts causal signal flow in the worm. I reproduced it first, then asked
what an identity-indexed model that never sees the connectome can do on the same maps.

## Results

### 1. Reproducing Creamer et al.

From their published models and shipped test matrices, using their own metric code (`src/reproduce_creamer.py` →
`results/creamer_reproduction.json`), no refitting:

| model | STAM r | fraction of train→test reproducibility (0.1734) |
|---|---|---|
| connectome-constrained | 0.1403 | 80.9% |
| fully connected | 0.1196 | 69.0% |
| shuffled connectome | 0.0403 | 23.3% |

Every directional claim reproduces (connectome-constrained far above shuffled). The 92% does not: I get 80.9%, and
the gap is mostly in the ceiling (the shipped matrices are 154 × 154; the paper says 156 neurons). On the STAM metric
the connectome-constrained model does not significantly beat the unconstrained one (+0.021, paired bootstrap p = 0.29,
`results/model_comparisons.json`).

### 2. An identity-indexed readout, same estimator and split

`src/worm_models.py` defines one architecture (a shared GRU core) where only the key that indexes the per-neuron
encoder, stimulus and decoder tables changes. M1 keys them by neuron name (or class, e.g. AVAL/AVAR → AVA), shared across
animals. M2 is M1 with names shuffled independently in each animal. Nothing in the model sees the connectome.

Scored with the authors' STAM estimator on my 30 held-out test animals (`src/reconcile_m1.py`,
`results/creamer_reconcile_m1.json`, 5 seeds, zero-shot on animals the model never saw):

| model | r |
|---|---|
| M1, neuron-level | 0.223 ± 0.013 |
| M1, class-level | 0.248 ± 0.007 |
| Creamer fully connected | 0.153 |
| Creamer connectome-constrained | 0.105 |
| Creamer shuffled connectome | 0.050 |
| train→test reproducibility of the measured map | 0.188 |

The published Creamer weights were fit on a split that may include some of these 30 animals, so this comparison is
tilted toward Creamer. M1 also exceeds the measured map's own reproducibility, i.e. it denoises. Shuffling identities
(M2) drops it to ≈ 0 (`results/worm_v2/`). `src/reconcile_creamer.py` accounts for the difference between Creamer's
0.140 on their split and 0.105 on mine: −0.035 from the split, −0.017 from the estimator.

Conditioning on the animal's ongoing state contributes nothing here: with every burn-in observation hidden, M1 scores
0.225 / 0.250 (`results/m1_state_ablation.json`). The gain is in the learned, state-independent effect map.

### 3. State-free scoring and the ceiling

The STAM has no sham control. A model trained with an event-level response loss can predict part of each event's
post − pre change from the responder's own pre-stimulus activity, which is not causal. A model with shuffled identities
shows it plainly: 0.215 with the real burn-in, 0.004 with a blind one (`results/blind_final.json`). So every headline
number below is state-free (`--eval-blind`): the model predicts an impulse response from rest, which is what a STAM is.

The noiseless ceiling is estimated by split-half reliability plus Spearman-Brown (`src/headroom.py`, `src/ceiling_all.py`).

| evaluation | ceiling | identity model (state-free) | shuffled identities | Creamer connectome-constrained |
|---|---|---|---|---|
| fixed split, 30 test animals | 0.44 | 0.278 (two-model ensemble, 63%) | 0.004 | 0.105 |
| 5-fold CV over all 110 animals | 0.586 | **0.451 (77%)** | 0.001 | 0.127 |

Sources: `results/blind_final.json`, `results/headroom.json`, `results/cv_combine_worm_causal_final.json`,
`results/cv_combine_worm_causal_cvshuf.json`, `results/ceiling_all.json`. The CV score is a 2-seed ensemble of a
class-level GRU (hidden 128, open loop). Two caveats: three hyperparameters were chosen on an inner split of training
animals, some of which land in CV test folds (the leakage-free fixed split agrees in direction); and Creamer's published
weights were fit on 80 of the 110 CV animals, which favours Creamer. With more training animals per stimulated pair
(88 in CV vs 80 on the fixed split) the fraction of ceiling rises from 63% to 77%, so the remaining headroom looks
data-limited rather than capacity-limited.

### 4. The connectome predicts only the near field

Splitting STAM entries by the distance between responder and stimulated neuron (per-animal NeuroPAL positions,
`src/roi_positions.py`), on Creamer's own shipped test matrices (`src/near_far_cc.py` → `results/near_far_cc.json`):

| | near (< 10 µm) | far (≥ 15 µm) |
|---|---|---|
| connectome-constrained | 0.214 | 0.008 |
| shuffled connectome | −0.005 | 0.009 |
| fully connected | 0.191 | 0.013 |

All of the connectome's advantage over a shuffled connectome sits within 10 µm (finer strata in
`results/near_dissect.json`: 5–10 µm 0.206 vs −0.016; ≥ 15 µm 0.008 vs 0.009). Inside the near field the wiring
predicts beyond spatial adjacency alone: a distance-preserving shuffle of the anatomical connectome gives p = 0.002
(`src/dist_shuffle.py` → `results/dist_shuffle.json`). The far field is not empty: the state-free identity ensemble
predicts it at 0.125, against 0.065 for the measured map's own reproducibility and 0.000 for the connectome-constrained
model on my split (`results/blind_final.json`, `results/near_far.json`). So there is weak but real long-range,
identity-specific causal structure that the connectome does not see.

This reconciles "anatomy predicts propagation poorly" (Randi et al. 2023) with "a connectome-constrained model captures
most of the reproducible signal" (Creamer et al. 2025): they hold in different spatial regimes.

### 5. Never-trained neurons: expression vs wiring as a prior

For neuron classes the model has never been trained on, can a per-neuron feature vector stand in for the learned table
row? I compared CeNGEN single-cell expression profiles with connectome features, holding out each of 80 classes once
(5 folds).

- With a feature-to-row MLP (`results/worm_p3b/summary.json`), held-out entries score 0.106 (expression) and 0.076
  (connectome), against 0.013 for a mean-row fallback.
- With a non-parametric version of the same question, where a held-out neuron's map is the similarity-weighted average of
  its nearest training neurons' measured maps (`src/knn_prior.py` → `results/knn_prior.json`), the two priors tie:
  0.137 (expression, k = 40) vs 0.134 (connectome, k = 40), against 0.031 for random features.

I read this as a tie: both expression and wiring carry real information about an unseen neuron's causal map, and the
MLP gap reflects how that model used the connectome features, not a difference in the information itself.

### 6. Nerve-ring membrane contact predicts the causal map beyond distance and synapses

Head synapses form in the nerve-ring neuropil, so the distance between cell bodies is not the geometry that wiring
actually happens in. I added a third structural layer, membrane contact between neurites in the nerve ring (log contact
area from the adult and L4 nerve-ring adjacency tables), and let three layers compete for the causal map: soma distance
D, contact C, and synapses A (log chemical + log gap). The statistic is a partial Spearman correlation with the
authors'-estimator map over all 110 animals (12,966 responder/target pairs among the 125 cells covered by the contact
tables), `geometry/contact_geometry.py` → `geometry/results/contact_geometry.json`, adult contact:

| partial correlation with the causal map | value |
|---|---|
| contact, given distance and synapses | **0.040** (p = 0.003, distance-preserving null) |
| synapses, given distance | 0.009 |
| synapses, given distance and contact | −0.015 |
| soma distance, given contact and synapses | 0.291 |

So once you know which neurites touch, the synaptic connectome adds nothing to the prediction, while contact still
adds signal that a distance-preserving reshuffle of the contact matrix (300 permutations; p = 0.003 is the smallest
attainable) does not reproduce. Ninety-one percent of synaptically connected pairs are in contact, but only 52% of
contacting pairs have an annotated synapse. Cell-body distance itself keeps most of its predictive power (0.29), which is
the soma-proximity effect discussed in section 4.

The contact effect replicates on the L4 contact table (0.038, p = 0.003) and on the Randi et al. ΔF/F atlas from
`wormneuroatlas` (0.052, p = 0.003); on Creamer's shipped test matrices alone it is 0.024 (p = 0.066). Stress tests in
`geometry/contact_stress.py` → `geometry/results/contact_stress.json`: it survives a degree-preserving rewiring of the
contact graph (0.040, p = 0.005), per-neuron main effects (total contact and leave-one-out row/column means; 0.034,
p = 0.005) and dropping same-class pairs (0.038, p = 0.005). Restricted to synapse-free pairs it is weaker (0.025,
p = 0.035), and in far-field synapse-free pairs it is not detectable. It also adds a small, significant amount beyond the
best identity model's cross-validated prediction (0.026, p = 0.015; `geometry/contact_vs_model.py`). Whether the signal
needs neuropeptide release is not resolved: in *unc-31* animals the effect is the same size as in wild type on the
same pairs, with a CI that is wide on both sides (`geometry/results/contact_unc31.json`).

Other scripts in `geometry/` are exploratory and I am not making claims from them here: `mask_fit.py` (linear
propagation fit with W restricted to synapse, contact, nearest-neighbour or random supports), `contact_knn.py`
(contact profiles as a prior for unseen neurons), `wiring_economy.py` (wire-length-optimal placement vs the atlas), and
`simplicity_bias.py` / `witvliet_simplicity.py` (motif enrichment vs motif symmetry in the connectome and across the
eight Witvliet et al. EM datasets).

## Other analyses in the repo

These are in `src/` with their outputs in `results/`; they are exploratory and I am not making headline claims from them.

- `stereotypy.py`, `stereo_distance*.py`: how individual each animal's causal map is, by distance.
- `obs_*.py`: whether spontaneous activity (Atanas et al. 2023, and the same preparation) predicts optogenetic responses.
- `peptide_*.py`: whether ligand–receptor matching (the neuropeptide "wireless connectome") predicts propagation.
- `ranking.py`: retrospective ranking of which neuron to stimulate to drive a goal neuron, on held-out animals.
- `forecast*.py`, `spont_*.py`: forecasting spontaneous dynamics in new animals.

## How to run

Scripts are run from `src/`; paths resolve relative to the repo root (`work/` for downloaded code and caches, `data/`
for raw datasets, `results/` for outputs).

```bash
pip install -r requirements.txt
git clone https://github.com/Nondairy-Creamer/Creamer_LDS_2024 work/Creamer_LDS_2024
# put the OSF recordings in work/creamer_lds/2025_creamer_lds (see Data)
cd src
python reproduce_creamer.py      # -> results/creamer_reproduction.json (published models only, no training)
python compare_models.py         # paired bootstrap between Creamer's models
for s in 0 1 2 3 4; do for l in neuron class; do
  WORM_OUT=../results/worm_v2 python train_worm.py --scheme M1 --level $l --seed $s; done; done
python reconcile_m1.py           # M1 vs Creamer on one estimator; caches results/phase0/targets_test.npz
python obs_baselines.py          # among other things, adds the all-110-animal target used by cv_combine.py
./run_jobs.sh worm_causal_final ../results/worm_causal_final/jobs.txt   # state-free models, fixed split + 5-fold CV (GPU)
CVDIR=worm_causal_final python cv_combine.py
python near_far_cc.py            # -> results/near_far_cc.json
python knn_prior.py              # -> results/knn_prior.json (CPU, no training)
cd ..
python geometry/contact_geometry.py   # -> geometry/results/contact_geometry.json (CPU, a few minutes)
python geometry/contact_stress.py     # -> geometry/results/contact_stress.json
# contact_vs_model.py also needs the CV prediction files from the worm_causal_final run above
```

The 80/30 split is fixed in `results/phase0/split.json`. Each script's docstring gives its command line and output, and
says which earlier outputs it reads. A single GRU run took 7 to 20 minutes on one GPU.
`src/stubs/mpi4py` is an import-only stub: the authors' evaluation code imports mpi4py but only fitting uses it.
Each `results/<run>/jobs.txt` lists the exact `train_worm.py` arguments for that run, one job per line.

## Repo layout

```
src/worm_models.py      shared-core models (M0 per-animal, M1 identity, M2 shuffled, M3/M4 feature priors)
src/train_worm.py       training and zero-/few-shot evaluation driver
src/worm_data.py        preprocessing of the OSF recordings with the authors' own pipeline
src/load_creamer.py     loader for the processed recordings
src/connectome.py       anatomical connectome features
src/expression.py       CeNGEN expression features
src/phase0*_ceiling.py  split, targets and noise-ceiling checks
src/reproduce_creamer.py, reconcile_*.py   reproduction and reconciliation with Creamer et al.
src/near_far*.py, near_dissect.py, dist_shuffle.py   locality analyses
src/knn_*.py            non-parametric priors for unseen neurons
results/                JSON outputs of every script and training run (checkpoints not included)
geometry/               nerve-ring contact geometry analyses; outputs in geometry/results/
```

## Data

None of the data is in this repo.

- Creamer/Leifer processed recordings: OSF project qxhjd (https://osf.io/qxhjd), into `work/creamer_lds/2025_creamer_lds/`.
- Creamer et al. code, published models and anatomical connectome: https://github.com/Nondairy-Creamer/Creamer_LDS_2024,
  into `work/Creamer_LDS_2024/`.
- Randi et al. 2023 segmentations (per-animal positions, NeuroPAL confidences): DANDI 001075, into
  `data/randi2023_dandi001075_seg/`.
- CeNGEN thresholded expression (threshold 2, `021821_medium_threshold2.csv`): https://www.cengen.org, into `data/cengen/`.
- Nerve-ring membrane contact ("Adult and L4 nerve ring neighbors.xlsx", the nerve-ring adjacency data of
  Moyle et al. 2021 and Brittin et al. 2021, distributed through WormWiring, https://wormwiring.org), into
  `data/connectome/nerve_ring_neighbors/`. Only the `geometry/` scripts need it.
- Witvliet et al. 2021 developmental connectomes (8 datasets, https://nemanode.org), as one combined edge list
  `data/connectome/witvliet2020/witvliet_combined_edges.csv` with columns `pre, post, type, synapses, source_file` (one row per
  edge per dataset); only `geometry/witvliet_simplicity.py` uses it.
- Atanas et al. 2023 whole-brain recordings (only for the `obs_*` / `forecast*` analyses): WormWideWeb / DANDI 000776,
  extracted to `data/atanas2023_nwb_extract/`.

## References

- Randi, F., Sharma, A. K., Dvali, S., Leifer, A. M. (2023). Neural signal propagation atlas of *Caenorhabditis elegans*. *Nature*.
- Creamer, M. S., Leifer, A. M., Pillow, J. W. (2025). Bridging the gap between the connectome and whole-brain activity in *C. elegans*.
- Taylor, S. R. et al. (2021). Molecular topography of an entire nervous system (CeNGEN). *Cell*.
- Moyle, M. W. et al. (2021). Structural and developmental principles of neuropil assembly in *C. elegans*. *Nature* 591, 99–104.
- Brittin, C. A. et al. (2021). A multi-scale brain map derived from whole-brain volumetric reconstructions. *Nature* 591, 105–110.
- Witvliet, D. et al. (2021). Connectomes across development reveal principles of brain maturation. *Nature* 596, 257–261.
- Atanas, A. A. et al. (2023). Brain-wide representations of behavior spanning multiple timescales and states in *C. elegans*. *Cell*.
