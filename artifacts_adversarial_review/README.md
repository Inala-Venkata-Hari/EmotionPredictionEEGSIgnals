# Adversarial review against the paper

**Verdict: the implementation is not a complete 1:1 reproduction.** The audit found confirmed feature-definition differences, missing experimental stages, a kNN equivalence bug, and incomplete cache/verification safeguards. It did not identify a single demonstrated cause of the full 96% accuracy gap.

Reference: Lara-Arellano et al., [Feature Generation with Genetic Algorithms for Imagined Speech Electroencephalogram Signal Classification](https://doi.org/10.3390/eng6040075), Eng 2025, 6, 75. The workspace PDF matches SHA256 `74cce3e74179ec51a44ee816e8c0786495770404dc90746995a08b4ec52c8a22`.

The findings below distinguish omissions, confirmed departures from the printed paper, assumptions about unpublished author behavior, and bugs. Historical checkpoints and measured metrics were preserved. The two confirmed runtime/provenance bugs were fixed for future runs; no new 100-generation GA was run.

## Findings, in priority order

### 1. High: public-library parity was insufficient evidence of paper fidelity

Location: [author_features.py](../eegis/author_features.py), especially `author_series_features`, lines 130–159; Appendix A.3, A.8–A.9, A.14, A.19–A.21, A.32–A.36.

The latest profile deliberately reproduces available Signalytica behavior, but several functions differ from the printed equations. Calling it an exact implementation of the paper would be incorrect. A real example, No window 0, AF3:

| Quantity | Printed definition / alternative input | Current author-source profile |
|---|---:|---:|
| Coefficient of variation, A3 | std/mean = 0.0040571970 | variance = 290.1976474 |
| Hjorth activity, A14 | mean square = 17,629,859.1601 | centered variance = 290.1976474 |
| Q3, A9 | median of upper half = 4209.2307125 | interpolated percentile = 4208.9742428 |

The original equation-based profile is also incomplete: it chose spectral/recurrence/entropy parameters absent from the paper and supplied conventions for undefined expressions. Neither profile is a verified reference feature matrix from the actual study.

**Status:** confirmed differences from the printed method; actual author execution remains unknown. The source feature bank was preserved. Replacing its formulas silently would invalidate the existing measurements and source-parity evidence. A strict equation profile must be a separate, versioned experiment.

### 2. High: band-dependent functions were wired to already filtered inputs without author assembly code

Location: `author_features.py:153` and `author_features.py:173`; Appendix A.35–A.36.

All 36 functions are evaluated on each supplied band layer. The relative-power function then uses that band's PSD in both numerator and denominator. The amplitude function applies another band-pass filter to the already filtered input before taking RMS.

For the same No/AF3 example:

- Public beta-power function on raw EEG: **0.1816445**; on the beta layer, as wired here: **0.9850140**.
- Public delta-amplitude function on raw EEG, one filter: **4.1308979**; on the supplied delta layer, two filters: **3.5096565**.

The public scalar functions match our calculations. That does not establish that the authors called those functions on these same inputs. Appendix A.36 also prints an envelope/derivative-based mean amplitude, which differs from the RMS implementation. The raw-versus-band denominator in A.35 needs an explicit assembly decision.

**Status:** double filtering and the input-dependent values are confirmed. Whether this differs from the authors' unpublished program is unresolved. This is a material integration assumption, not a recovered implementation step.

### 3. High: the complete two-stage, ten-classifier study was not implemented in the latest runner

Location: `author_reproduction.py`, `fast_fitness` rejects classifiers other than kNN, and `run_ga` saves a single kNN result; Sections 2.8–3, Figures 8 and 10, Table 3.

The older `train_paper_binary --classifiers all` route can search ten classifier factories, but it uses the earlier equation-based feature bank. It is not an all-classifier route through the recovered author-source bank. Only kNN received complete author-source GA searches.

The main runner also refits the selected model on outer training and tests the original outer holdout. It did not subsequently pass each newly evolved mask through the paper's described **20/80 separate validation experiment**. Previous fixed-mask checks used the authors' published mask, not our newly evolved masks.

**Status:** confirmed omission. This review adds the isolated second-stage diagnostic for all three saved masks, with their selected k and the paper's later k=9. It does not implement or execute ten full author-source searches, nor establish a newly independent validation set.

### 4. High for comparability: our GA fitness protocol changes the published experimental design

Location: [train_paper_binary.py](../eegis/train_paper_binary.py), `split_binary`, lines 80–108; Section 2.1.

The paper describes an initial 80/20 split but does not clearly document an extra inner holdout for GA fitness. We introduced one deliberately. In the window experiment, the GA fits on **591** windows and evaluates fitness on **148**, leaving **185** outside the search. In the grouped experiment the corresponding counts are **600/144/180**.

A paper-style initial 80/20 split of all 924 examples would instead give approximately 739 fitting examples and 185 evaluation examples. The unpublished program is needed to establish precisely how that evaluation partition entered fitness.

We also group overlapping windows for the main results; the paper does not document that rule. These controls are useful for evaluating generalization, but they create a different experiment. A separate diagnostic route reproducing the published stage sequence is still needed for a full protocol comparison. Its adaptive fitness must not be presented as independent test performance.

The paper's initial 80/20 and subsequent 20/80 ratios can refer to two stages. They are not, by themselves, a contradiction. It reports k=7 for the GA output and k=9 for the later experiment; 9 is absent from the stated search candidates.

### 5. Medium: Torch kNN was not equivalent to scikit-learn for every legal genome — fixed

Location: [paper_ga.py](../eegis/paper_ga.py), `TorchKNN.neighbors`; [author_reproduction.py](../eegis/author_reproduction.py), `fast_fitness` and `fit_model`.

Torch uses stable training-row ordering for tied distances. Scikit-learn defaults can select a different neighbor subset, particularly when a sparse genome selects one discrete spectral-edge feature and `algorithm='auto'` uses a tree. A single maximum-k neighbor list is not a general replacement for independent library predictions for each k.

On five legal sparse masks using actual training/fitness data, **19 of 35 k-specific prediction sets differed**. For AF3/raw SEF75 at k=37, **47 of 144 predictions differed**. These sparse masks were outside the saved GA runs, whose feature selections contained at least 79 types.

To test relevance to past searches, this audit checked **189 actual historical genomes**, including every genome with at most three electrodes in the two Torch-backed searches, and 64 evenly spaced candidates per run. All **1,323** k-specific prediction sets matched scikit-learn. This is a sample, not an exhaustive audit of every searched genome.

**Fix:** the author-source CPU path now calls actual scikit-learn separately for each k and uses the same reference estimator for final fitting. `--knn-backend legacy-torch` preserves explicit replay of historical Torch-backed searches. The CUDA/Torch path remains available but is not claimed to have exact scikit-learn tie equivalence. New sparse-mask regression checks cover fitness and final inference.

The measured historical results remain measurements of the implementation that ran. This bug has not been shown to explain those results or the paper gap.

### 6. Medium: the feature-cache key omitted transitive dependencies — fixed

Location: `author_reproduction.prepare`; new [feature_provenance.py](../eegis/feature_provenance.py).

The legacy key hashed `author_features.py`, supplied data, and a specification. It did not hash the imported recurrence kernel in `paper_features.py` or include the actual `BANDS` configuration. Source edits in those dependencies could reuse stale features.

A controlled probe changed the delta feature setting from 0.5–4 to 0.5–3.5 Hz while preserving the raw inputs and main feature file. **28 feature values changed while the legacy specification key remained unchanged.** The new dependency fingerprint detects that change.

**Fix:** future cache reuse checks relevant source files and band limits. Unversioned caches are recomputed for training. Existing historical caches stay available for verification. Regression tests cover imported-file changes, band changes, and rejection of an unversioned cache.

The cached feature values checked in this review matched their calculations, so this is a demonstrated invalidation vulnerability, not evidence that the measured runs used stale features.

### 7. Medium: verification wording overstated independence — corrected

The previous fast-metric test compared two Torch-backed prediction paths. The previous verifier's count of approximately 10,000 “candidate fitness scores checked” meant consistency between stored candidate tables and population states; it did not independently rerun scikit-learn for every candidate.

**Fix:** verification receipts now explicitly distinguish stored-score consistency from independent prediction recomputation, identify the selected-fitness prediction backend, and state that all candidate predictions were not independently recomputed. Fresh selected-genome scikit-learn checks from the earlier recheck remain valid. This review adds sampled historical and adversarial sparse-mask comparisons.

### 8. Medium: exact GA parameters and execution settings were assumed

Location: `paper_ga.genetic_search`, lines 174–208; Sections 2.5–2.6.

The paper specifies rank selection, homogeneous crossover and bit-flip mutation, but not the full probability distributions, rates, random seed, invalid-mask handling, or tie policies. We chose linear rank weights, repeated weighted parent sampling, per-bit uniform crossover probability 0.5, mutation probability 1/228, repair of empty masks, seed 42, and deterministic survival ties.

The paper also leaves the standardization axis and exact library versions unspecified. Signalytica's package dependency pins are not proof of the environment used for the paper. CPU versus GPU hardware does not establish the cause of an accuracy difference.

**Status:** explicit reconstruction choices, not recovered hyperparameters. No exact numerical 1:1 claim is justified without additional author evidence.

## Step-by-step coverage

| Paper step | Audit outcome |
|---|---|
| Sí/No only; 444/480 examples | Correct binary counts and original one-second inputs |
| 14 EEG channels, 128 samples, exported CSV index | Index excluded; expected channel names/order validated |
| Five fourth-order Butterworth bands | Supplied bands match per-window filtering on the four audited examples; maximum absolute discrepancy 7.11e-15 |
| Raw + five band layers | All six loaded and aligned by manifest path |
| 34 raw + 5×36 features per electrode | Tensor shape correct; individual feature values and positions checked |
| Printed feature definitions | Several confirmed departures; other numerical settings underspecified |
| Band-power/amplitude input flow | Material raw-versus-band and double-filter assumptions remain |
| 228-bit genome | Published genome matches both extracted PDF copies bit for bit |
| Electrode mask and shared feature mask | Correct; published mask gives 8×103 = 824 flattened inputs |
| Population 100; 100 generations | Completed in historical author-source runs, plus initialization |
| Rank mating, crossover, mutation | Mechanisms present; exact rates/selection distribution unrecovered |
| 100 parents + 100 offspring; top 100 survive | Implemented |
| Accuracy fitness | Implemented; macro-F1 is recorded but not used to break fitness ties |
| Standardize before classifier input | Implemented; population versus within-window axis remains an interpretation |
| Flatten M×N feature map | Correct electrode/feature mapping in sampled tensor checks |
| Ten optimized classifiers | Incomplete in author-source runner |
| Seven candidate k values per genome | Implemented; CPU reference behavior corrected |
| Initial published split/evaluation sequence | Different inner/outer protocol in main experiments |
| Selected outputs → later 20/80 experiment | Previously absent for our masks; isolated diagnostic now executed |
| Accuracy/precision/recall/F1 | Standard sklearn metrics used; paper Eq. 3 prints a different precision denominator; its accuracy equation agrees |
| Independent participant testing | No verified participant map; cannot establish it |

## Feature-definition coverage

This is a comparison to the printed appendix, not certification of the authors' unpublished implementation.

| Equations | Feature | Latest profile versus paper |
|---|---|---|
| A1–A2 | Mean, standard deviation | Matching population-moment definitions |
| A3 | Coefficient of variation | Variance in library; std/mean in paper |
| A4–A7 | Median, mode, maximum, minimum | Definitions match; mode tie choice is unspecified in paper |
| A8–A10 | Quartiles, IQR | Interpolated percentiles versus medians of halves; IQR inherits the choice |
| A11–A12 | Kurtosis, skewness | Matching moment definitions |
| A13 | DFA | Restored AntroPy default; paper does not give full numerical configuration |
| A14 | Hjorth activity | Variance versus printed mean square |
| A15–A16 | Mobility, complexity | Variance-ratio forms implemented; discrete derivative convention explicit |
| A17–A19 | Permutation, approximate entropy | Restored AntroPy calls; embedding/tolerance/normalization choices not fully stated in paper |
| A20–A21 | Spectral entropy | Normalized AntroPy entropy versus printed entropy sum; spectral estimator choices matter |
| A22–A24 | Higuchi dimension | AntroPy defaults; printed length/sign conventions are insufficient to recover exact implementation |
| A25–A26 | Total/centroid PSD | Welch integration for total versus printed sum; centroid form agrees; estimator unspecified |
| A27–A31 | Recurrence features | Source threshold, identity-diagonal and suffix-count conventions retained; paper does not fully specify construction |
| A32–A34 | Spectral-edge frequencies | Discrete Welch cumulative-bin convention; full spectrum/interpolation setup unstated |
| A35–A38 | Hurst exponent | Source rescaled-range implementation; exact scales recovered from source, not specified by paper |
| A39–A41 | SVD entropy | Restored normalized AntroPy call; embedding/normalization configuration not fully stated |
| A42–A43 | Petrosian dimension | AntroPy formula differs from printed equation |
| A44–A45 | Katz dimension | AntroPy formula differs from printed endpoint formula |
| A46–A48 | Relative power | Public function applied to each band input; original-signal denominator interpretation remains unresolved |
| A49–A50 | Band amplitude | RMS after another filter versus printed envelope/derivative-based mean |

Some entropy differences are constant positive rescalings that disappear under per-feature z-scoring. Their presence is not, by itself, an explanation of accuracy. Nonlinear formulas and raw-versus-band data flow require separate controlled experiments.

## Newly executed later-stage diagnostics

All three frozen GA outputs were evaluated on the same declared stratified 20/80 split, seed 42: 184 fit windows and 740 evaluation windows. The plan declared both the selected k and k=9 before these predictions. Each mask keeps its original normalization interpretation.

| Saved mask | k | Correct/740 | Diagnostic accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Grouped, training scaling | 15 | 415 | 56.08% | 0.5433 |
| Grouped, training scaling | 9 | 416 | 56.22% | 0.5543 |
| Grouped, within-window scaling | 7 | 376 | 50.81% | 0.5051 |
| Grouped, within-window scaling | 9 | 361 | 48.78% | 0.4852 |
| Window-search mask, training scaling | 5 | 424 | 57.30% | 0.5715 |
| Window-search mask, training scaling | 9 | 421 | 56.89% | 0.5634 |

These scores are **diagnostic only**. Mask selection can already encode labels in this later partition, and 71 overlap groups cross its fit/evaluation boundary. The exact original author indices are unknown. These are not independent validation scores and no winner was selected. Adding the later evaluation step alone did not recover 96% under these assumptions.

## Verification and practical limits

- 2,568 assembled cached feature values checked on four real windows, three channels, and six layers: zero mismatches, maximum absolute error 7.77e-16. Of these, 1,992 compare to available public functions; 576 compare to reconstructed AntroPy calls and are not an independent author implementation.
- Published 228-bit genome checked against the PDF, not just against its channel/feature counts.
- 189 historical genomes / 1,323 k-specific prediction sets checked: no sampled mismatch.
- Five sparse masks / 35 k-specific prediction sets checked: 19 counterexamples to universal Torch/sklearn equivalence.
- Historical checkpoints, training arrays/scalers, predictions, confusion matrices, and metrics reverified after the code fixes. Stored-score consistency is reported separately from independent prediction checks.
- Main suite: 40 tests, 38 passed, two skipped. Pinned author suite: 11 passed. GPU execution remains untested.

The next complete reproduction needs a separately versioned paper-protocol runner, the full ten-classifier author-source route, and explicit feature/input-flow profiles. Original author feature matrices, complete code and split indices would resolve which profile actually corresponds to 96%. None of the new fixes has been demonstrated to improve full-run accuracy; no new 100-generation result is claimed.

## Replay

```bash
NUMBA_CACHE_DIR=/tmp/eegis-numba-paper OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-paper/bin/python -m eegis.adversarial_review
NUMBA_CACHE_DIR=/tmp/eegis-numba-paper OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-paper/bin/python -m unittest discover -s tests -p test_author_reproduction.py -v
```

The runner requires `pdftotext` and uses frozen cached features without invoking training preparation or replacing old models. `plan.json`, `feature_checks.json`, `neighbor_checks.json`, `cache_check.json`, six later-stage scores/predictions and `verification.json` provide the audit trail.
