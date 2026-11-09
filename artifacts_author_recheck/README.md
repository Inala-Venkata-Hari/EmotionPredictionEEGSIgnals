# Recheck of the paper's GA and kNN claim

## New implementation after this audit

The historical author-source runner assessed below is preserved. A separate printed-equation implementation is now in [paper_equations.py](../eegis/paper_equations.py), with all ten classifiers and the two-stage procedure in [paper_protocol.py](../eegis/paper_protocol.py). This fixes the audited formula/input-flow and workflow omissions in a new profile. Its [two-generation functional experiment](../artifacts_paper_method/smoke/README.md) completed and was independently verified: later-stage kNN54.05%, SVM54.32%, RF52.30%, MLP57.97%. These are diagnostic scores, not a full100-generation study or independent validation.

The new profile still cannot certify the authors' unpublished numerical execution: exact settings are omitted, and literal Higuchi requires logarithms of nonpositive curve lengths on actual EEG. The runnable alternative and undefined-value policy are explicit. The audit and measured historical results below describe the earlier profile, not scores from the new pipeline.

## Earlier author-source runner verdict

**The earlier author-source runner is not a 1:1 reproduction of the paper.** The PDF methodology and appendix were reread against that code after the adversarial-review fixes. Those fixes corrected CPU kNN handling and feature-cache invalidation; they did not replace that profile's feature definitions or implement the complete published experimental sequence. The separate new implementation is described above.

| Requirement | Current status | Evidence |
|---|---|---|
| 444 Sí + 480 No, one-second windows, 14 channels, 128 Hz | Matches stated inputs | Original binary manifest; no reconstructed 7.875-second inputs used here |
| Raw plus five fourth-order Butterworth bands | Matches stated structure; sampled numerical checks pass | Six supplied layers; audited filtering discrepancies below 7.11e-15 |
| 14×214 features; 228-bit channel/shared-feature genome | Structural match | Published mask matches the PDF bit for bit and selects 824 flattened inputs |
| Population 100; 100 generations; accuracy fitness; seven k values | Matches stated high-level settings | Full historical runs and current GA configuration |
| Printed coefficient of variation, Appendix A.3 | **Does not match** | [author_features.py:143](../eegis/author_features.py#L143) places variance in this position; paper specifies std/mean |
| Printed Hjorth activity, Appendix A.14 | **Does not match** | [author_features.py:145](../eegis/author_features.py#L145) uses variance; paper specifies mean square |
| Printed band amplitude, Appendix A.36 | **Does not match** | [author_features.py:157](../eegis/author_features.py#L157) filters again and returns RMS; paper prints a mean instantaneous-amplitude expression |
| Relative-power input and denominator | **Original implementation not established** | Functions receive already filtered band inputs; the full author assembly program was not recovered |
| Initial search/evaluation split | **Different experiment** | [split_binary](../eegis/train_paper_binary.py#L80) adds an inner fitness partition and an outer holdout; main results also group overlapping windows |
| GA comparison across ten classifiers | **Incomplete in latest runner** | [fast_fitness](../eegis/author_reproduction.py#L34) accepts kNN only; the older ten-classifier route uses another feature profile |
| GA output followed by separate 20/80 experiment | **Diagnostic coverage only** | [stage_two](../eegis/adversarial_review.py#L173) evaluates saved masks; this is not the complete original two-stage study with recovered split indices |
| Exact GA rates, seed, normalization axis, execution versions | **Not verifiable from available evidence** | Declared reconstruction choices; original settings remain unknown |
| CPU sklearn default kNN behavior | Corrected for future runs | Actual sklearn prediction per k; historical Torch runs retained explicitly for replay |

The strict conclusion is **a partial methodological reconstruction with measured results**. Matching public Signalytica functions does not override disagreements with the printed equations, and passing tests does not prove paper parity. The historical metrics predate the latest backend fixes; no new full GA result after those fixes is claimed.

This recheck verified that the six implementation/test files recorded in the [adversarial validation receipt](../artifacts_adversarial_review/validation_summary.json) are unchanged since their passing checks. Their implementation status can therefore be assessed without repeating training. The complete evidence and remaining steps are in the [adversarial review](../artifacts_adversarial_review/README.md).

## Earlier claim and experiment audit

The claim is correctly attributed: Lara-Arellano et al., [Eng 2025, 6, 75](https://doi.org/10.3390/eng6040075), report **96% binary Sí/No accuracy for GA + kNN**. This audit does not establish why their number differs from ours. Our project remains a partially verified reconstruction, not an exact reproduction of the complete authors' experiment.

The workspace PDF SHA256 is `74cce3e74179ec51a44ee816e8c0786495770404dc90746995a08b4ec52c8a22`, matching the previously recorded paper.

## What the paper actually reports

- Section 2.1, page 5: 444 Sí and 480 No one-second windows, 14 channels, 128 Hz, 48-sample overlap, and an initial 80/20 train/test split.
- Sections 2.5–2.8, pages 8–10: 228-bit electrode/feature genomes, population 100, rank selection, homogeneous crossover, bit-flip mutation, accuracy fitness, and evaluation with ten classifiers. Figure 2 specifies 100 generations.
- Section 2.8, page 10: all classifiers use defaults except kNN, which searches `{5,7,13,15,17,21,37}` for every candidate in every generation.
- Section 3, page 11: Figure 8 shows GA convergence above 96% after approximately 20 generations; the published genome selects eight electrodes and 103 feature positions, with k=7 reported as the GA output.
- Section 3, page 11: a **separate validation experiment** uses the selected features, a 20/80 split, and k=9.
- Table 3, page 13: kNN accuracy is 0.96; Figure 10 presents normalized confusion matrices. This is a reported final result as well as a convergence claim, so it must not simply be dismissed as a fitness score.

Correction to earlier project wording: **80/20 followed by 20/80 can describe two stages. The ratios alone are not a contradiction.** The unexplained change from k=7 to k=9 remains relevant, particularly because 9 is absent from the stated candidate set. Exact split indices and the relationship between search data and later validation data are unavailable.

## Confirmed implementation differences

| Item | Paper | Executed implementation |
|---|---|---|
| Full classifier comparison | GA results for ten classifiers | Full binary searches executed for kNN only; earlier nine-class baselines are a different task |
| Search evaluation | Describes an 80/20 split; does not document a separate inner fitness partition | Explicit inner fit/fitness split plus outer test; grouped sizes are 600/144/180 and window sizes 591/148/185 |
| Overlapping windows | Acknowledges overlap; does not specify group-preserving split rules | Main results keep all windows of each overlap component together; window splits are diagnostic only |
| Later validation | Selected features, separate 20/80 experiment, k=9 | Full GA runs refit on outer training and evaluate the original outer test; fixed published-mask diagnostics also check 20/80 and both k values |
| Feature implementation | Lists equations and a library spelled “signalityca” | Initial equation-based bank differed; latest bank ports public Signalytica 0.1.20 behavior, restoring eight commented AntroPy calls |
| Normalization | Column standardization before flattening; axis unspecified | Training-population scaling and within-window electrode scaling tested; global scaling separately marked diagnostic |
| Randomness and GA rates | Seed and probabilities unspecified | Seed 42, uniform bit mask p=0.5, mutation probability 1/228, rank-weighted mating, top-100 survival |
| Software | Names packages, without execution-version pins | Explicit pinned reconstruction environment; package-declared NumPy/SciPy pins are not proof of the original study's execution versions |

Running the other nine classifiers would complete a missing comparison. It would not change the existing kNN search, which already searches all seven published candidate k values for every genome.

The initial 50-generation run was a smoke test. The later author-source runs use population 100 and 100 generations. Too few generations is therefore not a sufficient explanation of the current gap.

## Existing results independently checked again

| Run | Selected-genome fitness | Outer test | Correct/total | Selected k |
|---|---:|---:|---:|---:|
| Grouped, training-population scaling | 82.64% | 46.67% | 84/180 | 15 |
| Grouped, within-window scaling | 82.64% | 51.67% | 93/180 | 7 |
| Window split, training-population scaling | 93.24% | 61.62% | 114/185 | 5 |

For each selected genome, all seven fitness scores were recomputed with independent scikit-learn kNN calls: **21/21 matched**. Final test predictions were recomputed using scikit-learn for all three runs and matched the saved predictions. Accuracy and macro-F1 were recomputed. The eight author-reproduction unit tests also passed, including public-library numerical parity and the tied-neighbor regression.

The 93.24% is repeatedly optimized inner fitness, not an independent test score. The fitness estimator and final refitted estimator also have different training sets. The fitness/test gap is consistent with adaptive selection overfitting and/or differences between recordings, but these causes were not isolated. Neither proves what caused the authors' 96% result.

The window diagnostic shares training EEG samples with 165 of 185 test windows. Grouped results have no exact cross-partition EEG sample overlap, but participant independence is not established. Grouped majority accuracy is 53.33%, above both kNN grouped test accuracies. Previously reported test sets have been reused, so follow-up results are exploratory.

## New check: published mask with within-window scaling

This combination had not been tested previously. Six cases were declared before their new predictions: the fixed published mask, both k=7 and k=9, and all three existing split cases. No GA, seed search, model selection, or changes to old checkpoints were performed.

| Split | k | Fit/test windows | Accuracy | Correct/total | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Grouped | 7 | 744/180 | 46.67% | 84/180 | 0.4661 |
| Grouped | 9 | 744/180 | 45.00% | 81/180 | 0.4492 |
| Window 80/20 | 7 | 739/185 | 60.00% | 111/185 | 0.5986 |
| Window 80/20 | 9 | 739/185 | 59.46% | 110/185 | 0.5929 |
| Window 20/80 | 7 | 184/740 | 53.92% | 399/740 | 0.5391 |
| Window 20/80 | 9 | 184/740 | 55.27% | 409/740 | 0.5527 |

These are all diagnostics: the published mask was selected on this public dataset; window splits additionally share EEG samples. Under training-population scaling, the previous published-mask 20/80 scores were 53.51% at k=7 and 54.32% at k=9. Neither scaling interpretation reproduces 96% on our declared 20/80 split.

Failure with the fixed author mask means failure to evolve the same genome is not, by itself, an adequate explanation under these tested feature/scaling/split choices. It does not establish parity with the authors' unpublished feature assembly or original split.

## What remains unresolved

The raw binary counts, one-second input format, chromosome dimensions, population size, generation count, and k-search set match the stated method. The complete authors' feature-assembly/training code, exact search/validation indices, normalization axis, random state, and GA probabilities are still missing. Public scalar-function parity is evidence for those functions only. It cannot certify an unpublished end-to-end pipeline.

No current evidence assigns the entire accuracy gap to one cause. In particular, “we prevented overlap” is insufficient: our overlapping-window diagnostics also fail to reach 96%. The appropriate next reproduction evidence is the authors' complete program, exact splits, and ideally their saved feature matrix for numerical comparison.

Reproduce this audit from the project root:

```bash
NUMBA_CACHE_DIR=/tmp/eegis-numba-paper OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-paper/bin/python -m eegis.audit_paper_claim
```

The plan, all six new scores and predictions, independent existing-run checks, and verification receipt are saved alongside this report. Earlier measurements and checkpoints remain unchanged.
