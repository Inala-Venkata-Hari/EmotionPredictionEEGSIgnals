# One-generation integration check

This implementation check uses population 4 and one GA generation, plus initialization. It is not the 100-generation experiment.

| Protocol | Measured final accuracy | Macro-F1 | Majority accuracy |
|---|---:|---:|---:|
| Paper diagnostic | 54.05% | 0.5393 | 51.89% |
| Grouped exploratory holdout | 43.89% | 0.4389 | 53.33% |

The independent verifier checked source/data hashes, four rebuilt raw examples, both frozen selections, population/checkpoint consistency, all seven k values for the selected genomes, sampled historical candidates, final refits, predictions, metrics and exact-overlap audits. See verification.json. This is not an exhaustive independent rerun of every candidate prediction.

Neither result establishes subject independence. The paper-style score has overlap and selection-label reuse; the grouped holdout was reported previously. These settings were a functional test, not a hyperparameter search or final-score-based tuning exercise.
