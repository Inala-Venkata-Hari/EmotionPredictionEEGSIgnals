# Printed-paper method implementation

The new pipeline implements the printed feature profile and the two-stage, ten-classifier workflow. Earlier experiments are preserved. See [the verified run report](smoke/README.md) for measured results and assumptions. Literal Higuchi and unspecified author settings prevent an unqualified numerical 1:1 claim.

The [full binary kNN study](knn100/README.md) completed population 100 for 100 generations under two predeclared protocols. Independently verified final accuracy is **45.00%** for the exploratory overlap-group holdout (macro-F1 **0.4500**, majority **53.33%**) and **55.95%** for the paper-style diagnostic (macro-F1 **0.5585**). Both selections were frozen before either final evaluation. The other nine classifiers have only the small functional-test results above. The 96% target was not reproduced, and this run did not improve the earlier grouped measurements of 46.67% and 51.67% from the separate author-source profiles.

A separate [one-generation integration check](knn100_check/README.md) tests checkpoint resume, independent verification, and final prediction paths. Its measurements are implementation checks, not full-study results.
