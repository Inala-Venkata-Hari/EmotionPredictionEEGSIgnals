# Printed-paper method implementation

The new pipeline implements the printed feature profile and the two-stage, ten-classifier workflow. Earlier experiments are preserved. See [the verified run report](smoke/README.md) for measured results and assumptions. Literal Higuchi and unspecified author settings prevent an unqualified numerical 1:1 claim.

The [full binary kNN study](knn100/README.md) runs population 100 for 100 generations under two predeclared protocols: the paper-style diagnostic and an exploratory overlap-group holdout. It is still in progress. Both selections must be frozen before either final evaluation. The other nine classifiers have only the small functional-test results above.

A separate [one-generation integration check](knn100_check/README.md) tests checkpoint resume, independent verification, and final prediction paths. Its measurements are implementation checks, not full-study results.
