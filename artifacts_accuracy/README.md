# Binary accuracy research: search in progress

This follow-up tests No (6) versus Sí (8) using 19 model families and 86 hyperparameter configurations. The initial prototype target is 70% balanced accuracy; no passable result is claimed yet.

Five outer group folds measure the complete model-selection procedure. Each outer fitting partition uses three inner group folds to select features, preprocessing and model parameters. Search uses only the original 47 training overlap components (564 one-second windows). Original validation/test partitions are excluded from search, tuning and epoch selection. All EEGIS examples appeared in earlier analyses, so these results will remain exploratory even with correct nested grouping. No verified participant IDs are available.

Two decision durations are evaluated separately: one second and a reconstructed 1,008-sample component (7.875 seconds). Each original component stays together in every fold. Longer-context accuracy must not be presented as one-second accuracy.

Representations include spectral shape, common-average referencing, regularized covariance/tangent features, supervised filter-bank CSP, and fixed random multivariate temporal kernels with PPV/max pooling. Classifiers include regularized linear models, SVM, shrinkage LDA, Extra Trees, gradient boosting and majority controls. Imputation is unnecessary for these finite feature banks. Learned spatial filters, feature scaling and classifiers fit only the current training fold.

The temporal kernels are an independent implementation inspired by [ROCKET](https://arxiv.org/abs/1910.13051), not a claim of MiniROCKET reference-code parity. Weights and bias thresholds use a declared seed and input shape, without fitting EEG statistics or labels. Other feature-bank calculations are independent for each example.

Run `.venv/bin/python -m eegis.train_accuracy`, then `.venv/bin/python -m eegis.verify_accuracy`. Use four numerical/Numba threads and the modern requirements-lock.txt environment. Source/data/runtime/configuration hashes guard resumes. Feature caches and models remain local; final measurements will be shared after verification. All earlier experiment results are preserved.
