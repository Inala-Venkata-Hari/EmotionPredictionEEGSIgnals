# Binary kNN study: training in progress

Status snapshot: 2026-11-09T07:11:41.033882+00:00. This note is a publication snapshot, not live training telemetry. See publication_status.json for completed generation counts.

Two searches use population 100 and 100 genetic generations, plus initialization. Both use the fixed printed-equation feature bank for 480 No and 444 Sí windows. Both masks must be frozen before either final evaluation. **Final accuracy and macro-F1 are not yet available.** Adaptive GA fitness must not be reported as final test accuracy.

The paper-style protocol searches with 739 fitting and 185 fitness windows, then fits the frozen mask on 184 windows and evaluates 740 with k=9. EEG overlap and selection-label reuse make that score diagnostic.

The exploratory grouped protocol excludes the original 180-window holdout from the whole GA. It uses 600 fitting and 144 fitness windows internally, then fits 744 windows before the final evaluation. Entire exact-overlap components stay together. This holdout was reported previously, and participant identifiers are unavailable; the result will not establish subject-independent accuracy.

The grouped checkpoint is predeclared as the primary saved model. Final scores do not select the protocol, seed, normalization, k or feature mask. See study_plan.json for frozen source/data hashes and split indices.

The median-handling optimization retains the original batched sklearn transform bit for bit. Reference sklearn predicts independently for each of the seven candidate k values. Optimization parity and sampled live checkpoint audits are recorded separately; they do not replace final verification.

Use the pinned Python 3.10 environment from requirements-author-lock.txt. Regenerate the verified feature bank first using the smoke command in ../smoke/README.md, with --out artifacts_paper_method/smoke, then run:

```bash
NUMBA_CACHE_DIR=/tmp/eegis-numba-paper MPLCONFIGDIR=/tmp/eegis-matplotlib-paper OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-paper/bin/python -u -m eegis.train_knn100
NUMBA_CACHE_DIR=/tmp/eegis-numba-paper OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-paper/bin/python -m eegis.verify_knn100
.venv-paper/bin/python -m eegis.report_knn100
```

Training resumes a local checkpoint only when code, data and configuration hashes match. Binary caches and search checkpoints are not included in GitHub. The other nine classifiers have only the earlier small functional-test results. Exact unpublished author settings remain unavailable; no unqualified numerical 1:1 claim is made.
