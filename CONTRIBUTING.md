# Team setup and collaboration

1. Clone the shared GitHub repository and create a branch for your change.
2. Create a Python virtual environment. Install the direct dependencies from `requirements.txt`, or use `requirements-lock.txt` for the original Python 3.14 CPU environment. PyTorch CPU wheels may require `--extra-index-url https://download.pytorch.org/whl/cpu`.
3. Download [EEGIS version 1](https://data.mendeley.com/datasets/73g4fw884c/1) and extract it in the project root with the directory name shown in the README. Data are CC BY 4.0; retain the source attribution.
4. Run `.venv/bin/python -m unittest discover -s tests -v`.
5. Run `.venv/bin/python -m eegis.train` to reproduce baseline checkpoints. Read the README before running the more expensive follow-up search.

The repository includes source code, measured reports, split manifests, plots, and tests. Raw EEG files, dataset archives, virtual environments, training logs, and binary model checkpoints are excluded. Regenerate the checkpoints locally with the training scripts before running checkpoint verification or inference. The recorded results are real; low accuracy is a finding, not a placeholder.

Keep `artifacts/manifest.csv` fixed when comparing methods. Keep overlap groups together and fit all learned transforms on training data only. Use training-group cross-validation and validation metrics for model selection. Do not choose models, split seeds, or hyperparameters by test performance. The original test set has already been reported; follow-up scores are exploratory. No verified participant identifiers are available, so do not claim subject-independent performance.

Use a branch and pull request for changes. Describe the experiment, selection criterion, split, measured accuracy/macro-F1, and limitations. Commit small metrics and figures, and keep large model files out of Git. Never replace a measured result with an estimate or a target.

To prepare the same shareable bundle locally, run `python scripts/package_share.py`. It writes `share/eegis-team-project.zip` and an upload manifest. Completed follow-up measurements are included only after verification. An unfinished study can share its immutable plan and an explicitly dated status note; partial histories are excluded.

## Reconstructed-recording research branch

The `experiment/reconstructed-recordings` branch contains continuous-recording reconstruction, recording-level classification, and training-only leakage controls. To inspect it:

```bash
git fetch origin
git switch --track origin/experiment/reconstructed-recordings
```

Read the final README section before comparing scores. Recording-level decisions use 7.875 seconds of EEG. Random-window control scores are deliberately invalid diagnostic results; never present them as speech-classification test accuracy. The controls also test arbitrary labels assigned to complete recording groups. Their predictions and exact overlap counts are audited separately.

Reproduce the branch experiments with `eegis.epochs`, `eegis.verify_epochs`, `eegis.diagnose_leakage`, `eegis.verify_diagnostics`, and `eegis.report_epochs`. The original split and benchmark files remain unchanged. New small reports are in `artifacts_epochs/` and `artifacts_diagnostics/`; checkpoints are regenerated locally.

The same research branch also includes a further one-second comparison of supervised filter-bank CSP, Haar-wavelet features, and a compact power CNN. Run `eegis.train_alternatives`, `eegis.verify_alternatives`, `eegis.compare_alternatives`, then `eegis.report_alternatives` to reproduce its frozen selection and measured reports. This search uses the original training-only grouped CV folds and keeps all nine classes. The CNN trains for a prespecified 50 epochs in every fold; no checkpoint is chosen using holdout scores. See `artifacts_alternatives/` for the plan, CV membership, predictions, and audits. These results remain exploratory because the test set was previously reported.

The grouped ensemble experiment combines the four learned one-second methods through hard voting, group-calibrated soft voting, and logistic stacking. Run `eegis.ensemble`, `eegis.verify_ensemble`, then `eegis.report_ensemble` after regenerating the alternative-method base checkpoints. Three outer folds and two inner folds keep outer held-out groups out of meta-feature generation. Base hyperparameters are frozen from the earlier training search, so these CV scores remain selection estimates. Only the final frozen bundle is evaluated on the original validation/test split. The bundle and derived OOF score matrix stay local; public reports and fit-group audits are in `artifacts_ensembles/`.

The binary paper-method experiment is separate from the nine-class benchmark. Run `eegis.train_paper_binary --generations 50 --population 100 --device auto`, then `eegis.verify_paper_binary` and `eegis.report_paper_binary`. It uses No (6) and Sí (8), the original one-second raw windows, 214 features per channel, and a 228-bit channel/feature mask. The default smoke run searches kNN; `--classifiers all` runs ten separate searches and takes substantially longer. These are genetic generations, not neural-network epochs.

Read `artifacts_paper_binary/feature_spec.json` and `plan.json` before describing the method as a reproduction. The paper leaves parameters undefined. Its initial 80/20 and later 20/80 splits can describe separate stages; k=7 during GA selection versus k=9 in later validation remains unexplained. The grouped result prevents exact overlap leakage; the nested `paper_window_diagnostic/` directory intentionally measures an overlapping-window split and must never be reported as valid decoder accuracy. The old nine-class benchmark remains intact. The checkpoint returns original class IDs 6 and 8.

The CUDA-capable kNN uses the existing PyTorch dependency for distance calculations and neighbor sorting. SciPy feature extraction, standardization, voting, and other scikit-learn models remain on CPU. `--device cuda` requires a working NVIDIA driver and a CUDA-enabled PyTorch build; it fails clearly when unavailable. The pinned `requirements-lock.txt` describes the measured CPU environment, not a CUDA installation. The current host could only validate CPU execution. The test suite conditionally runs a real CPU/CUDA prediction comparison when CUDA is available.

The further author-source reproduction uses `.venv-paper` with Python 3.10 and `requirements-author.txt`. Signalytica 0.1.20 pins NumPy 1.21.6 and SciPy 1.7.3, so do not install it into the main Python 3.14 environment. The other paper-era ML versions are explicit compatible choices, not recovered original pins. Follow `artifacts_author_reproduction/README.md` to run the two 100-generation searches, the predeclared within-example normalization interpretation, eight fixed published-mask checks, independent verification, and reporting. Run `test_author_reproduction.py` in that isolated environment to exercise direct parity with the author's library. The modern suite skips only that isolated-library check and unavailable CUDA hardware.

The within-example normalization run uses actual sklearn scaling and independent kNN predictions for every k. Two-electrode masks create many equal distances; slicing a common neighbor list can disagree with sklearn defaults. A rejected incomplete run exposed this difference and was archived locally. The corrected run repeats the full search and has a regression test for these ties. Keep test scores out of normalization, seed, feature-mask and k selection. The recommendation compares the two grouped runs by inner fitness only; window-split and externally selected published-mask scores are diagnostics, never independent performance claims. No author or teammate messages are sent.

The newer printed-equation workflow is documented in `artifacts_paper_method/`. Its ten-classifier functional run and two full kNN searches have independent verification receipts. Follow `artifacts_paper_method/knn100/README.md` for fresh-clone feature preparation, training into a new output directory, verification, plots, and raw-CSV prediction. A clean-copy replay rebuilt the complete feature array exactly and reproduced the recorded integration check. Both full kNN searches used population 100 and 100 generations; the other classifiers have only functional-test results. Read the diagnostic/exploratory limitations before citing any accuracy.

The additional accuracy research is summarized in `artifacts_accuracy_refit/README.md`. Six batches use five outer and three inner folds inside the original training partition, with original validation/test excluded from tuning. A separately frozen final refit adds validation groups for fitting and scores the previously reported test once. Preserve the original training-only models and all negative results. The prototype target was not reached; do not describe these checkpoints as useful decoders.

MiniRocket helpers retain their BSD license, while LaBraM architecture/channel code retains the author's MIT license. No new packages are needed for these adapters. To reproduce pretrained/partial-fine-tuning studies, first download the official pinned checkpoint with `python scripts/download_labram.py`; it remains excluded from Git. All candidate CV rows and CNN histories are shared in consolidated unit-root CSVs; per-family resume caches and binary models stay local.
