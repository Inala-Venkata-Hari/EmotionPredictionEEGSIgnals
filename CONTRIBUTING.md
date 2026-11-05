# Team setup and collaboration

1. Clone the shared GitHub repository and create a branch for your change.
2. Create a Python virtual environment. Install the direct dependencies from `requirements.txt`, or use `requirements-lock.txt` for the original Python 3.14 CPU environment. PyTorch CPU wheels may require `--extra-index-url https://download.pytorch.org/whl/cpu`.
3. Download [EEGIS version 1](https://data.mendeley.com/datasets/73g4fw884c/1) and extract it in the project root with the directory name shown in the README. Data are CC BY 4.0; retain the source attribution.
4. Run `.venv/bin/python -m unittest discover -s tests -v`.
5. Run `.venv/bin/python -m eegis.train` to reproduce baseline checkpoints. Read the README before running the more expensive follow-up search.

The repository includes source code, measured reports, split manifests, plots, and tests. Raw EEG files, dataset archives, virtual environments, training logs, and binary model checkpoints are excluded. Regenerate the checkpoints locally with the training scripts before running checkpoint verification or inference. The recorded results are real; low accuracy is a finding, not a placeholder.

Keep `artifacts/manifest.csv` fixed when comparing methods. Keep overlap groups together and fit all learned transforms on training data only. Use training-group cross-validation and validation metrics for model selection. Do not choose models, split seeds, or hyperparameters by test performance. The original test set has already been reported; follow-up scores are exploratory. No verified participant identifiers are available, so do not claim subject-independent performance.

Use a branch and pull request for changes. Describe the experiment, selection criterion, split, measured accuracy/macro-F1, and limitations. Commit small metrics and figures, and keep large model files out of Git. Never replace a measured result with an estimate or a target.

To prepare the same shareable bundle locally, run `python scripts/package_share.py`. It writes `share/eegis-team-project.zip` and an upload manifest. The follow-up report directory is included only after its verification completes.

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
