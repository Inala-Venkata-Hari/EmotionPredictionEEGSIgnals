# EEGIS imagined-speech EEG classification

This project trains an RBF SVM, Random Forest, and compact EEGNet-style CNN on the locally supplied EEGIS version 1 archive. Experiments use real CSV signals; results are recorded by the training program.

## Dataset and provenance

Source: [EEGIS version 1, DOI 10.17632/73g4fw884c.1](https://data.mendeley.com/datasets/73g4fw884c/1), Edgar Lara, Andras Takacs, and Juvenal Rodríguez, CC BY 4.0. The source describes ten participants, Emotiv EPOC+, and 128 Hz sampling. The release contains raw signals plus delta, theta, alpha, beta, and gamma variants. Windows contain 128 time samples and 14 channels, with an exported row-index column. The source reports 48-sample overlap after trimming each state recording. Channel names are AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4.

The class mapping is 0 rest, 1 Ayuda, 2 Baño, 3 Dolor, 4 Gracias, 5 Hambre, 6 No, 7 Sed, 8 Sí. The observed raw counts by class 0–8 are 456, 504, 372, 492, 444, 444, 480, 408, and 444. The local archive SHA-256 and size are in `provenance.json`. The inspection report records actual CSV counts, shapes, ranges, and channel names. Every CSV in all six variants is checked for layout, numeric finite values, and index consistency. Only raw CSVs enter training; filtered copies are never treated as additional examples.

## Evaluation and leakage controls

The release provides class folders and numeric filenames, with no verified participant, session, or original trial identifiers. Numeric filenames are **not** interpreted as participant IDs. Consequently, the measured experiment is an overlap-component holdout, **not subject-independent testing**. Participants and sessions may appear in multiple partitions; these scores cannot establish performance on a new person.

For each raw window, the loader hashes its first and last 48 multichannel samples and its full content. Union-find joins matching windows globally, regardless of filename order or class folder. All windows in each connected component stay in one partition. A second audit rejects any identical 14-channel time sample crossing partitions. This prevents leakage of overlapping source samples and exact duplicated windows in the supplied data. It cannot recover unknown session/participant relationships.

Components are shuffled separately within each class with seed 42 and assigned approximately 60% training, 20% validation, 20% test. All classes must occur in every partition. The exact assignments are saved in `artifacts/manifest.csv`. No new overlapping windows are generated. No test examples are used for fitting scalers, hyperparameter search, early stopping, or model selection. Test examples are evaluated after the model selection decision has been frozen. Test scores are window-level; windows within a component remain correlated, so the window count is not an independent-trial count.

If verified participant metadata becomes available, supply a CSV with `path,subject_id`, where paths look like `raw/class_0/0.csv`. It must cover every raw file, and overlapping windows must agree on subject. `--subject-map participants.csv --out artifacts_subjects` switches to participant-group splits, keeps subjects disjoint across all partitions, and checks class coverage. No participant IDs were invented for the present experiment.

## Signal processing and models

Each window is independently demeaned per channel, then filtered with a fourth-order 0.5–40 Hz Butterworth bandpass using forward/backward filtering. Processing a single second limits low-frequency resolution and introduces edge effects; no continuous recording is available for improved filtering. No ICA or unsupported artifact-removal claim is made.

The baselines receive 238 features: for each channel, five log absolute and five relative Welch bandpowers (0.5–4, 4–8, 8–13, 13–30, 30–40 Hz), log variance, log mean squared first difference, Hjorth mobility and complexity, skewness, excess kurtosis, and peak-to-peak amplitude. Welch uses 128 samples (1 Hz bins); relative powers normalize across the five estimates. Band-edge bins are included in both adjacent bands. SVM features are standardized by a scaler fit on training data only. Random Forest uses unscaled features.

SVM searches C in {0.1, 1, 10, 100} and gamma in {scale, 0.001}. Random Forest searches maximum depth in {unlimited, 15} and minimum leaf size in {1, 3}, with 300 trees. Both use balanced class weights and validation macro-F1 for selection.

The CNN uses a temporal convolution (8 filters, kernel 33), depthwise spatial convolution (depth multiplier 2 across all channels), and a separable temporal convolution (kernel 17). Batch normalization, ELU, average pooling by 4 then 8, and dropout 0.35 precede a nine-class linear layer. This is a compact EEGNet-style implementation, not a claim of exact reference-code reproduction. Channel mean/std are fit on training windows only. Training uses class-weighted cross entropy, Adam (learning rate 0.001, weight decay 0.0001), batch size 64, at most 80 epochs, and early stopping after 15 epochs without validation macro-F1 improvement. The best validation checkpoint is restored. CPU operations are seeded and deterministic.

The overall saved winner is selected by validation macro-F1 across the three models. It is retained as trained on the training partition, allowing direct reproduction of its reported test predictions; there is no subsequent train+validation refit.

## Measured results

Measured on 792 held-out windows (66 complete overlap groups), using seed 42. Training used 2,460 windows (205 groups); validation used 792 windows (66 groups). All 24,264 CSVs were inspected; the six variants each contain 4,044 windows. All 337 raw overlap components contain exactly 12 windows.

| Model | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| SVM | 0.0986 | 10.48% | 0.0939 |
| RandomForest | 0.1308 | 8.08% | 0.0798 |
| EEGNet | 0.0895 | 11.24% | 0.1034 |

The training-majority baseline achieved 12.12% accuracy and 0.0240 macro-F1. A uniform nine-class guess has expected accuracy 11.11%; this reference is analytical, not another measured model.

Random Forest had the highest validation macro-F1 and was saved as `artifacts/best_model.joblib` (300 trees, unlimited depth, minimum leaf size 1). SVM selected C=0.1 and gamma=scale. EEGNet achieved the highest observed test scores, but the test set was not used to change the validation-selected winner; its checkpoint is available at `artifacts/EEGNet.pt`. The mismatch between validation and test rankings illustrates uncertainty in this small grouped holdout.

EEGNet stopped after 29 epochs and restored epoch 14. The full experiment took 146.6 seconds on CPU, including inspection, fitting, and evaluation. These models show no useful classification performance on this holdout. They should not be described as a successful speech decoder. No extra tuning was performed after viewing test results.

![CNN training curves](artifacts/training_curves.png)

![Measured test comparison](artifacts/comparison.png)

The confusion matrices and per-class precision/recall/F1 are saved for every model. Reported macro-F1 is the unweighted average over all nine classes, including rest. No per-window random split or fabricated subject-independent score is reported.

## Reproduce

Run from this workspace with Python 3.14 (the measured environment):

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
MPLCONFIGDIR=/tmp/eegis-matplotlib .venv/bin/python -u -m eegis.train > experiment.log 2>&1
.venv/bin/python -m unittest discover -s tests -v
MPLCONFIGDIR=/tmp/eegis-matplotlib .venv/bin/python -m eegis.verify
```

`requirements-lock.txt` records the exact installed versions, including the CPU PyTorch build. `requirements.txt` lists direct dependencies for platforms that require different wheels. The dataset is expected in `EEGIS - Electroencephalogram Imagined Speech dataset/`. Keep the original directory layout; the dataset is not redistributed as project source.

To classify one correctly ordered raw window with the saved selected model:

```bash
.venv/bin/python -m eegis.predict 'EEGIS - Electroencephalogram Imagined Speech dataset/raw/class_0/0.csv'
```

Inference includes the same preprocessing and feature extraction, or saved CNN normalization. It returns a class ID/name; it does not estimate calibrated confidence.

## Outputs and source

- `eegis/data.py`: inspection, overlap reconstruction, splits, leakage audit, preprocessing, features.
- `eegis/models.py`, `eegis/train.py`: CNN, fitting, model selection, evaluation, plots.
- `eegis/predict.py`: saved-model inference.
- `tests/test_leakage.py`: overlap transitivity, rejection of leaking splits, grouped splitting, and feature/preprocessing invariance, and participant-map consistency.
- `artifacts/inspection.json`, `manifest.csv`, `leakage_audit.json`: dataset and partition audit.
- `artifacts/results.json`, `comparison.csv`: measured metrics, per-class reports, parameters, runtime, software versions.
- `artifacts/baseline_search.csv`, `selection.json`: all baseline candidates and validation selection.
- `artifacts/training_history.csv`, `training_curves.png`: CNN training and validation graphs.
- `artifacts/*_confusion.csv`, `*_confusion.png`, `comparison.png`: raw count matrices and comparison graphs.
- `artifacts/test_predictions.csv`: labels and predictions for every held-out window.
- `artifacts/SVM.joblib`, `RandomForest.joblib`, `EEGNet.pt`: all selected model checkpoints.
- `artifacts/best_model.*`, `best_model.json`: selected checkpoint and its input contract.

Validation: all four automated tests passed. `eegis.verify` reloads each checkpoint, compares every test prediction, recomputes accuracy/macro-F1, checks confusion counts and model selection, and verifies single-window inference. Its outcome is saved in `artifacts/verification.json`.

These are exploratory results from one fixed holdout split. More seeds and verified subject/session identifiers are needed for stronger generalization claims. Nine-class performance includes rest and should not be presented as eight-word-only accuracy.
