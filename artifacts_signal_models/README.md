# Phase connectivity and neural accuracy research: in progress

This second predeclared batch tests phase-locking value, weighted phase-lag index, imaginary coherence, amplitude-envelope correlation, delayed cross-channel correlations, and higher-frequency spectra. A 0.5–60 Hz branch applies a 50 Hz notch; spectral summaries omit 49–51 Hz. It also tests a small EEG power CNN for 50 fixed epochs per fit, with training-only circular shifts and noise augmentation.

The study uses the same five outer/three inner grouped folds restricted to the original 47 training components as the first accuracy study. Feature selection is fitted inside each training fold. Original validation/test partitions remain excluded from search. The batch follows the first train-only comparison and remains exploratory; no independent confirmation or subject-independent claim is possible from this release alone.

One-second and 7.875-second decisions are reported separately. No scores are invented or selected by changing split seeds. The CNN uses CUDA automatically when available; the present environment has CPU-only PyTorch and no GPU execution is claimed. Checkpoints return original labels 6 (No) and 8 (Sí).

Run `.venv/bin/python -m eegis.train_signal_models`, then `.venv/bin/python -m eegis.verify_accuracy --artifacts artifacts_signal_models`. The frozen plan records every candidate and the numerical environment. Raw data, caches, binary checkpoints and logs stay local. Final reports will be shared after verification.
