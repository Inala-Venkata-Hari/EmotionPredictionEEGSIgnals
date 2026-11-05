"""Append measured recording reconstruction and leakage-control results to README."""
import json
from pathlib import Path
import pandas as pd


def main():
    out = Path('artifacts_epochs')
    r = json.loads((out/'results.json').read_text())
    audit = json.loads((out/'reconstruction_audit.json').read_text())
    verification = json.loads((out/'verification.json').read_text())
    diagnostic = json.loads(Path('artifacts_diagnostics/results.json').read_text())
    diagnostic_verification = json.loads(Path('artifacts_diagnostics/verification.json').read_text())
    assert verification['metrics_recomputed'] and diagnostic_verification['metrics_and_summary_recomputed']
    winner = r['best_model']; test = r['scores']['test']
    outcome = ('None of the learned candidates beats the majority baseline in training CV. '
               'The selected constant predictor is saved for reproducibility; this is not an accuracy improvement or a usable imagined-speech decoder.'
               if winner == 'Majority' else
               'This recording-level candidate was chosen by training CV; its score does not replace or directly improve a one-second result.')
    fits = sum(len(pd.read_csv(p))*3 for p in out.glob('*_cv.csv'))
    rows = [
        '## Reconstructed-recording experiment and leakage diagnosis', '',
        'Work branch: `experiment/reconstructed-recordings`. The original one-second experiments and their results are preserved. This experiment reconstructs each overlap component into a continuous 1,008-sample recording (7.875 seconds). It differs from the earlier context experiment, which averaged predictions from separately processed windows.', '',
        f"All {audit['recordings']} components were reconstructed using exact 48-sample prefix/suffix matches, without filenames or labels determining window order. Every original window is reproduced exactly at its recovered offset. Duplicate, missing, mixed, ambiguous, or cyclic chains are rejected. The audit found {audit['cross_split_shared_samples']} identical multichannel samples crossing partitions. The same 205/66/66 training/validation/test components are retained. Participant identities are still unavailable.", '',
        'Continuous processing may reduce the relative influence of one-second filtering boundaries and allows a 256-sample Welch segment (0.5 Hz bins). Spectral features comprise five log absolute and five relative bandpowers per channel, log variance, Hjorth mobility/complexity, and spectral entropy: 196 features. Band energy sums use half-open frequency intervals, avoiding shared band-edge bins. Signals are detrended and filtered per recording. Common-average referencing is tested both enabled and disabled.', '',
        'Spatial features use six regularized channel covariance matrices (broadband plus the five EEG bands), followed by training-fitted covariance whitening, matrix logarithms, and scaling: 630 features. The models are shrinkage LDA, PCA + RBF SVM, logistic regression, and spatial shrinkage LDA. All fitted steps stay inside the cross-validation pipelines.', '',
        f'The search used {fits} model fits in three-fold cross-validation on the 205 training recordings, plus the majority baseline. Family and hyperparameter selection use training CV accuracy, with macro-F1 breaking family ties. Validation and test scores do not choose the winner. Only the frozen winner and the training-majority baseline are evaluated on the held-out partitions. The test split was previously reported, so this remains exploratory.', '',
        '| Candidate | Training CV accuracy | Training CV macro-F1 |',
        '|---|---:|---:|']
    for name, item in r['training_candidates'].items():
        rows.append(f"| {name} | {item['training_cv_accuracy']:.2%} | {item['training_cv_macro_f1']:.4f} |")
    rows += ['',
        f"**Selected: {winner}.** Validation accuracy is {r['scores']['validation']['selected']['accuracy']:.2%}; test accuracy is {test['selected']['accuracy']:.2%}, macro-F1 {test['selected']['macro_f1']:.4f}, across {r['split_sizes']['test']} recordings. The training-majority test baseline is {test['majority']['accuracy']:.2%}. {outcome}", '',
        f"The recording experiment took {r['elapsed_seconds']:.1f} seconds on CPU. `artifacts_epochs/reconstruction_audit.json` records the source-manifest checksum and exact reconstruction checks. `verification.json` confirms replayed reconstruction, unchanged partitions, reloaded predictions, recomputed metrics and confusion counts, training-only selection, and inference from reversed input-window order. The saved model is `artifacts_epochs/best_model.joblib`; its input contract is in `best_model.json`.", '',
        '![Training-only recording CV](artifacts_epochs/training_cv.png)', '',
        '### Training-only leakage controls', '',
        'These deliberately include an invalid random-window protocol to diagnose inflated scores. They are not held-out test performance and must not be used as claimed system accuracy. The original validation and test partitions are excluded completely. A fixed 300-tree Random Forest uses the original 238 features. Seeds 42, 43, and 44 produce three 80/20 splits inside the original training partition; the group-safe and random-window controls each train on 1,968 windows and evaluate on 492.', '',
        'For the shuffled control, labels are permuted between the 205 recording groups and remain consistent across every window of a group. This removes the original word-to-recording label assignment while retaining opportunities to recognize a recording. Both protocols are rerun on those arbitrary group labels.', '',
        '| Protocol | Target labels | Diagnostic accuracy, mean ± SD | Diagnostic macro-F1 |',
        '|---|---|---:|---:|']
    for item in diagnostic['summary']:
        rows.append(f"| {item['protocol']} | {item['target']} | {item['accuracy_mean']:.2%} ± {item['accuracy_std']:.2%} | {item['macro_f1_mean']:.4f} |")
    rows += ['',
        'The invalid random-window scores remain elevated after group labels are shuffled. This is evidence that shared samples and recording-specific structure can inflate accuracy without useful word classification. The valid group controls have zero shared multichannel samples; the invalid controls expose most diagnostic holdout windows to training samples. The standard deviations describe three repeated splits of the same training data, not confidence intervals. These controls do not prove that the dataset contains no speech information.', '',
        '![Training-only leakage controls](artifacts_diagnostics/leakage_control.png)', '',
        'Every diagnostic prediction is saved in `artifacts_diagnostics/diagnostic_predictions.csv`. The independent verification script regenerates the internal split indices, checks group-label permutations, recomputes exact sample-overlap counts, and recomputes the metrics and summary from those predictions. No control model is promoted or saved as a speech classifier.', '',
        'Reproduce this branch work:', '', '```bash',
        'MPLCONFIGDIR=/tmp/eegis-matplotlib .venv/bin/python -u -m eegis.epochs > epochs.log 2>&1',
        'MPLCONFIGDIR=/tmp/eegis-matplotlib .venv/bin/python -m eegis.verify_epochs',
        'MPLCONFIGDIR=/tmp/eegis-matplotlib .venv/bin/python -u -m eegis.diagnose_leakage > leakage_diagnostic.log 2>&1',
        '.venv/bin/python -m eegis.verify_diagnostics',
        '.venv/bin/python -m eegis.report_epochs',
        '.venv/bin/python -m unittest discover -s tests -v',
        '```', '',
        'Single-recording inference is available as `.venv/bin/python -m eegis.epochs --predict <12 CSV paths>` and rejects invalid chains. All 11 automated tests pass. Source, small CSV/JSON reports, figures, and audit files are shareable; raw signals, virtual environments, and binary checkpoints remain excluded from Git.', '',
        'The next useful step is to obtain verified participant/session/trial identifiers and establish an untouched evaluation set before making stronger generalization claims. Further tuning against the already reported holdout risks choosing noise. The existing nine-class task includes rest; reducing the class set would be a different experiment and must be labeled separately.', '',
        'Related primary research: the dataset authors’ [Feature Generation with Genetic Algorithms for Imagined Speech Electroencephalogram Signal Classification](https://doi.org/10.3390/eng6040075) discusses recognizing Sí and No and describes full-vocabulary EEGIS as future work. Compare published results only after matching the label set, channels, feature-selection procedure, and split protocol. This nine-class experiment is not a reproduction of that binary study.']
    path = Path('README.md')
    existing = path.read_text().split('\n## Reconstructed-recording experiment and leakage diagnosis')[0]
    path.write_text(existing+'\n\n'+'\n'.join(rows)+'\n')
    print('README updated from verified measured reports')


if __name__ == '__main__':
    main()
