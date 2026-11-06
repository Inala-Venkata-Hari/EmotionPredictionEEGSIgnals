"""Append measured nested-ensemble results to README without replacing prior runs."""
import json
from pathlib import Path
import pandas as pd


def main():
    out = Path('artifacts_ensembles')
    r = json.loads((out/'results.json').read_text())
    verified = json.loads((out/'verification.json').read_text())
    assert verified['outer_groups_excluded_from_inner_meta_features']
    winner = r['best_model']; best = r['models'][winner]
    learned = ['HardVote', 'CalibratedSoftVote', 'Stacking']
    observed = max(learned, key=lambda name: r['models'][name]['test']['accuracy'])
    rows = ['## Group-safe ensemble experiment', '',
        'This experiment combines the four previously trained one-second learners: FBCSP SVM, Haar SVM, Haar shrinkage LDA, and the power CNN. Their previously selected training hyperparameters are frozen. No original validation or test labels enter calibration, stacking, vote tie-breaking, or ensemble selection. The original nine-class split remains unchanged.', '',
        'Three ensemble methods are compared:', '',
        '- **Hard voting:** each base model supplies one class vote. Training class priors resolve vote ties; remaining exact ties use fixed class order.',
        '- **Calibrated soft voting:** each member’s score vector is converted to nine-class probabilities by a scaler + multinomial logistic-regression calibrator (C=0.1), fitted on group-excluded predictions. Member probabilities are averaged equally. SVM decision scores are used without enabling built-in probability fitting. The learned probability mapping does not by itself guarantee well-calibrated confidence.',
        '- **Stacking:** a scaler + logistic-regression meta-classifier (C=0.1) learns from the concatenated 36 base scores, using group-excluded predictions. It never receives the base learners’ in-sample predictions.', '',
        'A nested training-group procedure prevents a second layer of leakage: three outer folds assess each ensemble, and two inner folds produce meta-training predictions entirely within each outer training partition. The outer held-out groups are excluded from every inner fit and prediction set. A separate full outer-training base fit predicts the outer held-out fold. There are 36 base-model fits in total, including nine fixed 50-epoch CNN fits. No epoch or ensemble weight is tuned on the holdout.', '',
        'The three outer base predictions also form one out-of-group score row for every original training window. After ensemble selection by pooled outer-fold accuracy (macro-F1 tie-break), those rows fit the final calibrators and stacker. The already fitted full-training base checkpoints are reused for validation/test inference and checked by SHA-256. This nested procedure guards meta-feature fitting, but the base hyperparameters were previously selected on this training partition, so CV remains a model-selection estimate. The original test set was already reported; these are exploratory follow-up results.', '',
        '| Method | Nested training CV accuracy | Validation accuracy | Test accuracy | Test macro-F1 | Test classes predicted |',
        '|---|---:|---:|---:|---:|---:|']
    for name, d in r['models'].items():
        rows.append(f"| {name} | {d['training_cv']['accuracy']:.2%} | {d['validation']['accuracy']:.2%} | {d['test']['accuracy']:.2%} | {d['test']['macro_f1']:.4f} | {d['predicted_classes']} |")
    rows += ['',
        f"The frozen CV-selected method is **{winner}**, with {best['test']['accuracy']:.2%} test accuracy and {best['test']['macro_f1']:.4f} macro-F1. The highest observed ensemble test accuracy is {observed} ({r['models'][observed]['test']['accuracy']:.2%}); observing the test ranking does not change the selected method. The previous best learned one-second result was Haar SVM at 11.74%, while the training-majority reference scores 12.12%. A method predicting only one class is a reference-level result rather than useful word recognition.", '',
        'Member disagreement, joint errors, and error correlations are measured from the group-excluded base predictions in `diversity.csv`. These are descriptive measures on correlated EEG windows, not independent significance tests. Diversity alone does not establish that members carry useful word information. This experiment does not establish subject-independent performance.', '',
        f"Runtime: {r['elapsed_seconds']:.1f} seconds on CPU. All 19 automated tests pass. `fit_audit.json` records the fit, prediction, and forbidden groups for every base fit. Independent verification checks all 36 fits, inner/outer role assignments, reconstructed meta-models from OOF scores, source-checkpoint hashes, every held-out prediction/metric, confusion counts, selection, and one-window inference.", '',
        '![Nested ensemble comparison](artifacts_ensembles/comparison.png)', '',
        'Reproduce after generating the base checkpoints with `eegis.train_alternatives`:', '', '```bash',
        'MPLCONFIGDIR=/tmp/eegis-matplotlib .venv/bin/python -u -m eegis.ensemble > ensemble.log 2>&1',
        'MPLCONFIGDIR=/tmp/eegis-matplotlib .venv/bin/python -m eegis.verify_ensemble',
        '.venv/bin/python -m eegis.report_ensemble',
        '.venv/bin/python -m unittest discover -s tests -v',
        '.venv/bin/python -m eegis.predict --artifacts artifacts_ensembles "EEGIS - Electroencephalogram Imagined Speech dataset/raw/class_0/0.csv"',
        '```', '',
        'The selected checkpoint (`artifacts_ensembles/best_model.joblib`) bundles full base learners, calibrators, stacker, prior, and frozen method. Its `predict_all` method exposes all four comparisons. Binary checkpoints and the derived OOF score matrix are kept local; regenerate training outputs before running verification after cloning. Small reports, group assignments, OOF labels/predictions, confusion matrices, and plots are shareable. Work stays on `experiment/reconstructed-recordings`; main is unchanged.']
    if winner == 'Majority':
        rows += ['', 'None of the three learned ensembles beats the majority reference in nested training CV. No successful accuracy improvement is claimed.']
    p = Path('README.md')
    text = p.read_text().split('\n## Group-safe ensemble experiment')[0]
    p.write_text(text+'\n\n'+'\n'.join(rows)+'\n')
    print('README updated from verified ensemble measurements')


if __name__ == '__main__':
    main()
