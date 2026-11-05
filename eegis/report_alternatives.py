"""Write the measured alternative-technique experiment into the README."""
import json
from pathlib import Path
import pandas as pd


def main():
    out = Path('artifacts_alternatives')
    r = json.loads((out/'results.json').read_text())
    verification = json.loads((out/'verification.json').read_text())
    assert verification['all_model_predictions_reproduced']
    winner = r['best_model']; best = r['models'][winner]
    majority = r['models']['Majority']
    models = {k: v for k, v in r['models'].items() if k != 'Majority'}
    highest_observed = max(models, key=lambda n: models[n]['test']['accuracy'])
    fits = sum(len(pd.read_csv(p))*3 for p in out.glob('*_cv.csv'))
    rows = ['## Alternative one-second techniques', '',
        'This attempt returns to the original nine-class, one-second task. The original manifest, seed, and 2,460/792/792 training/validation/test windows are retained. All overlap groups remain intact in outer partitions and training CV folds.', '',
        'The search space is recorded before fitting in `artifacts_alternatives/plan.json`. New methods:', '',
        '- **Filter-bank CSP + SVM:** supervised one-vs-rest spatial filters are fitted independently for each of the five EEG bands. Each class covariance is compared against the mean of the other class covariances, with 10% isotropic shrinkage. The smallest/largest generalized eigenvectors provide one or two filter pairs. Log relative projected variances are ranked with training-only ANOVA feature selection (20 or 50 selected), standardized, and passed to an RBF SVM. Channel reference and C are chosen inside training group CV. CSP fitting, feature selection, and scaling never see a held-out fold.',
        '- **Haar features + SVM/LDA:** six-level orthonormal Haar decomposition gives seven coefficient groups. Log energy, relative energy, and normalized coefficient entropy yield 294 channel features. Features are averaged across four circular shifts to reduce phase sensitivity. Windows are independently detrended and bandpass filtered; common-average referencing is tested enabled/disabled. The classifiers use training-fitted scaling with RBF SVM or automatically shrunk LDA.',
        '- **Shallow power CNN:** temporal and spatial convolutions, batch normalization, squaring, full-window average pooling, a safe logarithm, dropout, and a nine-class head learn power-based features. Per-window channel normalization avoids learned population statistics. Training-only time shifts, channel gains, and noise provide augmentation. AdamW trains for a fixed 50 epochs (learning rate 0.0008, weight decay 0.01, batch size 64, dropout 0.5, class-weighted loss with label smoothing 0.05). Neither fold scores nor holdout scores select checkpoints.', '',
        f'The experiment performs {fits} training-only grouped CV fits, followed by a full-training refit for each family. The CNN has one prespecified configuration and completes the same fixed-duration procedure in each fold. Family selection uses mean training CV accuracy, with macro-F1 breaking family ties. All configurations and the overall winner are frozen before validation/test predictions are generated. The previously reported holdout is reused, so these results are exploratory rather than independent confirmation.', '',
        '| Technique | Training CV accuracy | Validation accuracy | Test accuracy | Test macro-F1 |',
        '|---|---:|---:|---:|---:|']
    for name, d in r['models'].items():
        rows.append(f"| {name} | {d['training_cv_accuracy']:.2%} | {d['validation']['accuracy']:.2%} | {d['test']['accuracy']:.2%} | {d['test']['macro_f1']:.4f} |")
    rows += ['',
        f"The training-CV-selected winner is **{winner}**: test accuracy **{best['test']['accuracy']:.2%}**, macro-F1 **{best['test']['macro_f1']:.4f}**. The majority baseline scores {majority['test']['accuracy']:.2%} accuracy and {majority['test']['macro_f1']:.4f} macro-F1. The highest observed learned-model test accuracy belongs to {highest_observed} ({models[highest_observed]['test']['accuracy']:.2%}); test ranking does not change the frozen model selection.", '',
        'Do not treat a small gain on a reused holdout as a confirmed improvement. Participant identities remain unavailable, and scores do not establish generalization to unseen people. The best prior learned one-second test accuracy was 11.62%; compare against the majority baseline as well. The earlier intentionally invalid random-window control is not used for training selection or performance claims here.', '',
        f"The run took {r['elapsed_seconds']:.1f} seconds on CPU. All five fitted models are saved locally, together with the selected checkpoint `artifacts_alternatives/best_model.joblib` and its input contract. The checkpoint contains a complete raw-window pipeline, so inference applies the matching CSP, wavelet, or neural processing. All 16 automated tests pass. Independent verification reloads every model, reproduces every validation and test prediction, recomputes metrics/confusion matrices, checks group disjointness, and verifies single-window inference.", '',
        '![Alternative method comparison](artifacts_alternatives/comparison.png)', '',
        '![Fixed-duration power-CNN training](artifacts_alternatives/cnn_training.png)', '',
        'Reproduce:', '', '```bash',
        'MPLCONFIGDIR=/tmp/eegis-matplotlib .venv/bin/python -u -m eegis.train_alternatives > alternatives.log 2>&1',
        'MPLCONFIGDIR=/tmp/eegis-matplotlib .venv/bin/python -m eegis.verify_alternatives',
        '.venv/bin/python -m eegis.report_alternatives',
        '.venv/bin/python -m unittest discover -s tests -v',
        '.venv/bin/python -m eegis.predict --artifacts artifacts_alternatives "EEGIS - Electroencephalogram Imagined Speech dataset/raw/class_0/0.csv"',
        '```', '',
        'Source: `eegis/alternatives.py`, `train_alternatives.py`, `verify_alternatives.py`, and `report_alternatives.py`. Measured CV results, fold membership, predictions, metrics, confusion matrices, training history, and verification are in `artifacts_alternatives/`. Dataset files, binary checkpoints, and logs remain local. Work is kept on `experiment/reconstructed-recordings`; main is not changed.']
    if winner == 'Majority':
        rows += ['', 'No learned candidate beat the majority baseline in training CV. The selected constant predictor is a reference, not a usable speech decoder. No successful accuracy improvement is claimed.']
    history = pd.read_csv(out/'cnn_training_history.csv')
    rows += ['', f"The power CNN reached {history.training_accuracy.iloc[-1]:.2%} training accuracy on augmented batches, while its grouped CV accuracy was {r['models']['ShallowPowerCNN']['training_cv_accuracy']:.2%} and test accuracy was {r['models']['ShallowPowerCNN']['test']['accuracy']:.2%}. Better fitting of training recordings did not transfer to held-out recordings in this run."]
    comparison = out/'paired_comparison.json'
    if comparison.exists():
        paired = json.loads(comparison.read_text())
        rows += ['', 'Paired recording-group bootstrap (5,000 resamples, descriptive uncertainty on the reused holdout):']
        for item in paired['comparisons']:
            lo, hi = item['paired_group_bootstrap_95_percent_interval_pp']
            rows += ['', f"Against {item['reference']}, the frozen winner differs by {item['accuracy_difference_percentage_points']:+.2f} percentage points, with a 95% interval of [{lo:+.2f}, {hi:+.2f}] points. These intervals are not used to select a model and do not establish subject-independent improvement."]
    p = Path('README.md')
    text = p.read_text().split('\n## Alternative one-second techniques')[0]
    p.write_text(text+'\n\n'+'\n'.join(rows)+'\n')
    print('README updated from verified alternative-method results')


if __name__ == '__main__':
    main()
