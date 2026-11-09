"""Prespecified alternative one-second methods, retaining the original split."""
import argparse
import copy
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .alternatives import FilterBankCSP, HaarFeatures, ShallowPowerClassifier
from .data import CHANNELS, LABELS, audit_shared_samples
from .epochs import read_window
from .improve import score


def candidates():
    return [
        ('FBCSP_SVM', Pipeline([('csp', FilterBankCSP()), ('select', SelectKBest(f_classif)),
                               ('scale', StandardScaler()), ('model', SVC(class_weight='balanced'))]),
         {'csp__pairs': [1, 2], 'csp__reference': ['none', 'car'],
          'select__k': [20, 50], 'model__C': [.1, 1]}),
        ('HaarSVM', Pipeline([('wavelet', HaarFeatures()), ('scale', StandardScaler()),
                              ('model', SVC(class_weight='balanced'))]),
         {'wavelet__reference': ['none', 'car'], 'model__C': [.1, 1, 10]}),
        ('HaarLDA', Pipeline([('wavelet', HaarFeatures()), ('scale', StandardScaler()),
                              ('model', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))]),
         {'wavelet__reference': ['none', 'car']}),
        ('ShallowPowerCNN', ShallowPowerClassifier(), {}),
        ('Majority', Pipeline([('wavelet', HaarFeatures()), ('model', DummyClassifier(strategy='most_frequent'))]), {})]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='EEGIS - Electroencephalogram Imagined Speech dataset')
    parser.add_argument('--manifest', default='artifacts/manifest.csv')
    parser.add_argument('--out', default='artifacts_alternatives')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True, parents=True)
    started = time.time(); m = pd.read_csv(args.manifest)
    if m.path.duplicated().any() or m.groupby('overlap_group').split.nunique().max() != 1:
        raise ValueError('Invalid source split manifest')
    m.to_csv(out/'manifest.csv', index=False)
    plan = {'seed': 42, 'selection': 'highest training-only grouped three-fold CV accuracy; family macro-F1 tie-break',
            'evaluation': 'original nine-class one-second task; original split unchanged',
            'holdout_policy': 'freeze all hyperparameters and family winner before evaluating validation or test',
            'test_reused': True, 'subject_independent': False,
            'grids': {name: grid for name, _, grid in candidates()},
            'cnn': ShallowPowerClassifier().get_params(),
            'new_techniques': ['supervised one-vs-rest filter-bank CSP', 'cycle-spun Haar features', 'fixed-duration square/log power CNN']}
    (out/'plan.json').write_text(json.dumps(plan, indent=2))
    print('Loading frozen one-second dataset...', flush=True)
    x = np.stack([read_window(Path(args.data)/p) for p in m.path])
    audit = audit_shared_samples(x, m)
    (out/'leakage_audit.json').write_text(json.dumps(audit, indent=2))
    y = m.label.to_numpy(); groups = m.overlap_group.to_numpy()
    ix = {s: np.flatnonzero(m.split == s) for s in ['train', 'validation', 'test']}
    tr = ix['train']
    cv = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(x[tr], y[tr], groups[tr]))
    cv_manifest = m.iloc[tr][['path', 'overlap_group', 'label']].copy()
    cv_manifest['cv_fold'] = -1
    for fold, (train, heldout) in enumerate(cv):
        assert set(groups[tr][train]).isdisjoint(set(groups[tr][heldout]))
        cv_manifest.iloc[heldout, cv_manifest.columns.get_loc('cv_fold')] = fold
    cv_manifest.to_csv(out/'training_cv_manifest.csv', index=False)
    models, details = {}, {}
    with threadpool_limits(limits=4):
        for name, estimator, grid in candidates():
            print('Training-only CV:', name, flush=True)
            search = GridSearchCV(estimator, grid, scoring={'accuracy': 'accuracy', 'macro_f1': 'f1_macro'},
                                  refit='accuracy', cv=cv, n_jobs=1, error_score='raise', verbose=1)
            search.fit(x[tr], y[tr])
            pd.DataFrame(search.cv_results_).to_csv(out/f'{name}_cv.csv', index=False)
            index = search.best_index_
            details[name] = {'training_cv_accuracy': float(search.best_score_),
                             'training_cv_macro_f1': float(search.cv_results_['mean_test_macro_f1'][index]),
                             'parameters': search.best_params_}
            models[name] = search.best_estimator_
            joblib.dump(models[name], out/f'{name}.joblib')
            print(name, details[name], flush=True)
        winner = max(details, key=lambda n: (details[n]['training_cv_accuracy'], details[n]['training_cv_macro_f1']))
        selection = {'winner': winner, 'criterion': plan['selection'], 'candidates': copy.deepcopy(details)}
        (out/'selection.json').write_text(json.dumps(selection, indent=2))
        joblib.dump(models[winner], out/'best_model.joblib')
        (out/'best_model.json').write_text(json.dumps({'model': winner, 'file': 'best_model.joblib',
             'input_transform': 'raw_window_pipeline', 'channels': CHANNELS, 'labels': LABELS,
             'sample_rate_hz': 128, 'input_shape': [128, 14]}, indent=2, ensure_ascii=False))
        for split in ['validation', 'test']:
            indices = ix[split]; predictions = m.loc[indices, ['path', 'label', 'overlap_group']].copy()
            for name, model in models.items():
                p = model.predict(x[indices]); predictions[name] = p
                details[name][split] = score(y[indices], p)
                if split == 'test':
                    details[name]['classification_report'] = classification_report(y[indices], p, labels=range(9), target_names=LABELS, output_dict=True, zero_division=0)
                    np.savetxt(out/f'{name}_confusion.csv', confusion_matrix(y[indices], p, labels=range(9)), delimiter=',', fmt='%d')
                print(split.upper(), name, details[name][split], flush=True)
            predictions.to_csv(out/f'{split}_predictions.csv', index=False)
    history = pd.DataFrame(models['ShallowPowerCNN'].history_)
    history.to_csv(out/'cnn_training_history.csv', index=False)
    results = {'best_model': winner, 'selection_criterion': plan['selection'], 'models': details,
               'split_sizes': {s: len(i) for s, i in ix.items()}, 'arguments': vars(args),
               'elapsed_seconds': time.time()-started,
               'limitations': 'Previously reported holdout reused; participants unidentified; no confirmed subject-independent improvement'}
    (out/'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
    table = pd.DataFrame([{'model': name, 'training_cv_accuracy': d['training_cv_accuracy'],
                          'validation_accuracy': d['validation']['accuracy'],
                          'test_accuracy': d['test']['accuracy'], 'test_macro_f1': d['test']['macro_f1']} for name, d in details.items()])
    table.to_csv(out/'comparison.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    for name in models:
        fig, ax = plt.subplots(figsize=(8, 7))
        cm = np.loadtxt(out/f'{name}_confusion.csv', delimiter=',').astype(int)
        ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(f'{name}: original held-out test windows'); fig.tight_layout()
        fig.savefig(out/f'{name}_confusion.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.epoch, history.training_loss); axes[0].set_ylabel('Training loss')
    axes[1].plot(history.epoch, history.training_accuracy); axes[1].set_ylabel('Training accuracy')
    for ax in axes: ax.set_xlabel('Epoch')
    fig.tight_layout(); fig.savefig(out/'cnn_training.png', dpi=160); plt.close(fig)
    ax = table.set_index('model')[['training_cv_accuracy', 'test_accuracy', 'test_macro_f1']].plot.bar(rot=20, ylim=(0, .5))
    ax.figure.tight_layout(); ax.figure.savefig(out/'comparison.png', dpi=160); plt.close(ax.figure)
    print('Frozen CV-selected winner:', winner, 'seconds:', results['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
