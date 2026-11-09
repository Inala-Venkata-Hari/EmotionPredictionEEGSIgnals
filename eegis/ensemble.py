"""Group-safe hard voting, calibrated soft voting, and nested out-of-fold stacking."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import CHANNELS, LABELS, audit_shared_samples
from .epochs import read_window
from .improve import score

MEMBERS = ['FBCSP_SVM', 'HaarSVM', 'HaarLDA', 'ShallowPowerCNN']
METHODS = ['HardVote', 'CalibratedSoftVote', 'Stacking', 'Majority']


def assert_group_separation(groups, train, heldout, forbidden=None):
    a, b = set(groups[train]), set(groups[heldout])
    if a & b:
        raise ValueError('Overlap groups cross fit and prediction partitions')
    if forbidden is not None and (a | b) & set(forbidden):
        raise ValueError('An outer held-out group entered inner meta-feature generation')


def raw_scores(model, x):
    if not np.array_equal(model.classes_, np.arange(9)):
        raise ValueError('Every base model must retain all nine classes in order')
    if hasattr(model, 'predict_proba'):
        return np.log(np.clip(model.predict_proba(x), 1e-8, 1))
    result = np.asarray(model.decision_function(x))
    if result.shape != (len(x), 9) or not np.isfinite(result).all():
        raise ValueError('Invalid base decision scores')
    return result


def fit_meta(scores, labels):
    """Only group-excluded predictions may be supplied here."""
    calibrators = []
    for k in range(scores.shape[1]):
        calibrator = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=2000))
        calibrator.fit(scores[:, k], labels)
        calibrators.append(calibrator)
    stacker = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=2000))
    stacker.fit(scores.reshape(len(scores), -1), labels)
    return calibrators, stacker


def method_probabilities(scores, hard_labels, calibrators, stacker, prior):
    n = len(scores)
    votes = np.eye(9)[hard_labels].sum(axis=1)
    # Training class priors break exact vote ties; class number never breaks a non-tie.
    hard = votes + 1e-8*np.asarray(prior)[None]
    hard /= hard.sum(axis=1, keepdims=True)
    soft = np.mean([cal.predict_proba(scores[:, k]) for k, cal in enumerate(calibrators)], axis=0)
    stack = stacker.predict_proba(scores.reshape(n, -1))
    majority = np.zeros((n, 9)); majority[:, int(np.argmax(prior))] = 1
    return dict(zip(METHODS, [hard, soft, stack, majority]))


class GroupedEnsemble:
    """Serializable full base models and meta-estimators, with a frozen mode."""
    def __init__(self, models, calibrators, stacker, prior, method):
        self.models = models
        self.calibrators = calibrators
        self.stacker = stacker
        self.prior = prior
        self.method = method
        self.classes_ = np.arange(9)

    def predict_all(self, x):
        with threadpool_limits(limits=4):
            scores = np.stack([raw_scores(self.models[name], x) for name in MEMBERS], axis=1)
            hard = np.stack([self.models[name].predict(x) for name in MEMBERS], axis=1)
            return method_probabilities(scores, hard, self.calibrators, self.stacker, self.prior)

    def predict_proba(self, x):
        return self.predict_all(x)[self.method]

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)


def fit_predictions(templates, x, y, groups, train, heldout, audit, phase, outer, forbidden=None):
    assert_group_separation(groups, train, heldout, forbidden)
    scores, hard = [], []
    for name in MEMBERS:
        print('Fit', phase, 'outer', outer, name, 'windows', len(train), flush=True)
        model = clone(templates[name]).fit(x[train], y[train])
        scores.append(raw_scores(model, x[heldout]))
        hard.append(model.predict(x[heldout]))
        audit.append({'phase': phase, 'outer_fold': outer, 'member': name,
                      'fit_groups': sorted(map(int, np.unique(groups[train]))),
                      'prediction_groups': sorted(map(int, np.unique(groups[heldout]))),
                      'forbidden_outer_groups': sorted(map(int, [] if forbidden is None else forbidden))})
    return np.stack(scores, axis=1), np.stack(hard, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='EEGIS - Electroencephalogram Imagined Speech dataset')
    parser.add_argument('--base', default='artifacts_alternatives')
    parser.add_argument('--out', default='artifacts_ensembles')
    args = parser.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    started = time.time(); base = Path(args.base)
    m = pd.read_csv(base/'manifest.csv')
    pd.testing.assert_frame_equal(m, pd.read_csv('artifacts/manifest.csv'))
    m.to_csv(out/'manifest.csv', index=False)
    x = np.stack([read_window(Path(args.data)/p) for p in m.path])
    outer_audit = audit_shared_samples(x, m)
    (out/'leakage_audit.json').write_text(json.dumps(outer_audit, indent=2))
    full_models = {name: joblib.load(base/f'{name}.joblib') for name in MEMBERS}
    templates = {name: clone(model) for name, model in full_models.items()}
    selection_base = json.loads((base/'selection.json').read_text())
    plan = {'members': MEMBERS, 'methods': METHODS, 'outer_folds': 3, 'inner_folds': 2,
            'seed': 42, 'calibration_and_stacker_C': .1,
            'selection': 'nested training-group CV accuracy; macro-F1 tie-break',
            'base_parameters': {name: selection_base['candidates'][name]['parameters'] for name in MEMBERS},
            'base_checkpoint_sha256': {name: hashlib.sha256((base/f'{name}.joblib').read_bytes()).hexdigest() for name in MEMBERS},
            'limitations': 'Base hyperparameters were previously selected on the training set. Nested fits prevent meta-feature leakage, but CV remains a selection estimate. Previously reported holdout reused; participant identifiers unavailable.',
            'test_used_for_selection': False}
    (out/'plan.json').write_text(json.dumps(plan, indent=2))
    train_indices = np.flatnonzero(m.split == 'train')
    xt = x[train_indices]; yt = m.label.to_numpy()[train_indices]; gt = m.overlap_group.to_numpy()[train_indices]
    outer = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(xt, yt, gt))
    oof_scores = np.full((len(xt), 4, 9), np.nan)
    oof_hard = np.full((len(xt), 4), -1, dtype=int)
    ensemble_oof = {name: np.full(len(xt), -1, dtype=int) for name in METHODS}
    audits, assignments = [], []
    with threadpool_limits(limits=4):
        for fold, (tr, va) in enumerate(outer):
            print('OUTER GROUP FOLD', fold, flush=True)
            assert_group_separation(gt, tr, va)
            inner_scores = np.full((len(tr), 4, 9), np.nan)
            inner = StratifiedGroupKFold(2, shuffle=True, random_state=100+fold)
            inner_fold = np.full(len(xt), -1, dtype=int)
            for inner_id, (a, b) in enumerate(inner.split(xt[tr], yt[tr], gt[tr])):
                inner_fold[tr[b]] = inner_id
                scores, _ = fit_predictions(templates, xt, yt, gt, tr[a], tr[b], audits,
                                            f'inner_{inner_id}', fold, np.unique(gt[va]))
                inner_scores[b] = scores
            if not np.isfinite(inner_scores).all():
                raise AssertionError('Missing inner group-excluded predictions')
            calibrators, stacker = fit_meta(inner_scores, yt[tr])
            scores, hard = fit_predictions(templates, xt, yt, gt, tr, va, audits, 'outer', fold)
            oof_scores[va] = scores; oof_hard[va] = hard
            prior = np.bincount(yt[tr], minlength=9)/len(tr)
            for name, probability in method_probabilities(scores, hard, calibrators, stacker, prior).items():
                ensemble_oof[name][va] = probability.argmax(axis=1)
            for group in np.unique(gt):
                positions = np.flatnonzero(gt == group)
                role = 'heldout' if group in set(gt[va]) else 'train'
                assignments.append({'outer_fold': fold, 'overlap_group': int(group),
                                    'outer_role': role, 'inner_validation_fold': int(inner_fold[positions[0]])})
            print('Outer scores so far', fold, {name: score(yt[va], p[va]) for name, p in ensemble_oof.items()}, flush=True)
        if not np.isfinite(oof_scores).all() or (oof_hard < 0).any():
            raise AssertionError('Missing final out-of-group base predictions')
        cv_scores = {name: score(yt, p) for name, p in ensemble_oof.items()}
        winner = max(cv_scores, key=lambda n: (cv_scores[n]['accuracy'], cv_scores[n]['macro_f1']))
        (out/'selection.json').write_text(json.dumps({'winner': winner, 'criterion': plan['selection'], 'training_cv': cv_scores}, indent=2))
        (out/'fit_audit.json').write_text(json.dumps(audits, indent=2))
        pd.DataFrame(assignments).to_csv(out/'nested_cv_group_assignments.csv', index=False)
        saved_oof = m.iloc[train_indices][['path', 'label', 'overlap_group']].copy()
        for k, name in enumerate(MEMBERS): saved_oof[name] = oof_hard[:, k]
        for name, p in ensemble_oof.items(): saved_oof[name] = p
        saved_oof.to_csv(out/'training_oof_predictions.csv', index=False)
        # The score matrix contains derived model outputs, not raw signals; kept locally only.
        np.savez_compressed(out/'training_oof_scores.npz', scores=oof_scores, labels=yt)
        calibrators, stacker = fit_meta(oof_scores, yt)
        prior = np.bincount(yt, minlength=9)/len(yt)
        ensemble = GroupedEnsemble(full_models, calibrators, stacker, prior, winner)
        joblib.dump(ensemble, out/'best_model.joblib')
        (out/'best_model.json').write_text(json.dumps({'model': winner, 'file': 'best_model.joblib', 'input_transform': 'raw_window_pipeline',
             'channels': CHANNELS, 'labels': LABELS, 'sample_rate_hz': 128, 'input_shape': [128, 14]}, indent=2, ensure_ascii=False))
        details = {name: {'training_cv': cv_scores[name]} for name in METHODS}
        for split in ['validation', 'test']:
            indices = np.flatnonzero(m.split == split)
            predictions = m.loc[indices, ['path', 'label', 'overlap_group']].copy()
            for name, probability in ensemble.predict_all(x[indices]).items():
                p = probability.argmax(axis=1); predictions[name] = p
                details[name][split] = score(predictions.label, p)
                if split == 'test':
                    details[name]['predicted_classes'] = int(len(np.unique(p)))
                    details[name]['classification_report'] = classification_report(predictions.label, p, labels=range(9), target_names=LABELS, output_dict=True, zero_division=0)
                    np.savetxt(out/f'{name}_confusion.csv', confusion_matrix(predictions.label, p, labels=range(9)), delimiter=',', fmt='%d')
                print(split, name, details[name][split], flush=True)
            predictions.to_csv(out/f'{split}_predictions.csv', index=False)
        diversity = []
        for a in range(4):
            for b in range(a+1, 4):
                ea, eb = oof_hard[:, a] != yt, oof_hard[:, b] != yt
                diversity.append({'member_a': MEMBERS[a], 'member_b': MEMBERS[b],
                                  'disagreement': float(np.mean(oof_hard[:, a] != oof_hard[:, b])),
                                  'both_wrong': float(np.mean(ea & eb)), 'error_correlation': float(np.corrcoef(ea, eb)[0, 1])})
        pd.DataFrame(diversity).to_csv(out/'diversity.csv', index=False)
    report = {'best_model': winner, 'models': details, 'arguments': vars(args), 'elapsed_seconds': time.time()-started,
              'base_models': MEMBERS, 'limitations': plan['limitations']}
    (out/'results.json').write_text(json.dumps(report, indent=2))
    table = pd.DataFrame([{'method': name, 'training_cv_accuracy': d['training_cv']['accuracy'],
                          'validation_accuracy': d['validation']['accuracy'], 'test_accuracy': d['test']['accuracy'],
                          'test_macro_f1': d['test']['macro_f1']} for name, d in details.items()])
    table.to_csv(out/'comparison.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    for name in METHODS:
        cm = np.loadtxt(out/f'{name}_confusion.csv', delimiter=',').astype(int)
        fig, ax = plt.subplots(figsize=(8, 7)); ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(name); fig.tight_layout(); fig.savefig(out/f'{name}_confusion.png', dpi=160); plt.close(fig)
    ax = table.set_index('method')[['training_cv_accuracy', 'test_accuracy', 'test_macro_f1']].plot.bar(rot=20, ylim=(0, .5))
    ax.figure.tight_layout(); ax.figure.savefig(out/'comparison.png', dpi=160); plt.close(ax.figure)
    print('Frozen ensemble winner:', winner, flush=True)


if __name__ == '__main__':
    from eegis.ensemble import main
    main()
