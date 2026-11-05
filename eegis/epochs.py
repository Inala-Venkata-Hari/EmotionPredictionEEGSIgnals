"""Reconstruct complete overlap components and classify 7.875-second recordings."""
import argparse
import copy
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .data import BANDS, CHANNELS, LABELS, audit_shared_samples, digest
from .improve import TangentFeatures, score


def read_window(path):
    frame = pd.read_csv(path)
    if frame.shape != (128, 15) or list(frame.columns[1:]) != CHANNELS:
        raise ValueError(f'Unexpected EEGIS CSV layout: {path}')
    if not np.array_equal(frame.iloc[:, 0], np.arange(128)):
        raise ValueError(f'Invalid sample index: {path}')
    x = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f'Nonfinite signal: {path}')
    return x


def reconstruct(windows):
    """Recover chronological order using exact samples, with no filenames or labels."""
    x = np.asarray(windows, dtype=np.float64)
    if x.shape != (12, 128, 14) or not np.isfinite(x).all():
        raise ValueError('Expected exactly 12 finite windows of shape (128, 14)')
    if len({digest(a) for a in x}) != len(x):
        raise ValueError('Duplicate windows are not a complete recording')
    prefixes = {digest(a[:48]): i for i, a in enumerate(x)}
    suffixes = {digest(a[-48:]): i for i, a in enumerate(x)}
    if len(prefixes) != 12 or len(suffixes) != 12:
        raise ValueError('Branched or ambiguous overlap chain')
    heads = set(prefixes) - set(suffixes)
    if len(heads) != 1:
        raise ValueError('Recording must have exactly one starting window')
    current = prefixes[heads.pop()]
    order = []
    while current not in order:
        order.append(current)
        following = prefixes.get(digest(x[current, -48:]))
        if following is None:
            break
        if not np.array_equal(x[current, -48:], x[following, :48]):
            raise ValueError('Overlap hash collision')
        current = following
    if len(order) != 12 or prefixes.get(digest(x[order[-1], -48:])) is not None:
        raise ValueError('Disconnected or cyclic overlap chain')
    recording = np.concatenate([x[order[0]], *[x[i, 48:] for i in order[1:]]], axis=0)
    if recording.shape != (1008, 14):
        raise AssertionError('Unexpected reconstructed duration')
    for position, window in enumerate(order):
        np.testing.assert_array_equal(recording[position*80:position*80+128], x[window])
    return recording, order


def load_recordings(data, manifest):
    recordings, rows = [], []
    for group, frame in manifest.groupby('overlap_group', sort=True):
        if frame.label.nunique() != 1 or frame.split.nunique() != 1:
            raise ValueError('Mixed labels or partitions in an overlap component')
        windows = np.stack([read_window(Path(data)/p) for p in frame.path])
        recording, order = reconstruct(windows)
        recordings.append(recording)
        paths = frame.path.to_numpy()
        rows.append({'overlap_group': int(group), 'label': int(frame.label.iloc[0]),
                     'split': frame.split.iloc[0], 'source_paths': json.dumps(paths[order].tolist()),
                     'samples': len(recording), 'sha256': digest(recording)})
    return np.stack(recordings), pd.DataFrame(rows)


class RecordingFeatures(TransformerMixin, BaseEstimator):
    """Per-recording transforms; no fitted population statistics or labels."""
    def __init__(self, kind='spectral', reference='none'):
        self.kind = kind
        self.reference = reference

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 3 or x.shape[1:] != (1008, 14) or not np.isfinite(x).all():
            raise ValueError('Expected recordings with shape (N, 1008, 14)')
        z = signal.detrend(x, axis=1, type='linear')
        if self.reference == 'car':
            z = z - z.mean(axis=2, keepdims=True)
        elif self.reference != 'none':
            raise ValueError('Unknown reference')
        if self.kind == 'covariance':
            matrices = []
            for band in [(0.5, 40), *BANDS]:
                sos = signal.butter(4, band, btype='bandpass', fs=128, output='sos')
                a = signal.sosfiltfilt(sos, z, axis=1)
                a -= a.mean(axis=1, keepdims=True)
                c = np.einsum('ntc,ntd->ncd', a, a)/(len(a[0])-1)
                scale = (np.trace(c, axis1=-2, axis2=-1)/14).clip(1e-12)
                matrices.append(.9*c + .1*scale[:, None, None]*np.eye(14))
            return np.stack(matrices, axis=1)
        if self.kind != 'spectral':
            raise ValueError('Unknown feature kind')
        sos = signal.butter(4, [.5, 40], btype='bandpass', fs=128, output='sos')
        a = signal.sosfiltfilt(sos, z, axis=1)
        frequencies, psd = signal.welch(a, fs=128, nperseg=256, noverlap=128, axis=1)
        # Bin-energy sum: half-open bands ensure no shared boundary bins.
        powers = np.stack([psd[:, (frequencies >= lo) & (frequencies < hi)].sum(axis=1)*.5
                           for lo, hi in BANDS], axis=1)
        relative = powers/(powers.sum(axis=1, keepdims=True)+1e-12)
        variance = a.var(axis=1)+1e-12
        first = np.diff(a, axis=1).var(axis=1)+1e-12
        second = np.diff(a, n=2, axis=1).var(axis=1)+1e-12
        mobility = np.sqrt(first/variance)
        complexity = np.sqrt(second/first)/(mobility+1e-12)
        p = psd[:, (frequencies >= .5) & (frequencies < 40)]
        p = p/(p.sum(axis=1, keepdims=True)+1e-12)
        entropy = -(p*np.log(p+1e-12)).sum(axis=1)/np.log(p.shape[1])
        result = np.concatenate([np.log(powers+1e-12).reshape(len(a), -1),
                                 relative.reshape(len(a), -1), np.log(variance),
                                 mobility, complexity, entropy], axis=1)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite recording features')
        return result


def candidates():
    return [
        ('SpectralLDA', Pipeline([('features', RecordingFeatures()), ('scale', StandardScaler()),
          ('model', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))]),
          {'features__reference': ['none', 'car']}),
        ('SpectralSVM', Pipeline([('features', RecordingFeatures()), ('scale', StandardScaler()),
          ('pca', PCA(svd_solver='full')), ('model', SVC())]),
          {'features__reference': ['none', 'car'], 'pca__n_components': [20, 50],
           'model__C': [.1, 1, 10], 'model__kernel': ['rbf']}),
        ('SpectralLogistic', Pipeline([('features', RecordingFeatures()), ('scale', StandardScaler()),
          ('model', LogisticRegression(max_iter=2000))]),
          {'features__reference': ['none', 'car'], 'model__C': [.01, .1, 1]}),
        ('SpatialLDA', Pipeline([('features', RecordingFeatures(kind='covariance')),
          ('tangent', TangentFeatures()), ('scale', StandardScaler()),
          ('model', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))]),
          {'features__reference': ['none', 'car'], 'tangent__normalize': [False, True]})]


def predict_recording(paths, artifacts='artifacts_epochs'):
    root = Path(artifacts)
    info = json.loads((root/'best_model.json').read_text())
    recording, _ = reconstruct(np.stack([read_window(p) for p in paths]))
    model = joblib.load(root/info['file'])
    model_input = recording.reshape(1, -1) if info.get('flatten') else recording[None]
    with threadpool_limits(limits=4):
        label = int(model.predict(model_input)[0])
    return {'model': info['model'], 'class_id': label, 'label': LABELS[label], 'seconds': 7.875}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='EEGIS - Electroencephalogram Imagined Speech dataset')
    parser.add_argument('--manifest', default='artifacts/manifest.csv')
    parser.add_argument('--out', default='artifacts_epochs')
    parser.add_argument('--predict', nargs='+', help='Exactly 12 CSVs from a complete overlap chain')
    args = parser.parse_args()
    if args.predict:
        print(json.dumps(predict_recording(args.predict, args.out), ensure_ascii=False))
        return
    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)
    started = time.time()
    original = pd.read_csv(args.manifest)
    if original.path.duplicated().any() or original.groupby('overlap_group').split.nunique().max() != 1:
        raise ValueError('Invalid source split manifest')
    x, records = load_recordings(args.data, original)
    records.to_csv(out/'manifest.csv', index=False)
    audit = audit_shared_samples(x, records)
    audit.update({'recordings': len(records), 'samples_each': 1008,
                  'all_original_windows_reproduced_exactly': True,
                  'unique_recordings': int(records.sha256.nunique()),
                  'source_manifest_sha256': __import__('hashlib').sha256(Path(args.manifest).read_bytes()).hexdigest()})
    (out/'reconstruction_audit.json').write_text(json.dumps(audit, indent=2))
    ix = {s: np.flatnonzero(records.split == s) for s in ['train', 'validation', 'test']}
    tr, va, te = [ix[s] for s in ix]
    y = records.label.to_numpy()
    cv = list(StratifiedKFold(3, shuffle=True, random_state=42).split(x[tr], y[tr]))
    # One reconstructed recording per overlap component: CV rows are independent of overlap.
    print('Reconstructed partitions:', {s: len(i) for s, i in ix.items()}, flush=True)
    plan = {'selection': 'highest mean training-only three-fold CV accuracy; macro-F1 tie-break',
            'seed': 42, 'test_policy': 'evaluate only the frozen winner and the training-majority baseline',
            'evaluation_unit': 'one reconstructed 7.875-second overlap component',
            'subject_independent': False, 'test_previously_reported': True,
            'candidates': {name: grid for name, _, grid in candidates()}}
    (out/'plan.json').write_text(json.dumps(plan, indent=2))
    models, summary = {}, {}
    with threadpool_limits(limits=4):
        for name, model, grid in candidates():
            search = GridSearchCV(model, grid, scoring={'accuracy': 'accuracy', 'macro_f1': 'f1_macro'},
                                  refit='accuracy', cv=cv, n_jobs=1, error_score='raise')
            search.fit(x[tr], y[tr])
            pd.DataFrame(search.cv_results_).to_csv(out/f'{name}_cv.csv', index=False)
            best = search.best_index_
            summary[name] = {'training_cv_accuracy': float(search.best_score_),
                             'training_cv_macro_f1': float(search.cv_results_['mean_test_macro_f1'][best]),
                             'parameters': search.best_params_}
            models[name] = search.best_estimator_
            print(name, summary[name], flush=True)
        dummy = DummyClassifier(strategy='most_frequent')
        baseline_cv = cross_validate(dummy, x[tr].reshape(len(tr), -1), y[tr], cv=cv,
                                     scoring={'accuracy': 'accuracy', 'macro_f1': 'f1_macro'})
        summary['Majority'] = {'training_cv_accuracy': float(baseline_cv['test_accuracy'].mean()),
                               'training_cv_macro_f1': float(baseline_cv['test_macro_f1'].mean()), 'parameters': {}}
        dummy.fit(x[tr].reshape(len(tr), -1), y[tr])
        models['Majority'] = dummy
        winner = max(summary, key=lambda n: (summary[n]['training_cv_accuracy'], summary[n]['training_cv_macro_f1']))
        selection = {'winner': winner, 'criterion': plan['selection'], 'candidates': copy.deepcopy(summary)}
        (out/'selection.json').write_text(json.dumps(selection, indent=2))
        joblib.dump(models[winner], out/'best_model.joblib')
        meta = {'model': winner, 'file': 'best_model.joblib', 'input': 'one recording reconstructed from 12 windows',
                'shape': [1008, 14], 'sample_rate_hz': 128, 'seconds': 7.875,
                'channels': CHANNELS, 'labels': LABELS, 'flatten': winner == 'Majority'}
        (out/'best_model.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        # Descriptive negative control at fixed hyperparameters; not a formal permutation significance test.
        control = []
        if winner != 'Majority':
            rng = np.random.default_rng(42)
            for _ in range(5):
                shuffled = rng.permutation(y[tr])
                control.append(float(cross_val_score(clone(models[winner]), x[tr], shuffled, cv=cv, scoring='accuracy').mean()))
        predictions, evaluation = {}, {}
        for split in ['validation', 'test']:
            indices = ix[split]
            a = x[indices].reshape(len(indices), -1) if winner == 'Majority' else x[indices]
            p = models[winner].predict(a)
            baseline = dummy.predict(x[indices].reshape(len(indices), -1))
            evaluation[split] = {'selected': score(y[indices], p), 'majority': score(y[indices], baseline)}
            predictions[split] = records.loc[indices, ['overlap_group', 'label']].copy()
            predictions[split]['prediction'] = p
            predictions[split]['majority'] = baseline
            predictions[split].to_csv(out/f'{split}_predictions.csv', index=False)
        report = {'best_model': winner, 'training_candidates': summary, 'scores': evaluation,
                  'label_shuffle_cv_accuracies': control, 'label_shuffle_note': 'Fixed selected hyperparameters; descriptive only, not a formal significance test',
                  'test_report': classification_report(y[te], predictions['test'].prediction, labels=range(9), target_names=LABELS, output_dict=True, zero_division=0),
                  'split_sizes': {s: len(i) for s, i in ix.items()}, 'arguments': vars(args),
                  'elapsed_seconds': time.time()-started, 'limitations': 'Reused holdout; recording-level task; participants unidentified; not subject-independent'}
        (out/'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
        cm = confusion_matrix(y[te], predictions['test'].prediction, labels=range(9))
        np.savetxt(out/'confusion.csv', cm, delimiter=',', fmt='%d')
        print('Selected', winner, 'measured scores:', evaluation, flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f'{winner}: 66 held-out reconstructed recordings')
    fig.tight_layout(); fig.savefig(out/'confusion.png', dpi=160); plt.close(fig)
    table = pd.DataFrame([{'model': name, 'training_cv_accuracy': v['training_cv_accuracy']} for name, v in summary.items()])
    table.to_csv(out/'comparison.csv', index=False)
    ax = table.set_index('model').plot.bar(rot=25, ylim=(0, .5), legend=False, ylabel='Training-only CV accuracy')
    ax.figure.tight_layout(); ax.figure.savefig(out/'training_cv.png', dpi=160); plt.close(ax.figure)


if __name__ == '__main__':
    # Keep serialized custom transformer classes importable after module execution.
    from eegis.epochs import main
    main()
