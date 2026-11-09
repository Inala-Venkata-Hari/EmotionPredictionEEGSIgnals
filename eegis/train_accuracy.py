"""Training-only nested grouped comparisons for binary EEG accuracy research.

The original validation/test labels are excluded from all search and selection.
Earlier analyses used every EEGIS example, so all subsequent results remain
exploratory. This module never reports an untouched evaluation claim.
"""
import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numba
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .accuracy_features import BinaryAccuracyModel, BinaryCSP, make_banks
from .data import audit_shared_samples, digest
from .epochs import load_recordings
from .improve import TangentFeatures
from .train_paper_binary import binary_data


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=lambda a: a.item() if isinstance(a, np.generic) else str(a), allow_nan=False)+'\n')
    temporary.replace(path)


def score(y, p):
    return {'accuracy':float(accuracy_score(y, p)), 'balanced_accuracy':float(balanced_accuracy_score(y, p)),
            'macro_f1':float(f1_score(y, p, labels=[6, 8], average='macro', zero_division=0))}


def candidates():
    result = []
    for bank in ['spectral', 'car', 'shape', 'rocket', 'derivative', 'combined']:
        result.append((bank+'_ridge', bank, Pipeline([('scale', StandardScaler()),
                      ('model', RidgeClassifier(class_weight='balanced', solver='lsqr', tol=1e-4))]),
                      {'model__alpha':[1., 10., 100., 1000.]}))
        if bank in ['spectral', 'car', 'shape']:
            result.append((bank+'_svm', bank, Pipeline([('scale', StandardScaler()), ('model', SVC(class_weight='balanced'))]),
                          {'model__C':[.01, .1, 1., 10.]}))
    result.append(('spectral_lda', 'spectral', Pipeline([('scale', StandardScaler()),
                  ('model', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[.5, .5]))]), {}))
    for bank in ['covariance', 'car_covariance']:
        result.append((bank+'_tangent', bank, Pipeline([('spatial', TangentFeatures()), ('scale', StandardScaler()),
                      ('model', RidgeClassifier(class_weight='balanced', solver='lsqr', tol=1e-4))]),
                      {'spatial__normalize':[False, True], 'model__alpha':[1., 100., 1000.]}))
        result.append((bank+'_svm', bank, Pipeline([('spatial', TangentFeatures()), ('scale', StandardScaler()),
                      ('model', SVC(class_weight='balanced'))]),
                      {'spatial__normalize':[False, True], 'model__C':[.1, 1., 10.]}))
        result.append((bank+'_csp', bank, Pipeline([('spatial', BinaryCSP()), ('scale', StandardScaler()),
                      ('model', LogisticRegression(class_weight='balanced', max_iter=2000))]),
                      {'spatial__pairs':[1, 2, 3], 'model__C':[.01, .1, 1.]}))
    result.append(('extra_trees', 'spectral', ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=4,
                  class_weight='balanced'), {'min_samples_leaf':[2, 8], 'max_features':['sqrt', .5]}))
    result.append(('boosted', 'spectral', HistGradientBoostingClassifier(random_state=42, max_iter=150,
                  max_leaf_nodes=7, early_stopping=False, class_weight='balanced'), {'l2_regularization':[10., 100.]}))
    result.append(('majority', 'spectral', DummyClassifier(strategy='most_frequent'), {}))
    return result


def folds(manifest, indices, n, seed):
    sub = manifest.iloc[indices]
    groups = sub.groupby('overlap_group', sort=True).label.first()
    result = []
    for fit, val in StratifiedKFold(n, shuffle=True, random_state=seed).split(groups.index, groups.values):
        a = np.flatnonzero(sub.overlap_group.isin(groups.index[fit]))
        b = np.flatnonzero(sub.overlap_group.isin(groups.index[val]))
        assert set(sub.overlap_group.iloc[a]).isdisjoint(set(sub.overlap_group.iloc[b]))
        result.append((a, b))
    return result


def choose_index(results):
    return max(range(len(results['params'])), key=lambda i:(results['mean_test_balanced_accuracy'][i],
                results['mean_test_macro_f1'][i], -i))


def search(banks, labels, rows, cv, output, progress):
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, bank, estimator, grid in candidates():
        filename = output/(name+'.json')
        if filename.exists():
            value = json.loads(filename.read_text())
        else:
            estimator = GridSearchCV(estimator, grid, scoring={'balanced_accuracy':'balanced_accuracy', 'macro_f1':'f1_macro'},
                                     refit=choose_index, cv=cv, n_jobs=1, error_score='raise')
            estimator.fit(banks[bank][rows], labels[rows])
            index = estimator.best_index_
            value = {'name':name, 'bank':bank, 'parameters':estimator.best_params_,
                     'inner_balanced_accuracy':float(estimator.cv_results_['mean_test_balanced_accuracy'][index]),
                     'inner_macro_f1':float(estimator.cv_results_['mean_test_macro_f1'][index])}
            pd.DataFrame(estimator.cv_results_).to_csv(output/(name+'_cv.csv'), index=False)
            write(filename, value)
        summaries.append(value)
        print(progress, name, round(value['inner_balanced_accuracy'], 4), flush=True)
    # Stable declared family order breaks remaining ties.
    winner = max(summaries, key=lambda r:(r['inner_balanced_accuracy'], r['inner_macro_f1']))
    write(output/'selection.json', {'winner':winner, 'families':summaries})
    lookup = {name:(bank, model) for name, bank, model, _ in candidates()}
    bank, estimator = lookup[winner['name']]
    estimator = clone(estimator).set_params(**winner['parameters']).fit(banks[bank][rows], labels[rows])
    return winner, estimator


def fingerprint():
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['accuracy_features.py', 'train_accuracy.py', 'data.py', 'epochs.py', 'improve.py', 'train_paper_binary.py', 'paper_ga.py', 'paper_features.py']}


def run(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    libs = {name:version(name) for name in ['numpy', 'scipy', 'scikit-learn', 'numba', 'pandas', 'joblib']}
    print('Loading original binary EEG...', flush=True)
    windows, wm = binary_data(args.data, args.manifest)
    plan = {'seed':42, 'outer_folds':5, 'inner_folds':3, 'kernels_per_representation':args.kernels,
            'units':['window', 'recording'], 'classes':[6, 8], 'target_balanced_accuracy':.70,
            'selection':'inner training-group CV balanced accuracy; macro-F1 and declared order break ties',
            'outer_protocol':'nested grouped CV restricted to original train partition',
            'final_fit':'original train only; do not fit original validation/test',
            'original_validation_test_excluded_from_search':True, 'all_results_exploratory':True,
            'previous_research_exposed_all_EEGIS_examples':True, 'subject_independent':False,
            'holdout_evaluated_by_this_script':False, 'python':platform.python_version(), 'libraries':libs,
            'window_data_sha256':digest(windows),
            'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(), 'threads':4,
            'source_sha256':fingerprint(),
            'candidates':[{'name':n, 'bank':b, 'parameters':m.get_params(deep=False).__repr__(), 'grid':g,
                           'configurations':len(list(ParameterGrid(g)))} for n,b,m,g in candidates()],
            'method_reference':'https://arxiv.org/abs/1910.13051',
            'random_features':'independent multivariate ROCKET-inspired implementation; weights and biases fixed without training examples',
            'longer_context':'1008 reconstructed raw samples, 7.875s; not comparable to 1-second decisions'}
    # sklearn estimator reprs are stable; no memory-address objects included.
    path = out/'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Configuration/source/runtime differs from frozen search; use a new output')
    write(path, plan)
    source = wm.rename(columns={'original_split':'split'})
    audit = audit_shared_samples(windows, source)
    recordings, rm = load_recordings(args.data, source)
    rm['label'] = rm.label.astype(int)
    wm = wm.rename(columns={'original_split':'split'})
    wm.to_csv(out/'window_manifest.csv', index=False); rm.to_csv(out/'recording_manifest.csv', index=False)
    write(out/'raw_audit.json', {'window_sha256':digest(windows), 'recording_sha256':digest(recordings),
                               'exact_overlap':audit, 'subjects_available':False})
    for unit, x, m in [('recording', recordings, rm), ('window', windows, wm)]:
        directory = out/unit; directory.mkdir(exist_ok=True)
        cache = directory/'features.joblib'
        if cache.exists():
            saved = joblib.load(cache)
            if saved['source_sha'] != digest(x) or saved['code'] != fingerprint():
                raise ValueError('Feature cache source differs')
            banks, kernels = saved['banks'], saved['kernels']
        else:
            print('Computing', unit, 'features', flush=True)
            banks, kernels = make_banks(x, args.kernels)
            for name, a in banks.items():
                if not np.isfinite(a).all():
                    raise ValueError('Nonfinite features '+name)
            joblib.dump({'source_sha':digest(x), 'code':fingerprint(), 'banks':banks, 'kernels':kernels}, cache, compress=1)
        training = np.flatnonzero(m.split == 'train')
        y = m.label.to_numpy()
        outer = folds(m, training, 5, 42)
        predictions = m.iloc[training][['overlap_group', 'label']].copy()
        predictions['outer_fold'] = -1; predictions['prediction'] = -1; predictions['majority'] = -1
        fold_summary = []
        membership = []
        for fold, (fit, val) in enumerate(outer):
            fit_rows, val_rows = training[fit], training[val]
            inner = folds(m, fit_rows, 3, 100+fold)
            fold_dir = directory/('outer_'+str(fold))
            member = {'outer_fold':fold, 'fit_groups':sorted(m.overlap_group.iloc[fit_rows].unique().tolist()),
                      'validation_groups':sorted(m.overlap_group.iloc[val_rows].unique().tolist()),
                      'inner':[{'fit_groups':sorted(m.overlap_group.iloc[fit_rows[a]].unique().tolist()),
                                'validation_groups':sorted(m.overlap_group.iloc[fit_rows[b]].unique().tolist())} for a,b in inner]}
            membership.append(member)
            write(directory/'cv_membership.json', membership)
            selected, estimator = search(banks, y, fit_rows, inner, fold_dir, unit+' fold'+str(fold))
            predicted = estimator.predict(banks[selected['bank']][val_rows])
            majority = int(pd.Series(y[fit_rows]).mode().iloc[0])
            predictions.iloc[val, predictions.columns.get_loc('outer_fold')] = fold
            predictions.iloc[val, predictions.columns.get_loc('prediction')] = predicted
            predictions.iloc[val, predictions.columns.get_loc('majority')] = majority
            fold_summary.append({'fold':fold, 'winner':selected, **score(y[val_rows], predicted)})
            print('NESTED', unit, fold, score(y[val_rows], predicted), flush=True)
        assert (predictions.prediction != -1).all()
        predictions.to_csv(directory/'nested_predictions.csv', index=False)
        results = {'unit':unit, 'duration_seconds':x.shape[1]/128, 'windows':len(training),
                   'independent_overlap_groups':m.overlap_group.iloc[training].nunique(),
                   'nested':score(predictions.label, predictions.prediction),
                   'majority':score(predictions.label, predictions.majority), 'folds':fold_summary,
                   'exploratory':True, 'subject_independent':False, 'original_validation_test_evaluated':False}
        write(directory/'nested_results.json', results)
        cv = folds(m, training, 3, 142)
        winner, estimator = search(banks, y, training, cv, directory/'final_search', unit+' final')
        model = BinaryAccuracyModel(estimator, winner['bank'], kernels, x.shape[1])
        joblib.dump(model, directory/'best_model.joblib')
        # Raw inference must match the feature-bank path before exporting.
        np.testing.assert_array_equal(model.predict(x[training[:2]]), estimator.predict(banks[winner['bank']][training[:2]]))
        write(directory/'best_model.json', {'winner':winner, 'training_groups':sorted(m.overlap_group.iloc[training].unique().tolist()),
                                          'samples':x.shape[1], 'class_ids':[6,8], 'raw_inference_checked':True})
        print('COMPLETE', unit, results['nested'], flush=True)
    write(out/'search_completed.json', {'numerical_source_sha256':fingerprint(), 'original_holdouts_not_evaluated':True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='EEGIS - Electroencephalogram Imagined Speech dataset')
    parser.add_argument('--manifest', default='artifacts/manifest.csv')
    parser.add_argument('--out', default='artifacts_accuracy')
    parser.add_argument('--kernels', type=int, default=2048)
    args = parser.parse_args()
    if args.kernels < 1:
        parser.error('Require at least one random kernel')
    numba.set_num_threads(4)
    with threadpool_limits(limits=4):
        run(args)


if __name__ == '__main__':
    main()
