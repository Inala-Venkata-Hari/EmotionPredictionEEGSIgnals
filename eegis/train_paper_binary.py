"""Run the paper's binary feature/GA/kNN experiment for 50 generations.

Default: grouped evaluation plus an explicitly invalid overlapping-window diagnostic.
Use --classifiers all to run a separate GA for all ten paper classifiers.
"""
import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .data import CHANNELS, audit_shared_samples, digest, overlap_groups
from .diagnose_leakage import leakage_counts
from .epochs import read_window
from .paper_features import COLUMNS, FEATURE_SPEC, paper_features
from .paper_ga import (CLASS_IDS, CLASS_NAMES, CLASSIFIERS, K_VALUES, fit_final,
                       genetic_search, genome_columns, resolve_device)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def binary_data(data, source_manifest):
    source = pd.read_csv(source_manifest)
    if source.path.duplicated().any():
        raise ValueError('Duplicate source paths')
    m = source.loc[source.label.isin(CLASS_IDS)].copy().reset_index(drop=True)
    if set(m.label) != set(CLASS_IDS) or m.groupby('overlap_group').label.nunique().max() != 1:
        raise ValueError('Expected unmixed No/Sí overlap groups')
    if m.groupby('overlap_group').split.nunique().max() != 1:
        raise ValueError('Original overlap groups cross partitions')
    x = np.stack([read_window(Path(data)/p) for p in m.path])
    reconstructed = overlap_groups(x)
    equivalence = pd.DataFrame({'saved': m.overlap_group, 'computed': reconstructed})
    if (equivalence.groupby('saved').computed.nunique().max() != 1
            or equivalence.groupby('computed').saved.nunique().max() != 1):
        raise ValueError('Saved overlap groups disagree with exact raw EEG matches')
    if not (m.groupby('overlap_group').size() == 12).all():
        raise ValueError('Expected complete 12-window overlap groups')
    m = m.rename(columns={'split': 'original_split'})
    m['class_name'] = m.label.map(dict(zip(CLASS_IDS, CLASS_NAMES)))
    return x, m


def features_cached(x, out, batch_size):
    source_sha = digest(x)
    spec_sha = hashlib.sha256(json.dumps(FEATURE_SPEC, sort_keys=True).encode()).hexdigest()
    code_sha = hashlib.sha256(Path(__file__).with_name('paper_features.py').read_bytes()).hexdigest()
    path = out/'feature_cache.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            if (str(cache['source_sha']) == source_sha and str(cache['spec_sha']) == spec_sha
                    and str(cache['code_sha']) == code_sha):
                features = cache['features']
                if features.shape != (len(x), 14, 214) or not np.isfinite(features).all():
                    raise ValueError('Invalid cached features')
                print('Reusing source/spec/code-checked feature bank', flush=True)
                return features, source_sha, spec_sha, code_sha
    features = paper_features(x, batch_size=batch_size, progress=lambda s: print(s, flush=True))
    np.savez_compressed(path, features=features, source_sha=source_sha,
                        spec_sha=spec_sha, code_sha=code_sha)
    return features, source_sha, spec_sha, code_sha


def split_binary(manifest, protocol, seed=42):
    m = manifest.copy()
    labels = m.label.to_numpy()
    all_rows = np.arange(len(m))
    if protocol == 'grouped':
        outer_train = np.flatnonzero(m.original_split != 'test')
        test = np.flatnonzero(m.original_split == 'test')
        rng = np.random.default_rng(seed)
        validation = []
        for label in CLASS_IDS:
            groups = m.loc[(m.label == label) & (m.original_split != 'test'), 'overlap_group'].unique().copy()
            rng.shuffle(groups)
            chosen = groups[:max(1, round(.2*len(groups)))]
            validation.extend(m.index[(m.label == label) & m.overlap_group.isin(chosen)].tolist())
        validation = np.sort(validation)
        fit = np.setdiff1d(outer_train, validation)
        groups = m.overlap_group.to_numpy()
        for a, b in [(outer_train, test), (fit, validation)]:
            if set(groups[a]) & set(groups[b]):
                raise ValueError('Overlapping EEG groups cross a grouped partition')
    elif protocol == 'paper-window':
        # The paper does not specify a grouping rule or exact split seed.
        # This is an explicit conventional window-split diagnostic, not a proven reproduction.
        outer_train, test = train_test_split(all_rows, test_size=.2, random_state=seed, stratify=labels)
        fit, validation = train_test_split(outer_train, test_size=.2, random_state=seed+1,
                                         stratify=labels[outer_train])
        outer_train, test, fit, validation = map(np.sort, [outer_train, test, fit, validation])
    else:
        raise ValueError('Unknown protocol')
    m['outer_split'] = 'train'
    m.loc[test, 'outer_split'] = 'test'
    m['search_role'] = 'fit'
    m.loc[validation, 'search_role'] = 'fitness_validation'
    m.loc[test, 'search_role'] = 'test'
    for rows in [outer_train, test, fit, validation]:
        if set(labels[rows]) != set(CLASS_IDS):
            raise ValueError('A binary partition lacks one class')
    return m, outer_train, test, fit, validation


def metrics(y, p):
    return {'accuracy': float(accuracy_score(y, p)),
            'macro_f1': float(f1_score(y, p, labels=CLASS_IDS, average='macro', zero_division=0))}


def plot_run(out, results, histories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    title = 'Window split: overlap diagnostic only' if results['diagnostic_only'] else 'Grouped binary evaluation'
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, history in histories.items():
        h = pd.DataFrame(history)
        ax.plot(h.generation, h.best_validation_accuracy, label=f'{name}: best fitness')
        ax.plot(h.generation, h.mean_survivor_accuracy, alpha=.5, linestyle='--', label=f'{name}: survivor mean')
    ax.set(xlabel='GA generation (0 = initialization)', ylabel='Search-validation accuracy', ylim=(0, 1), title=title)
    ax.legend(); fig.tight_layout(); fig.savefig(out/'training_graph.png', dpi=140); plt.close(fig)
    for name in results['models']:
        cm = np.loadtxt(out/f'{name}_confusion.csv', delimiter=',', dtype=int)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
        ax.set_title(f'{name}: {title}')
        fig.tight_layout(); fig.savefig(out/f'{name}_confusion.png', dpi=140); plt.close(fig)
    names = list(results['models'])+['majority']
    values = [results['models'][n]['test']['accuracy'] for n in results['models']]+[results['majority']['accuracy']]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, values); ax.set(ylabel='Diagnostic score' if results['diagnostic_only'] else 'Test accuracy', ylim=(0, 1), title=title)
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=140); plt.close(fig)


def run_protocol(args, x, source, features, protocol, out, environment):
    out.mkdir(parents=True, exist_ok=True)
    (out/'verification.json').unlink(missing_ok=True)
    started = time.time()
    m, outer_train, test, fit, validation = split_binary(source, protocol, args.seed)
    m.to_csv(out/'manifest.csv', index=False)
    y, groups = m.label.to_numpy(), m.overlap_group.to_numpy()
    outer_overlap = leakage_counts(x, groups, outer_train, test)
    inner_overlap = leakage_counts(x, groups, fit, validation)
    if protocol == 'grouped':
        audit_shared_samples(x, m.rename(columns={'outer_split': 'split'}))
        inner_m = m.iloc[outer_train].copy()
        inner_m['split'] = inner_m.search_role
        audit_shared_samples(x[outer_train], inner_m)
    diagnostic = protocol == 'paper-window'
    plan = {
        'paper_doi': FEATURE_SPEC['paper_doi'], 'protocol': protocol,
        'diagnostic_only': diagnostic, 'classes': dict(zip(CLASS_IDS, CLASS_NAMES)),
        'classifiers': args.classifiers, 'generations': args.generations,
        'population_size': args.population, 'seed': args.seed,
        'device_requested': args.device, 'knn_device': environment['resolved_device'],
        'ga_not_neural_training': True, 'knn_candidates': K_VALUES,
        'selection': 'highest search-validation accuracy; stable classifier/genome/k order resolves ties',
        'fitness_policy': 'one fixed inner validation partition within outer training; no test labels enter search',
        'final_fit': 'refit scaler and selected estimator on all outer training windows',
        'split_sizes': {'outer_train': len(outer_train), 'test': len(test),
                        'fitness_fit': len(fit), 'fitness_validation': len(validation)},
        'groups': {'outer_train': sorted(map(int, set(groups[outer_train]))), 'test': sorted(map(int, set(groups[test]))),
                   'fitness_fit': sorted(map(int, set(groups[fit]))), 'fitness_validation': sorted(map(int, set(groups[validation])))},
        'overlap_audit': {'outer': outer_overlap, 'inner': inner_overlap},
        'departures_and_unknowns': [
            '50 GA generations requested instead of the paper\'s 100; initialization is generation 0.',
            'Default smoke run uses kNN, the paper\'s best classifier; --classifiers all enables ten separate searches.',
            'Paper lists k=7 then k=9; this code uses the explicitly listed seven k candidates and searches k.',
            'Paper states 80/20 then 20/80; this code uses approximately 80% outer training and 20% testing.',
            'Paper does not specify the fitness-validation protocol; this code reserves 20% of outer training for fitness.',
            'Group-safe mode retains the old test groups and combines old train+validation for outer training.',
            'Window mode assumes stratified row splits; paper does not establish this exact split. It knowingly leaks EEG overlap.',
            'Uniform bit crossover probability 0.5 and per-bit mutation 1/228 are specified implementation choices.',
            'Raw feature omissions, nonlinear parameters, filter direction and ambiguous formulas are specified in feature_spec.json.',
            'CUDA runs kNN distance and neighbor sorting; voting, SciPy features and other sklearn classifiers remain CPU.',
        ],
        'subject_independent': False, 'test_previously_reported': protocol == 'grouped',
        'performance_interpretation': 'overlap-leaking diagnostic; not valid decoder accuracy' if diagnostic else 'exploratory grouped binary window classification; participants unidentified',
    }
    dump(out/'plan.json', plan)
    print(f'Protocol {protocol}: {plan["split_sizes"]}; overlap={plan["overlap_audit"]}', flush=True)
    search_results, histories = {}, {}
    fit_local = np.flatnonzero(np.isin(outer_train, fit))
    val_local = np.flatnonzero(np.isin(outer_train, validation))
    # Neither held-out test features nor test labels are passed into the optimizer.
    for name in args.classifiers:
        result, history, populations = genetic_search(
            features[outer_train], y[outer_train], fit_local, val_local,
            classifier=name, generations=args.generations, population_size=args.population,
            seed=args.seed, device=args.device, progress=lambda s: print(s, flush=True),
            history_path=out/f'{name}_history.csv')
        evaluations = result.pop('evaluations')
        pd.DataFrame([{'genome': ''.join(map(str, r['genome'])), 'accuracy': r['accuracy'],
                       'macro_f1': r['macro_f1'], 'k': r['k']}
                      for r in evaluations]).to_csv(out/f'{name}_search_evaluations.csv', index=False)
        np.savez_compressed(out/f'{name}_k_search.npz',
                            accuracies=np.array([[score['accuracy'] for score in r['all_k_scores']] for r in evaluations]),
                            macro_f1=np.array([[score['macro_f1'] for score in r['all_k_scores']] for r in evaluations]))
        # Large population state is a local checkpoint rather than a public report.
        np.savez_compressed(out/f'{name}_populations.npz',
                            genomes=np.array([r['genomes'] for r in populations], dtype=np.uint8),
                            accuracies=np.array([r['accuracies'] for r in populations]))
        result['selected_channels'] = [c for c, bit in zip(CHANNELS, result['genome'][:14]) if bit]
        result['selected_features'] = [c for c, bit in zip(COLUMNS, result['genome'][14:]) if bit]
        result['flattened_features'] = len(genome_columns(result['genome']))
        search_results[name], histories[name] = result, history
        dump(out/f'{name}_selection.json', result)
    winner = max(args.classifiers, key=lambda n: search_results[n]['accuracy'])
    selection = {'winner': winner, 'criterion': plan['selection'],
                 'fitness': {n: {'accuracy': r['accuracy'], 'k': r['k']} for n, r in search_results.items()},
                 'frozen_before_test_evaluation': True}
    dump(out/'selection.json', selection)
    predictions = m.loc[test, ['path', 'label', 'class_name', 'overlap_group']].copy()
    majority = int(pd.Series(y[outer_train]).mode().iloc[0])
    predictions['majority'] = majority
    results = {'protocol': protocol, 'diagnostic_only': diagnostic, 'selected_model': winner,
               'models': {}, 'majority': metrics(y[test], np.full(len(test), majority)),
               'majority_label': majority, 'split_sizes': plan['split_sizes'],
               'subject_independent': False, 'test_previously_reported': plan['test_previously_reported'],
               'environment': environment, 'limitations': plan['performance_interpretation']}
    for name in args.classifiers:
        selected = search_results[name]
        model = fit_final(features, y, outer_train, selected['genome'], name,
                          args.seed, args.device, k=selected['k'] or 7)
        predicted = model.predict_features(features[test])
        predictions[name] = predicted
        test_scores = metrics(y[test], predicted)
        # sklearn kNN is the CPU reference for the CUDA-capable implementation.
        parity = None
        if name == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            cols = genome_columns(selected['genome'])
            flat = features.reshape(len(features), -1)[:, cols]
            reference = KNeighborsClassifier(n_neighbors=selected['k'], algorithm='brute')
            reference.fit(model.scaler.transform(flat[outer_train]), y[outer_train])
            np.testing.assert_array_equal(reference.predict(model.scaler.transform(flat[test])), predicted)
            parity = True
        joblib.dump(model, out/f'{name}.joblib')
        if name == winner:
            joblib.dump(model, out/'best_model.joblib')
        cm = confusion_matrix(y[test], predicted, labels=CLASS_IDS)
        np.savetxt(out/f'{name}_confusion.csv', cm, delimiter=',', fmt='%d')
        results['models'][name] = {'test': test_scores, 'fitness_validation_accuracy': selected['accuracy'],
                                  'k': selected['k'], 'channels': selected['selected_channels'],
                                  'feature_types': len(selected['selected_features']),
                                  'flattened_features': selected['flattened_features'],
                                  'sklearn_knn_prediction_parity': parity,
                                  'report': classification_report(y[test], predicted, labels=CLASS_IDS,
                                                                 target_names=CLASS_NAMES, output_dict=True, zero_division=0)}
        print(name, 'measured test:', test_scores, 'majority:', results['majority'], flush=True)
    predictions.to_csv(out/'test_predictions.csv', index=False)
    dump(out/'best_model.json', {'model': f'PaperBinary_{winner}', 'file': 'best_model.joblib',
                               'input_transform': 'raw_window_pipeline', 'class_ids': CLASS_IDS,
                               'class_names': CLASS_NAMES, 'channels': CHANNELS,
                               'shape': [128, 14], 'diagnostic_only': diagnostic})
    results['elapsed_seconds'] = time.time()-started
    dump(out/'results.json', results)
    plot_run(out, results, histories)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='EEGIS - Electroencephalogram Imagined Speech dataset')
    parser.add_argument('--manifest', default='artifacts/manifest.csv')
    parser.add_argument('--out', default='artifacts_paper_binary')
    parser.add_argument('--protocols', nargs='+', choices=['grouped', 'paper-window'], default=['grouped', 'paper-window'])
    parser.add_argument('--classifiers', nargs='+', default=['knn'])
    parser.add_argument('--generations', '--epochs', dest='generations', type=int, default=50,
                        help='GA generations; --epochs is an alias, not neural training')
    parser.add_argument('--population', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--feature-batch-size', type=int, default=96)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.classifiers == ['all']:
        args.classifiers = CLASSIFIERS.copy()
    if not args.classifiers or any(n not in CLASSIFIERS for n in args.classifiers):
        parser.error('Classifiers must be all or names from: '+', '.join(CLASSIFIERS))
    if len(set(args.classifiers)) != len(args.classifiers):
        parser.error('Duplicate classifiers')
    if args.generations < 1 or args.population < 2 or args.population % 2 or args.feature_batch_size < 1:
        parser.error('Need positive generations/batch size and an even population >=2')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    environment = {'python': platform.python_version(),
                   'libraries': {name: version(name) for name in ['numpy', 'scipy', 'pandas', 'scikit-learn', 'torch', 'joblib']},
                   'torch_cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
                   'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   'resolved_device': device, 'cuda_executed': device == 'cuda',
                   'preprocessing_device': 'cpu', 'non_knn_classifier_device': 'cpu'}
    print('Environment:', environment, flush=True)
    dump(out/'environment.json', environment)
    dump(out/'feature_spec.json', FEATURE_SPEC)
    pd.DataFrame({'position': np.arange(214), 'feature': COLUMNS}).to_csv(out/'feature_columns.csv', index=False)
    x, source = binary_data(args.data, args.manifest)
    source.to_csv(out/'source_manifest.csv', index=False)
    with threadpool_limits(limits=4):
        features, source_sha, spec_sha, code_sha = features_cached(x, out, args.feature_batch_size)
        dump(out/'feature_audit.json', {'windows': len(x), 'shape': list(features.shape),
                                      'finite': bool(np.isfinite(features).all()),
                                      'source_array_sha256': source_sha, 'feature_array_sha256': array_sha(features),
                                      'feature_spec_sha256': spec_sha, 'feature_code_sha256': code_sha,
                                      'source_manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                                      'arguments': vars(args)})
        if args.prepare_only:
            print('Feature preparation complete', flush=True)
            return
        for protocol in args.protocols:
            target = out if protocol == 'grouped' else out/'paper_window_diagnostic'
            run_protocol(args, x, source, features, protocol, target, environment)


if __name__ == '__main__':
    main()
