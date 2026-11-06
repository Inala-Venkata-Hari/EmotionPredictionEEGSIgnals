"""Bounded attempt to reproduce the paper using author-source features and 100 GA generations."""
import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .author_features import (AuthorBinaryModel, author_features, load_layers,
                              published_genome, source_spec)
from .data import CHANNELS, digest
from .paper_features import COLUMNS
from .paper_ga import CLASS_IDS, CLASS_NAMES, K_VALUES, TorchKNN, genetic_search, genome_columns, resolve_device
from .train_paper_binary import array_sha, binary_data, dump, metrics, plot_run, split_binary


def fast_fitness(features, labels, train, validation, genome, classifier, seed, device):
    """Same uniform Euclidean kNN objective; vectorized metrics avoid 7 sklearn fits/calls."""
    if classifier != 'knn':
        raise ValueError('This paper-winner experiment is kNN only')
    selected = features.reshape(len(features), -1)[:, genome_columns(genome)]
    scaler = StandardScaler().fit(selected[train])
    model = TorchKNN(device=device).fit(scaler.transform(selected[train]), labels[train])
    predictions = np.stack(model.predict_many(scaler.transform(selected[validation]), K_VALUES))
    target = labels[validation][None]
    accuracy = (predictions == target).mean(axis=1)
    class_f1 = []
    for label in CLASS_IDS:
        tp = ((predictions == label) & (target == label)).sum(axis=1)
        fp = ((predictions == label) & (target != label)).sum(axis=1)
        fn = ((predictions != label) & (target == label)).sum(axis=1)
        class_f1.append(2*tp/np.maximum(2*tp+fp+fn, 1))
    f1 = np.mean(class_f1, axis=0)
    records = [{'k': k, 'accuracy': float(a), 'macro_f1': float(f)} for k, a, f in zip(K_VALUES, accuracy, f1)]
    best = int(np.argmax(accuracy))
    return {**records[best], 'all_k_scores': records}


def prepare(data, manifest, out):
    x, source = binary_data(data, manifest)
    source.to_csv(out/'source_manifest.csv', index=False)
    layers = load_layers(data, source)
    np.testing.assert_array_equal(layers[:, 0], x)
    spec = source_spec()
    source_sha = digest(layers)
    code_sha = hashlib.sha256(Path(__file__).with_name('author_features.py').read_bytes()).hexdigest()
    spec_sha = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    cache_path = out/'feature_cache.npz'
    features = None
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache['source_sha']) == source_sha and str(cache['code_sha']) == code_sha and str(cache['spec_sha']) == spec_sha:
                features = cache['features']
    started = time.time()
    if features is None:
        features = author_features(layers, progress=lambda s: print(s, flush=True))
        np.savez_compressed(cache_path, features=features, source_sha=source_sha, code_sha=code_sha, spec_sha=spec_sha)
    else:
        print('Reusing author-source feature bank with checked source/code/spec hashes', flush=True)
    if features.shape != (len(source), 14, 214) or not np.isfinite(features).all():
        raise ValueError('Invalid author feature bank')
    dump(out/'feature_spec.json', spec)
    dump(out/'feature_audit.json', {'shape': list(features.shape), 'source_layers_sha256': source_sha,
                                  'feature_code_sha256': code_sha, 'feature_spec_sha256': spec_sha,
                                  'feature_array_sha256': array_sha(features),
                                  'source_manifest_sha256': hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                                  'feature_preparation_seconds_this_invocation': time.time()-started,
                                  'data': data, 'manifest': manifest})
    return x, source, layers, features


def fit_model(features, labels, train, genome, k, device='cpu', scaler_rows=None, normalization='train'):
    cols = genome_columns(genome)
    selected = features.reshape(len(features), -1)[:, cols]
    if normalization == 'within-window':
        from .author_window_scaling import WithinWindowScaler
        scaler = WithinWindowScaler(int(np.sum(genome[:14])), int(np.sum(genome[14:]))).fit(selected[train])
    elif normalization == 'train':
        scaler = StandardScaler().fit(selected[train if scaler_rows is None else scaler_rows])
    else:
        raise ValueError('Unknown normalization axis')
    if normalization=='within-window':
        if device=='cuda':
            raise ValueError('Within-window source-default tie handling uses sklearn CPU')
        model=KNeighborsClassifier(n_neighbors=k).fit(scaler.transform(selected[train]),labels[train])
    else:
        model = TorchKNN(k, device=device).fit(scaler.transform(selected[train]), labels[train])
    return AuthorBinaryModel(np.asarray(genome), scaler, model)


def fixed_cases(source, seed):
    y = source.label.to_numpy()
    _, grouped, test, _, _ = split_binary(source, 'grouped', seed)
    train80, test20 = train_test_split(np.arange(len(source)), test_size=.2, random_state=seed, stratify=y)
    train20, test80 = train_test_split(np.arange(len(source)), test_size=.8, random_state=seed, stratify=y)
    return [('grouped', np.sort(grouped), np.sort(test), False),
            ('window_80_20', np.sort(train80), np.sort(test20), False),
            ('window_20_80', np.sort(train20), np.sort(test80), False),
            ('window_80_20_global_scaling', np.sort(train80), np.sort(test20), True)]


def fixed_published_mask(x, source, features, out, seed, device):
    """Evaluate the two conflicting published k values without choosing by test accuracy."""
    from .diagnose_leakage import leakage_counts
    genome, y, groups = published_genome(), source.label.to_numpy(), source.overlap_group.to_numpy()
    rows, predictions, audit = [], [], []
    dump(out/'published_genome.json', {'genome': genome.tolist(), 'source': 'Paper Section 3',
                                     'channels': [name for name, bit in zip(CHANNELS, genome[:14]) if bit],
                                     'features': [name for name, bit in zip(COLUMNS, genome[14:]) if bit],
                                     'feature_types': 103, 'flattened_values': 824,
                                     'selection_may_have_used_entire_public_dataset': True})
    for case, train, test, global_scaling in fixed_cases(source, seed):
        counts = leakage_counts(x, groups, train, test)
        audit.append({'case': case, 'fit_rows': train.tolist(), 'test_rows': test.tolist(),
                      'scaler_rows': list(range(len(source))) if global_scaling else train.tolist(),
                      'overlap': counts, 'diagnostic_only': True,
                      'reason': 'Published feature mask was selected on this dataset; additional overlap/global-scaling leakage when indicated'})
        for k in [7, 9]:
            model = fit_model(features, y, train, genome, k, device,
                              scaler_rows=np.arange(len(source)) if global_scaling else None)
            predicted = model.predict_features(features[test])
            row = {'case': case, 'k': k, 'train_windows': len(train), 'test_windows': len(test),
                   'global_scaling': global_scaling, 'diagnostic_only': True,
                   **metrics(y[test], predicted), **counts}
            rows.append(row)
            p = source.iloc[test][['path', 'label', 'overlap_group']].copy()
            p['case'], p['k'], p['prediction'] = case, k, predicted
            predictions.append(p)
            joblib.dump(model, out/f'published_{case}_k{k}.joblib')
            print('Published mask:', row, flush=True)
    pd.DataFrame(rows).to_csv(out/'published_mask_results.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out/'published_mask_predictions.csv', index=False)
    dump(out/'published_mask_audit.json', audit)


def run_ga(args, x, source, features, protocol, out):
    from .diagnose_leakage import leakage_counts
    out.mkdir(parents=True, exist_ok=True)
    (out/'verification.json').unlink(missing_ok=True)
    m, train, test, fit, val = split_binary(source, protocol, args.seed)
    m.to_csv(out/'manifest.csv', index=False)
    y, groups = m.label.to_numpy(), m.overlap_group.to_numpy()
    overlap = {'outer': leakage_counts(x, groups, train, test), 'inner': leakage_counts(x, groups, fit, val)}
    diagnostic = protocol != 'grouped'
    normalization = getattr(args, 'normalization', 'train')
    evaluator = fast_fitness
    if normalization == 'within-window':
        from .author_window_scaling import window_fitness
        evaluator = window_fitness
    if not diagnostic and any(value for counts in overlap.values() for value in counts.values()):
        raise AssertionError('Grouped run leaks samples')
    dump(out/'plan.json', {'protocol': protocol, 'diagnostic_only': diagnostic, 'seed': args.seed,
                          'normalization': normalization,
                          'knn_backend': 'sklearn CPU defaults, independent prediction per k' if normalization=='within-window' else 'Torch float64, sklearn final parity checked',
                          'threadpool_limit':4,
                          'generations': args.generations, 'population_size': args.population,
                          'knn_candidates': K_VALUES, 'mutation_rate': 1/228,
                          'crossover': 'uniform independent bit mask p=0.5', 'selection': 'rank-weighted mating, top-100 parent+offspring survival',
                          'fitness': 'fixed inner validation accuracy only; never outer test labels',
                          'split_sizes': {'outer_train': len(train), 'test': len(test), 'fitness_fit': len(fit), 'fitness_validation': len(val)},
                          'overlap_audit': overlap, 'subject_independent': False, 'test_previously_reported': True,
                          'source': 'author-source bank described in ../feature_spec.json or feature_spec.json',
                          'exact_original_ga_rates_and_split_seed_unknown': True})
    fit_local, val_local = np.flatnonzero(np.isin(train, fit)), np.flatnonzero(np.isin(train, val))
    started = time.time()
    selected, history, populations = genetic_search(features[train], y[train], fit_local, val_local,
        generations=args.generations, population_size=args.population, seed=args.seed, device=args.device,
        evaluator=evaluator, history_path=out/'knn_history.csv', progress=lambda s: print(protocol+' '+normalization+': '+s, flush=True))
    evaluations = selected.pop('evaluations')
    pd.DataFrame([{'genome': ''.join(map(str, row['genome'])), 'accuracy': row['accuracy'],
                   'macro_f1': row['macro_f1'], 'k': row['k']} for row in evaluations]).to_csv(out/'knn_search_evaluations.csv', index=False)
    np.savez_compressed(out/'knn_populations.npz', genomes=np.array([p['genomes'] for p in populations], dtype=np.uint8),
                        accuracies=np.array([p['accuracies'] for p in populations]))
    selected['selected_channels'] = [c for c, bit in zip(CHANNELS, selected['genome'][:14]) if bit]
    selected['selected_features'] = [c for c, bit in zip(COLUMNS, selected['genome'][14:]) if bit]
    selected['flattened_features'] = len(genome_columns(selected['genome']))
    # Selection is saved before the first outer test prediction.
    dump(out/'knn_selection.json', selected)
    dump(out/'selection.json', {'winner': 'knn', 'criterion': 'highest fixed inner fitness accuracy; earliest k/genome breaks ties',
                               'frozen_before_test_evaluation': True})
    model = fit_model(features, y, train, selected['genome'], selected['k'], args.device, normalization=normalization)
    predicted = model.predict_features(features[test])
    cols = genome_columns(selected['genome'])
    flat = features.reshape(len(features), -1)[:, cols]
    reference = KNeighborsClassifier(n_neighbors=selected['k'], algorithm='brute').fit(model.scaler.transform(flat[train]), y[train])
    np.testing.assert_array_equal(reference.predict(model.scaler.transform(flat[test])), predicted)
    joblib.dump(model, out/'best_model.joblib')
    joblib.dump(model, out/'knn.joblib')
    majority = int(pd.Series(y[train]).mode().iloc[0])
    predictions = m.iloc[test][['path', 'label', 'overlap_group']].copy()
    predictions['knn'], predictions['majority'] = predicted, majority
    predictions.to_csv(out/'test_predictions.csv', index=False)
    np.savetxt(out/'knn_confusion.csv', confusion_matrix(y[test], predicted, labels=CLASS_IDS), fmt='%d', delimiter=',')
    scores = metrics(y[test], predicted)
    results = {'protocol': protocol, 'diagnostic_only': diagnostic, 'selected_model': 'knn',
               'normalization': normalization,
               'models': {'knn': {'test': scores, 'fitness_validation_accuracy': selected['accuracy'],
                                   'k': selected['k'], 'channels': selected['selected_channels'],
                                   'feature_types': len(selected['selected_features']), 'flattened_features': selected['flattened_features'],
                                   'report': classification_report(y[test], predicted, labels=CLASS_IDS, target_names=CLASS_NAMES, output_dict=True, zero_division=0)}},
               'majority': metrics(y[test], np.full(len(test), majority)), 'elapsed_seconds': time.time()-started,
               'target_accuracy': .96, 'target_met_on_grouped_test': not diagnostic and scores['accuracy'] >= .96,
               'subject_independent': False, 'test_previously_reported': True}
    dump(out/'results.json', results)
    dump(out/'best_model.json', {'model': 'AuthorSourceGA_kNN', 'file': 'best_model.joblib',
                               'input': 'six supplied layers, shape (N,6,128,14); use AuthorBinaryModel.predict_layers',
                               'class_ids': CLASS_IDS, 'class_names': CLASS_NAMES, 'diagnostic_only': diagnostic})
    plot_run(out, results, {'knn': history})
    print(protocol, 'FINAL MEASURED RESULT', results['models']['knn'], 'majority', results['majority'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='EEGIS - Electroencephalogram Imagined Speech dataset')
    p.add_argument('--manifest', default='artifacts/manifest.csv')
    p.add_argument('--out', default='artifacts_author_reproduction')
    p.add_argument('--generations', type=int, default=100)
    p.add_argument('--population', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    if args.generations < 1 or args.population < 2 or args.population % 2:
        p.error('Positive generations and even population >=2 required')
    device = resolve_device(args.device)
    torch.set_num_threads(4)
    out = Path(args.out); out.mkdir(exist_ok=True, parents=True)
    dump(out/'experiment_plan.json', {
        'target': 'Attempt the paper\'s 96% by recovering source behavior, not tuning test labels',
        'generations': args.generations, 'population': args.population, 'seed': args.seed,
        'protocols': ['grouped', 'paper-window'], 'fitness_objective': 'fixed inner validation accuracy',
        'additional_predeclared_variant': {'name':'within_window_grouped',
            'reason':'Paper standardizes electrode-feature columns before flattening; axis is not specified.',
            'generations':args.generations,'population':args.population,'seed':args.seed,
            'outer_split':'unchanged original grouped test','fitness':'same inner group split; outer test excluded'},
        'fixed_mask_cases': ['grouped k7/k9', 'window80/20 k7/k9', 'window20/80 k7/k9', 'window80/20 global-scaler diagnostic k7/k9'],
        'selection_policy': 'Never choose seeds, masks, normalization, split ratios, k, or protocols using outer test scores',
        'fitness_not_test_accuracy': True, 'no_target_based_early_stopping': True,
        'no_author_contact_or_messages_sent': True,
        'libraries': {name: version(name) for name in ['signalytica','antropy','numpy','scipy','scikit-learn','torch','numba']},
        'device': device, 'cuda_executed': device == 'cuda',
    })
    with threadpool_limits(limits=4):
        x, source, layers, features = prepare(args.data, args.manifest, out)
        if args.prepare_only:
            return
        fixed_published_mask(x, source, features, out, args.seed, args.device)
        for protocol in ['grouped', 'paper-window']:
            run_ga(args, x, source, features, protocol,
                   out if protocol == 'grouped' else out/'paper_window_diagnostic')


if __name__ == '__main__':
    main()
