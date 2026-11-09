"""Audit saved kNN runs and six predeclared published-mask scaling diagnostics.

This does not search seeds, train a GA, select a model, or overwrite prior runs.
Run in .venv-paper to retain the recorded numeric environment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .author_features import published_genome
from .author_reproduction import fixed_cases
from .author_window_scaling import WithinWindowScaler
from .paper_ga import CLASS_IDS, K_VALUES, genome_columns
from .train_paper_binary import array_sha, dump


def scores(labels, predicted):
    return {'accuracy': float(accuracy_score(labels, predicted)),
            'macro_f1': float(f1_score(labels, predicted, labels=CLASS_IDS, average='macro')),
            'correct': int(np.sum(labels == predicted)), 'total': len(labels),
            'confusion': confusion_matrix(labels, predicted, labels=CLASS_IDS).tolist()}


def run(root, out):
    out.mkdir(parents=True, exist_ok=True)
    cases = ['grouped', 'window_80_20', 'window_20_80']
    dump(out/'plan.json', {
        'purpose': 'Audit the paper claim; resolve an untested normalization/mask combination',
        'fixed_before_new_predictions': True, 'seed': 42, 'threadpool_limit': 4,
        'new_diagnostics': {'mask': 'published 228-bit genome', 'normalization': 'within-window',
                            'cases': cases, 'k_values': [7, 9]},
        'all_cases_reported': True, 'select_winner': False, 'ga_rerun': False,
        'diagnostic_only': True, 'previous_test_exposure': True,
        'reason': 'Published mask was selected on this dataset; window splits also share EEG samples',
    })
    audit = json.loads((root/'feature_audit.json').read_text())
    code = Path(__file__).with_name('author_features.py')
    assert hashlib.sha256(code.read_bytes()).hexdigest() == audit['feature_code_sha256']
    with np.load(root/'feature_cache.npz', allow_pickle=False) as cache:
        features = cache['features']
    assert array_sha(features) == audit['feature_array_sha256']
    source = pd.read_csv(root/'source_manifest.csv')
    labels = source.label.to_numpy()
    assert features.shape == (924, 14, 214)
    assert source.label.value_counts().to_dict() == {6: 480, 8: 444}
    assert source.overlap_group.nunique() == 77

    existing, fitness = [], []
    for subdir in ['', 'within_window_grouped', 'paper_window_diagnostic']:
        directory = root/subdir
        manifest = pd.read_csv(directory/'manifest.csv')
        pd.testing.assert_series_equal(manifest.path, source.path)
        np.testing.assert_array_equal(manifest.label, labels)
        fit = np.flatnonzero(manifest.search_role == 'fit')
        val = np.flatnonzero(manifest.search_role == 'fitness_validation')
        selected = json.loads((directory/'knn_selection.json').read_text())
        flat = features.reshape(len(features), -1)[:, genome_columns(selected['genome'])]
        if subdir == 'within_window_grouped':
            scaler = WithinWindowScaler(sum(selected['genome'][:14]), sum(selected['genome'][14:]))
        else:
            scaler = StandardScaler()
        scaler.fit(flat[fit])
        a, b = scaler.transform(flat[fit]), scaler.transform(flat[val])
        for row in selected['all_k_scores']:
            assert row['k'] in K_VALUES
            predicted = KNeighborsClassifier(n_neighbors=row['k']).fit(a, labels[fit]).predict(b)
            actual = scores(labels[val], predicted)
            assert actual['accuracy'] == row['accuracy']
            np.testing.assert_allclose(actual['macro_f1'], row['macro_f1'], rtol=0, atol=1e-14)
            fitness.append({'run': subdir or 'grouped', 'k': row['k'], **actual})
        saved = pd.read_csv(directory/'test_predictions.csv')
        train = np.flatnonzero(manifest.outer_split == 'train')
        test = np.flatnonzero(manifest.outer_split == 'test')
        np.testing.assert_array_equal(saved.path, manifest.path.iloc[test])
        scaler.fit(flat[train])
        reference = KNeighborsClassifier(n_neighbors=selected['k'])
        reference.fit(scaler.transform(flat[train]), labels[train])
        np.testing.assert_array_equal(reference.predict(scaler.transform(flat[test])), saved.knn)
        actual = scores(saved.label.to_numpy(), saved.knn.to_numpy())
        result = json.loads((directory/'results.json').read_text())
        for key in ['accuracy', 'macro_f1']:
            assert actual[key] == result['models']['knn']['test'][key]
        existing.append({'run': subdir or 'grouped', 'selected_k': selected['k'],
                         'fitness_accuracy': selected['accuracy'], **actual})
    dump(out/'existing_results_audit.json', {'test_runs': existing,
        'selected_genome_fitness_all_k_actual_sklearn': fitness,
        'scope': 'All seven k values for each selected genome, not every intermediate GA genome'})

    genome = published_genome()
    selected = features.reshape(len(features), -1)[:, genome_columns(genome)]
    matrices = selected.reshape(len(features), 8, 103)
    independent = np.stack([StandardScaler().fit_transform(row).ravel() for row in matrices])
    scaler = WithinWindowScaler(8, 103).fit(selected)
    np.testing.assert_array_equal(scaler.transform(selected), independent)
    rows, predictions = [], []
    for case, train, test, _ in fixed_cases(source, 42):
        if case not in cases:
            continue
        shared = set(source.overlap_group.iloc[train]) & set(source.overlap_group.iloc[test])
        for k in [7, 9]:
            predicted = KNeighborsClassifier(n_neighbors=k).fit(independent[train], labels[train]).predict(independent[test])
            row = {'case': case, 'k': k, 'normalization': 'within-window',
                   'train_windows': len(train), 'test_windows': len(test),
                   'diagnostic_only': True, 'shared_overlap_groups': len(shared),
                   **scores(labels[test], predicted)}
            rows.append(row)
            part = source.iloc[test][['path', 'label', 'overlap_group']].copy()
            part['case'], part['k'], part['prediction'] = case, k, predicted
            predictions.append(part)
            print(json.dumps(row), flush=True)
    pd.DataFrame(rows).to_csv(out/'published_mask_within_window_results.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out/'published_mask_within_window_predictions.csv', index=False)
    dump(out/'verification.json', {'cached_features_hash_verified': True,
        'binary_counts_and_group_count_verified': True,
        'existing_test_predictions_and_metrics_recomputed_with_sklearn': len(existing),
        'selected_genome_k_fitness_scores_recomputed_with_sklearn': len(fitness),
        'within_window_scaler_matches_independent_sklearn_bitwise': True,
        'new_fixed_diagnostic_cases': len(rows), 'winner_selected': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts_author_reproduction'))
    parser.add_argument('--out', type=Path, default=Path('artifacts_author_recheck'))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.artifacts, args.out)


if __name__ == '__main__':
    main()
