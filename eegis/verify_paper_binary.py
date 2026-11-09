"""Independently replay binary dataset, selected fitness, scaler, and test predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from .data import audit_shared_samples, digest
from .diagnose_leakage import leakage_counts
from .paper_features import FEATURE_SPEC, paper_features
from .paper_ga import CLASS_IDS, fitness_predictions, genome_columns
from .train_paper_binary import array_sha, binary_data, dump, metrics, split_binary


def verify_protocol(out, x, source, features):
    plan = json.loads((out/'plan.json').read_text())
    results = json.loads((out/'results.json').read_text())
    m, train, test, fit, val = split_binary(source, plan['protocol'], plan['seed'])
    pd.testing.assert_frame_equal(m, pd.read_csv(out/'manifest.csv'))
    groups, y = m.overlap_group.to_numpy(), m.label.to_numpy()
    for a, b, name in [(train, test, 'outer'), (fit, val, 'inner')]:
        assert leakage_counts(x, groups, a, b) == plan['overlap_audit'][name]
    if plan['protocol'] == 'grouped':
        audit_shared_samples(x, m.rename(columns={'outer_split': 'split'}))
        inner = m.iloc[train].copy(); inner['split'] = inner.search_role
        audit_shared_samples(x[train], inner)
        assert not plan['diagnostic_only']
    else:
        assert plan['diagnostic_only']
        assert plan['overlap_audit']['outer']['shared_overlap_groups'] > 0
    fit_local = np.flatnonzero(np.isin(train, fit))
    val_local = np.flatnonzero(np.isin(train, val))
    saved = pd.read_csv(out/'test_predictions.csv')
    pd.testing.assert_frame_equal(saved[['path', 'label', 'class_name', 'overlap_group']].reset_index(drop=True),
                                  m.loc[test, ['path', 'label', 'class_name', 'overlap_group']].reset_index(drop=True))
    majority = int(pd.Series(y[train]).mode().iloc[0])
    np.testing.assert_array_equal(saved.majority, np.full(len(test), majority))
    assert metrics(y[test], saved.majority) == results['majority']
    candidate_scores = {}
    verified = []
    for name in plan['classifiers']:
        selected = json.loads((out/f'{name}_selection.json').read_text())
        genome = np.array(selected['genome'], dtype=np.uint8)
        reproduced = fitness_predictions(features[train], y[train], fit_local, val_local,
                                          genome, name, plan['seed'], device='cpu')
        for key in ['accuracy', 'macro_f1', 'k', 'all_k_scores']:
            assert reproduced[key] == selected[key], (name, key)
        candidate_scores[name] = reproduced['accuracy']
        evaluations = pd.read_csv(out/f'{name}_search_evaluations.csv', dtype={'genome': str})
        assert len(evaluations) == selected['unique_evaluations']
        assert (evaluations.genome.str.len() == 228).all()
        assert evaluations.genome.nunique() == len(evaluations)
        assert np.isclose(evaluations.accuracy.max(), selected['accuracy'])
        lookup = dict(zip(evaluations.genome, evaluations.accuracy))
        history = pd.read_csv(out/f'{name}_history.csv')
        np.testing.assert_array_equal(history.generation, np.arange(plan['generations']+1))
        assert np.all(np.diff(history.best_validation_accuracy) >= -1e-12)
        assert np.all(np.diff(history.unique_evaluations) >= 0)
        with np.load(out/f'{name}_populations.npz', allow_pickle=False) as cache:
            population, accuracies = cache['genomes'], cache['accuracies']
            assert population.shape == (plan['generations']+1, plan['population_size'], 228)
            for generation, (genomes, scores) in enumerate(zip(population, accuracies)):
                expected = [lookup[''.join(map(str, g))] for g in genomes]
                np.testing.assert_allclose(scores, expected, atol=1e-12)
                assert np.all(np.diff(scores) <= 0)
                assert abs(scores[0]-history.best_validation_accuracy.iloc[generation]) < 1e-12
                assert abs(scores.mean()-history.mean_survivor_accuracy.iloc[generation]) < 1e-12
            np.testing.assert_array_equal(population[-1, 0], genome)
        model = joblib.load(out/f'{name}.joblib')
        columns = genome_columns(genome)
        raw_selected = features.reshape(len(features), -1)[:, columns]
        np.testing.assert_allclose(model.scaler.mean_, raw_selected[train].mean(axis=0))
        np.testing.assert_array_equal(model.genome, genome)
        if name == 'knn':
            model.estimator.device = 'cpu'
            np.testing.assert_allclose(model.estimator.x_, model.scaler.transform(raw_selected[train]))
            reference = KNeighborsClassifier(n_neighbors=selected['k'], algorithm='brute')
            reference.fit(model.scaler.transform(raw_selected[train]), y[train])
            np.testing.assert_array_equal(reference.predict(model.scaler.transform(raw_selected[test])), saved[name])
        predicted = model.predict_features(features[test])
        np.testing.assert_array_equal(predicted, saved[name])
        assert metrics(y[test], predicted) == results['models'][name]['test']
        np.testing.assert_array_equal(confusion_matrix(y[test], predicted, labels=CLASS_IDS),
                                       np.loadtxt(out/f'{name}_confusion.csv', delimiter=','))
        # Full raw-window inference must agree, not just cached-feature inference.
        np.testing.assert_array_equal(model.predict(x[test[:2]]), predicted[:2])
        verified.append(name)
    winner = max(plan['classifiers'], key=lambda n: candidate_scores[n])
    assert winner == results['selected_model'] == json.loads((out/'selection.json').read_text())['winner']
    best = joblib.load(out/'best_model.joblib')
    if hasattr(best.estimator, 'device'):
        best.estimator.device = 'cpu'
    np.testing.assert_array_equal(best.predict_features(features[test]), saved[winner])
    verification = {'protocol': plan['protocol'], 'diagnostic_only': plan['diagnostic_only'],
                    'source_split_reproduced': True, 'exact_sample_overlap_recounted': True,
                    'zero_cross_partition_overlap': plan['protocol'] == 'grouped',
                    'selected_inner_fitness_recomputed': True, 'all_generations_and_population_scores_checked': True,
                    'final_scaler_training_only': True, 'checkpoint_predictions_match': True,
                    'sklearn_knn_reference_matches': 'knn' in verified,
                    'raw_window_inference_matches': True, 'metrics_and_confusion_recomputed': True,
                    'selection_matches_fitness_only': True, 'classifiers_verified': verified,
                    'cuda_execution_verified': False,
                    'cuda_note': 'This independent replay uses CPU; hardware CUDA test is conditional in the test suite.'}
    dump(out/'verification.json', verification)
    print(json.dumps(verification, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', default='artifacts_paper_binary')
    args = parser.parse_args()
    out = Path(args.artifacts)
    audit = json.loads((out/'feature_audit.json').read_text())
    original = audit['arguments']
    assert hashlib.sha256(Path(original['manifest']).read_bytes()).hexdigest() == audit['source_manifest_sha256']
    x, source = binary_data(original['data'], original['manifest'])
    pd.testing.assert_frame_equal(source, pd.read_csv(out/'source_manifest.csv'))
    assert digest(x) == audit['source_array_sha256']
    assert hashlib.sha256(json.dumps(FEATURE_SPEC, sort_keys=True).encode()).hexdigest() == audit['feature_spec_sha256']
    assert hashlib.sha256(Path(__file__).with_name('paper_features.py').read_bytes()).hexdigest() == audit['feature_code_sha256']
    with np.load(out/'feature_cache.npz', allow_pickle=False) as cache:
        features = cache['features']
    assert array_sha(features) == audit['feature_array_sha256']
    with threadpool_limits(limits=4):
        subset = np.array([0, len(x)//2, len(x)-1])
        np.testing.assert_allclose(paper_features(x[subset]), features[subset], rtol=1e-10, atol=1e-10)
        targets = [p for p in [out, out/'paper_window_diagnostic'] if (p/'results.json').exists()]
        if not targets:
            raise ValueError('No completed experiment to verify')
        for target in targets:
            verify_protocol(target, x, source, features)


if __name__ == '__main__':
    main()
