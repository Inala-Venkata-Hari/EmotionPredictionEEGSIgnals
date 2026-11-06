import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier

from eegis.paper_features import COLUMNS, FEATURE_NAMES, _line_histogram, paper_features, series_features
from eegis.paper_ga import (CLASSIFIERS, TorchKNN, fit_final, genetic_search,
                            genome_columns, make_estimator, resolve_device)
from eegis.train_paper_binary import split_binary


class PaperBinaryTests(unittest.TestCase):
    def test_feature_layout_and_scalar_equations(self):
        t = np.arange(128)/128
        x = np.stack([3+np.sin(2*np.pi*8*t), np.ones(128)])
        f = series_features(x)
        self.assertEqual(f.shape, (2, 36))
        np.testing.assert_allclose(f[:, 0], x.mean(axis=1))
        np.testing.assert_allclose(f[:, 1], x.std(axis=1))
        np.testing.assert_allclose(f[:, 13], (x*x).mean(axis=1))
        self.assertEqual(len(FEATURE_NAMES), 36)
        self.assertEqual(len(COLUMNS), 214)
        self.assertEqual(COLUMNS[33], 'raw/katz_fractal_dimension')
        self.assertEqual(COLUMNS[34], 'delta/mean')
        self.assertTrue(np.isfinite(f).all())

    def test_line_histograms_do_not_join_distinct_lines_or_examples(self):
        a = np.zeros((2, 5, 5), dtype=bool)
        a[0, 0, 1] = a[0, 1, 2] = a[0, 2, 3] = True
        a[1, 0, 0] = a[1, 1, 0] = True
        diagonal = _line_histogram(a, True)
        vertical = _line_histogram(a, False)
        self.assertEqual(diagonal[0, 3], 1)
        self.assertEqual(diagonal[1, 1], 2)
        self.assertEqual(vertical[1, 2], 1)
        self.assertEqual(vertical[0, 1], 3)

    def test_feature_extraction_is_example_independent_and_batch_invariant(self):
        x = np.random.default_rng(5).normal(size=(2, 128, 14))
        f = paper_features(x, batch_size=14)
        self.assertEqual(f.shape, (2, 14, 214))
        np.testing.assert_allclose(f[:1], paper_features(x[:1], batch_size=7), rtol=1e-9, atol=1e-9)

    def test_torch_knn_matches_sklearn_and_cpu_checkpoints(self):
        rng = np.random.default_rng(12)
        x, q = rng.normal(size=(70, 23)), rng.normal(size=(13, 23))
        y = np.array([6, 8]*35)
        model = TorchKNN(n_neighbors=7, device='cpu').fit(x, y)
        reference = KNeighborsClassifier(n_neighbors=7, algorithm='brute').fit(x, y)
        np.testing.assert_array_equal(model.predict(q), reference.predict(q))
        np.testing.assert_allclose(model.predict_proba(q), reference.predict_proba(q))
        self.assertIsInstance(model.x_, np.ndarray)
        tied = TorchKNN(1, device='cpu').fit(np.array([[-1.], [1.]]), np.array([8, 6]))
        np.testing.assert_array_equal(tied.predict([[0.]]), [8])

    def test_device_request_fails_clearly_when_cuda_unavailable(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(resolve_device('auto'), 'cpu')
            with self.assertRaises(RuntimeError):
                resolve_device('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA build/driver unavailable on this machine')
    def test_actual_cuda_prediction_matches_cpu(self):
        rng = np.random.default_rng(9)
        x, q = rng.normal(size=(50, 11)), rng.normal(size=(8, 11))
        y = np.array([6, 8]*25)
        a = TorchKNN(7, 'cpu').fit(x, y)
        b = TorchKNN(7, 'cuda').fit(x, y)
        np.testing.assert_array_equal(a.predict(q), b.predict(q))

    def test_mask_and_train_only_scaler(self):
        genome = np.zeros(228, dtype=np.uint8)
        genome[[0, 13, 14, 227]] = 1
        np.testing.assert_array_equal(genome_columns(genome), [0, 213, 2782, 2995])
        with self.assertRaises(ValueError):
            genome_columns(np.zeros(228))
        rng = np.random.default_rng(11)
        x, y = rng.normal(size=(80, 14, 214)), np.array([6, 8]*40)
        train = np.arange(60)
        a = fit_final(x, y, train, genome, 'knn', device='cpu')
        x[60:] += 10000
        b = fit_final(x, y, train, genome, 'knn', device='cpu')
        np.testing.assert_array_equal(a.scaler.mean_, b.scaler.mean_)
        np.testing.assert_array_equal(a.estimator.x_, b.estimator.x_)

    def test_grouped_partition_and_diagnostic_are_distinct(self):
        rows = []
        for label in [6, 8]:
            for group in range(10):
                for window in range(12):
                    rows.append({'label': label, 'overlap_group': label*100+group,
                                 'original_split': 'test' if group >= 8 else 'train'})
        source = pd.DataFrame(rows)
        m, outer, test, fit, val = split_binary(source, 'grouped')
        groups = m.overlap_group.to_numpy()
        self.assertFalse(set(groups[outer]) & set(groups[test]))
        self.assertFalse(set(groups[fit]) & set(groups[val]))
        self.assertEqual(set(np.concatenate([fit, val])), set(outer))
        p, outer, test, _, _ = split_binary(source, 'paper-window')
        self.assertTrue(set(p.overlap_group.iloc[outer]) & set(p.overlap_group.iloc[test]))

    def test_ga_generations_are_reproducible_and_elitist(self):
        rng = np.random.default_rng(7)
        x, y = rng.normal(size=(80, 14, 214)), np.array([6, 8]*40)
        kwargs = dict(generations=2, population_size=4, device='cpu', seed=5)
        a, h, populations = genetic_search(x, y, np.arange(40), np.arange(40, 60), **kwargs)
        b, _, _ = genetic_search(x, y, np.arange(40), np.arange(40, 60), **kwargs)
        self.assertEqual(a['genome'], b['genome'])
        self.assertEqual(a['accuracy'], b['accuracy'])
        self.assertEqual([r['generation'] for r in h], [0, 1, 2])
        self.assertEqual(len(populations), 3)
        self.assertTrue(np.all(np.diff([r['best_validation_accuracy'] for r in h]) >= 0))
        with self.assertRaises(ValueError):
            genetic_search(x, y, np.arange(40), np.arange(39, 60), **kwargs)

    def test_all_ten_classifier_factories_use_existing_libraries(self):
        self.assertEqual(len(CLASSIFIERS), 10)
        rng = np.random.default_rng(27)
        x = rng.normal(size=(80, 6))
        y = np.where(x[:, 0] > 0, 8, 6)
        for name in CLASSIFIERS:
            estimator = make_estimator(name, device='cpu')
            self.assertTrue(callable(estimator.fit))
            self.assertTrue(callable(estimator.predict))
            estimator.fit(x, y)
            self.assertTrue(set(estimator.predict(x[:5])) <= {6, 8})


if __name__ == '__main__':
    unittest.main()
