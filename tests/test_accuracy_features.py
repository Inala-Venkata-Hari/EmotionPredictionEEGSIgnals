import unittest

import numpy as np
from scipy import signal

from eegis.accuracy_features import BinaryCSP, RandomTemporalFeatures, apply_kernels, covariance_features, spectral_features
from eegis.train_accuracy import folds


class AccuracyFeaturesTests(unittest.TestCase):
    def test_random_convolutions_match_independent_scipy_reference(self):
        rng = np.random.default_rng(27)
        x = rng.normal(size=(3, 47, 14)).astype(np.float32)
        bank = RandomTemporalFeatures(kernels=6).fit(x)
        actual = apply_kernels(x, bank.weights_, bank.channels_, bank.counts_, bank.lengths_,
                               bank.dilations_, bank.paddings_, bank.biases_)
        expected = np.empty_like(actual)
        for row in range(len(x)):
            for k in range(bank.kernels):
                length, dilation, padding = bank.lengths_[k], bank.dilations_[k], bank.paddings_[k]
                # Correlation with explicit zero-expanded dilated kernels.
                values = None
                for a in range(bank.counts_[k]):
                    kernel = np.zeros((length-1)*dilation+1)
                    kernel[::dilation] = bank.weights_[k, a, :length]
                    channel = np.pad(x[row, :, bank.channels_[k, a]], (padding, padding))
                    response = signal.correlate(channel, kernel, mode='valid', method='direct')
                    values = response if values is None else values+response
                values += bank.biases_[k]
                expected[row, 2*k] = np.mean(values > 0)
                expected[row, 2*k+1] = values.max()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)

    def test_features_are_independent_of_other_examples(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(8, 128, 14))
        for fn in [lambda a: spectral_features(a)[0], covariance_features]:
            np.testing.assert_allclose(fn(x)[[0, 5]], fn(x[[0, 5]]), rtol=1e-10, atol=1e-10)
        bank = RandomTemporalFeatures(kernels=8).fit(x)
        np.testing.assert_allclose(bank.transform(x)[[0, 5]], bank.transform(x[[0, 5]]), rtol=1e-5, atol=1e-6)

    def test_supervised_spatial_filter_recovers_known_class_covariance(self):
        cov = np.repeat(np.eye(14)[None, None], 80, axis=0)
        labels = np.repeat([6, 8], 40)
        cov[labels == 6, 0, 0, 0] = 10
        cov[labels == 8, 0, 1, 1] = 10
        fit = np.r_[0:30, 40:70]
        heldout = np.r_[30:40, 70:80]
        transform = BinaryCSP(pairs=1).fit(cov[fit], labels[fit])
        features = transform.transform(cov[heldout])
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(abs(features[:10, 0].mean()-features[10:, 0].mean()), 1.)

    def test_grouped_folds_never_split_correlated_windows(self):
        import pandas as pd
        frame = pd.DataFrame({'label':np.repeat([6, 8]*10, 12), 'overlap_group':np.repeat(np.arange(20), 12)})
        for fit, val in folds(frame, np.arange(len(frame)), 5, 42):
            self.assertFalse(set(frame.overlap_group.iloc[fit]) & set(frame.overlap_group.iloc[val]))
            self.assertEqual(set(frame.label.iloc[fit]), {6, 8})
            self.assertEqual(set(frame.label.iloc[val]), {6, 8})


if __name__ == '__main__':
    unittest.main()
