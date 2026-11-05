import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from eegis.alternatives import FilterBankCSP, HaarFeatures, ShallowPowerClassifier, haar_coefficients


class AlternativeTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(12).normal(size=(18, 128, 14))
        self.y = np.tile(np.arange(9), 2)

    def test_haar_energy_preserved(self):
        parts = haar_coefficients(self.x)
        recovered_energy = sum((part**2).sum(axis=1) for part in parts)
        np.testing.assert_allclose(recovered_energy, (self.x**2).sum(axis=1), rtol=1e-12)

    def test_wavelet_features_are_finite_and_example_local(self):
        for reference in ['none', 'car']:
            model = HaarFeatures(reference=reference).fit(self.x[:9])
            features = model.transform(self.x[9:])
            self.assertEqual(features.shape, (9, 294))
            self.assertTrue(np.isfinite(features).all())
            np.testing.assert_allclose(features[:1], model.transform(self.x[9:10]))

    def test_csp_transform_does_not_fit_on_heldout_data(self):
        csp = FilterBankCSP(pairs=2).fit(self.x[:9], self.y[:9])
        fitted = csp.filters_.copy()
        features = csp.transform(self.x[9:])
        self.assertEqual(features.shape, (9, 180))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(csp.filters_, fitted)
        np.testing.assert_allclose(features[:1], csp.transform(self.x[9:10]), rtol=1e-10)

    def test_power_cnn_checkpoint_is_reloadable(self):
        model = ShallowPowerClassifier(epochs=1).fit(self.x, self.y)
        probability = model.predict_proba(self.x[:3])
        self.assertEqual(probability.shape, (3, 9))
        np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.joblib'
            joblib.dump(model, path)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(model.predict(self.x[:3]), loaded.predict(self.x[:3]))
            np.testing.assert_allclose(probability, loaded.predict_proba(self.x[:3]))

    def test_power_cnn_supports_sklearn_cross_validation(self):
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        scores = cross_val_score(ShallowPowerClassifier(epochs=1), self.x, self.y,
                                 cv=StratifiedKFold(2), scoring='accuracy')
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == '__main__':
    unittest.main()
