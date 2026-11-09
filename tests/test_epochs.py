import unittest
import numpy as np
from eegis.epochs import reconstruct, RecordingFeatures


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)
        self.recording = self.rng.normal(size=(1008, 14))
        self.windows = np.stack([self.recording[i*80:i*80+128] for i in range(12)])

    def test_exact_reconstruction_without_filenames_or_labels(self):
        permutation = self.rng.permutation(12)
        recovered, order = reconstruct(self.windows[permutation])
        np.testing.assert_array_equal(recovered, self.recording)
        np.testing.assert_array_equal(permutation[order], np.arange(12))

    def test_mixed_and_duplicate_chains_rejected(self):
        mixed = self.windows.copy()
        mixed[5] = self.rng.normal(size=(128, 14))
        with self.assertRaises(ValueError):
            reconstruct(mixed)
        mixed[5] = mixed[4]
        with self.assertRaises(ValueError):
            reconstruct(mixed)
        with self.assertRaises(ValueError):
            reconstruct(self.windows[:11])

    def test_feature_shapes_finiteness_and_example_independence(self):
        x = np.stack([self.recording, self.recording*2+10])
        for reference in ['none', 'car']:
            spectral = RecordingFeatures(reference=reference).fit(x[:1])
            f = spectral.transform(x)
            self.assertEqual(f.shape, (2, 196))
            self.assertTrue(np.isfinite(f).all())
            np.testing.assert_allclose(f[:1], spectral.transform(x[:1]))
            covariance = RecordingFeatures(kind='covariance', reference=reference).fit(x[:1])
            c = covariance.transform(x)
            self.assertEqual(c.shape, (2, 6, 14, 14))
            self.assertTrue((np.linalg.eigvalsh(c)>0).all())
            np.testing.assert_allclose(c[:1], covariance.transform(x[:1]))

    def test_diagnostic_counts_real_shared_samples(self):
        from eegis.diagnose_leakage import leakage_counts
        counts = leakage_counts(self.windows[:2], np.array([1, 1]), np.array([0]), np.array([1]))
        self.assertEqual(counts['shared_overlap_groups'], 1)
        self.assertEqual(counts['heldout_windows_with_shared_samples'], 1)
        self.assertEqual(counts['shared_samples_counting_window_occurrences'], 48)
        independent = np.stack([self.windows[0], self.rng.normal(size=(128, 14))])
        counts = leakage_counts(independent, np.array([1, 2]), np.array([0]), np.array([1]))
        self.assertEqual(sum(counts.values()), 0)

if __name__ == '__main__':
    unittest.main()
