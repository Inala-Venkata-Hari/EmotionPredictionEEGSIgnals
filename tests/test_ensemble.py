import unittest
import numpy as np
from eegis.ensemble import assert_group_separation, fit_meta, method_probabilities


class EnsembleTests(unittest.TestCase):
    def test_group_guards_reject_direct_and_outer_leakage(self):
        groups = np.array([1, 1, 2, 2, 3, 3])
        assert_group_separation(groups, np.array([0, 1]), np.array([2, 3]), [3])
        with self.assertRaises(ValueError):
            assert_group_separation(groups, np.array([0]), np.array([1]))
        with self.assertRaises(ValueError):
            assert_group_separation(groups, np.array([0]), np.array([2]), [1])
        with self.assertRaises(ValueError):
            assert_group_separation(groups, np.array([0]), np.array([2]), [2])

    def test_all_modes_have_normalized_probabilities(self):
        rng = np.random.default_rng(5)
        scores = rng.normal(size=(90, 4, 9)); labels = np.tile(np.arange(9), 10)
        calibrators, stacker = fit_meta(scores, labels)
        heldout = rng.normal(size=(6, 4, 9)); hard = heldout.argmax(axis=-1)
        prior = np.bincount(labels, minlength=9)/len(labels)
        result = method_probabilities(heldout, hard, calibrators, stacker, prior)
        self.assertEqual(set(result), {'HardVote', 'CalibratedSoftVote', 'Stacking', 'Majority'})
        for p in result.values():
            self.assertEqual(p.shape, (6, 9))
            self.assertTrue(np.isfinite(p).all())
            np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-8)

    def test_hard_vote_uses_training_prior_only_for_ties(self):
        rng = np.random.default_rng(1)
        scores = rng.normal(size=(90, 4, 9)); labels = np.tile(np.arange(9), 10)
        calibrators, stacker = fit_meta(scores, labels)
        heldout = np.zeros((2, 4, 9)); hard = np.array([[2, 2, 4, 4], [2, 2, 2, 4]])
        prior = np.zeros(9); prior[4] = .9; prior[2] = .1
        predictions = method_probabilities(heldout, hard, calibrators, stacker, prior)['HardVote'].argmax(axis=1)
        np.testing.assert_array_equal(predictions, [4, 2])


if __name__ == '__main__':
    unittest.main()
