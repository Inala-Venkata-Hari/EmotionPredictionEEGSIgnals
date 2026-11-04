import unittest
import numpy as np
from eegis.improve import covariance_input,TangentFeatures

class SpatialTests(unittest.TestCase):
    def test_positive_covariance_and_finite_features(self):
        x=np.random.default_rng(1).normal(size=(5,128,14))
        c=covariance_input(x)
        self.assertEqual(c.shape,(5,5,14,14))
        self.assertTrue((np.linalg.eigvalsh(c)>0).all())
        transform=TangentFeatures().fit(c[:3])
        reference=transform.whitener_.copy()
        a=transform.transform(c[3:])
        self.assertEqual(a.shape,(2,525))
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_array_equal(transform.whitener_,reference)
        np.testing.assert_allclose(covariance_input(x[3:4]),c[3:4])

    def test_trace_normalization_is_amplitude_invariant(self):
        c=covariance_input(np.random.default_rng(2).normal(size=(3,128,14)))
        t=TangentFeatures(normalize=True).fit(c)
        np.testing.assert_allclose(t.transform(c),t.transform(c*9),atol=1e-10)

    def test_context_aggregation_keeps_groups_separate(self):
        from eegis.context import aggregate
        scores=np.array([[.9,.1],[.1,.9],[.7,.3],[.2,.8]])
        groups=np.array([1,2,1,2]);y=np.array([0,1,0,1])
        ids,p,labels=aggregate(scores,groups,y)
        np.testing.assert_array_equal(ids,[1,2])
        np.testing.assert_array_equal(p,labels)
        with self.assertRaises(ValueError):
            aggregate(scores,groups,np.array([0,1,1,1]))

if __name__=='__main__':unittest.main()
