import unittest

import numpy as np

from eegis.domain_features import LatentAlignment,topographic_banks


class DomainModelTests(unittest.TestCase):
    def test_alignment_is_fitted_only_on_training_examples(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(50,35));x[:25,:28]-=6;x[25:,:28]+=6
        x[:25,28:]-=3;x[25:,28:]+=3
        model=LatentAlignment(clusters=2,shrinkage=0).fit(x[:40])
        before=model.centres_.copy();test=x[40:].copy();a=model.transform(test)
        np.testing.assert_array_equal(before,model.centres_)
        np.testing.assert_allclose(a[:1],model.transform(test[:1]),rtol=1e-10,atol=1e-10)
        train=model.transform(x[:40])
        self.assertLess(abs(train[:25].mean()),1e-10)
        self.assertLess(abs(train[25:].mean()),1e-10)

    def test_homologous_contrasts_and_spectra_are_example_local(self):
        x=np.random.default_rng(9).normal(size=(5,128,14))
        whole=topographic_banks(x);single=topographic_banks(x[:1])
        self.assertEqual(whole['lateral'].shape,(5,35))
        for name,values in whole.items():
            self.assertTrue(np.isfinite(values).all())
            np.testing.assert_allclose(values[:1],single[name],rtol=1e-9,atol=1e-9)


if __name__=='__main__':unittest.main()
