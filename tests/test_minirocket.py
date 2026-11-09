import unittest

import numpy as np

from eegis.aeon_minirocket import MiniRocket,_static_fit,_static_transform_multi


class MiniRocketTests(unittest.TestCase):
    def test_adapter_preserves_reference_numeric_outputs_and_rng(self):
        x=np.random.default_rng(5).normal(size=(9,128,14))
        np.random.seed(17);before=np.random.get_state()
        model=MiniRocket(n_kernels=168).fit(x[:6])
        after=np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1])
        self.assertEqual(before[2:],after[2:])
        reference=_static_fit(model._input(x[:6]),168,32,42)
        for a,b in zip(model.parameters_,reference):np.testing.assert_array_equal(a,b)
        expected=_static_transform_multi(model._input(x[6:]),reference,MiniRocket._indices)
        np.testing.assert_array_equal(model.transform(x[6:]),expected)

    def test_test_examples_never_change_learned_thresholds(self):
        x=np.random.default_rng(8).normal(size=(12,128,14))
        model=MiniRocket(n_kernels=168).fit(x[:8]);before=model.parameters_[-1].copy()
        single=model.transform(x[8:9]);batch=model.transform(x[8:])
        np.testing.assert_array_equal(single,batch[:1])
        model.transform(x[8:]+5.)
        np.testing.assert_array_equal(before,model.parameters_[-1])
        different=MiniRocket(n_kernels=168).fit(x[8:])
        self.assertFalse(np.array_equal(before,different.parameters_[-1]))


if __name__=='__main__':unittest.main()
