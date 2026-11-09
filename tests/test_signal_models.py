import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from eegis.signal_models import BinaryPowerCNN, phase_connectivity, lagged_features, high_frequency_features


class SignalModelTests(unittest.TestCase):
    def test_connectivity_and_lags_are_example_local_and_bounded(self):
        rng=np.random.default_rng(18);x=rng.normal(size=(6,128,14))
        for fn in [phase_connectivity,lagged_features,high_frequency_features]:
            all_values=fn(x);separate=fn(x[[0,4]])
            self.assertTrue(np.isfinite(all_values).all())
            np.testing.assert_allclose(all_values[[0,4]],separate,rtol=1e-9,atol=1e-9)
        self.assertLessEqual(abs(lagged_features(x)).max(),1.+1e-12)

    def test_phase_locking_detects_known_synchronization(self):
        rng=np.random.default_rng(11);t=np.arange(1008)/128
        x=rng.normal(size=(2,1008,14))*.2
        x[0,:,0]=np.sin(2*np.pi*10*t)
        x[0,:,1]=np.sin(2*np.pi*10*t+.7)
        x[1,:,0]=np.sin(2*np.pi*10*t)
        x[1,:,1]=np.sin(2*np.pi*12*t)
        values=phase_connectivity(x)
        # Alpha is the third band; each band contains 4*91 pair features.
        alpha_first_pair=2*4*91
        self.assertGreater(values[0,alpha_first_pair],.95)
        self.assertLess(values[1,alpha_first_pair],.2)

    def test_neural_checkpoint_is_portable_and_records_training_only(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(12,128,14));y=np.array([6,8]*6)
        model=BinaryPowerCNN(epochs=2).fit(x,y)
        self.assertEqual(len(model.history_),2)
        self.assertEqual(model.predict(x[:3]).shape,(3,))
        self.assertTrue(set(model.predict(x[:3])).issubset({6,8}))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.joblib';joblib.dump(model,path)
            np.testing.assert_array_equal(joblib.load(path).predict(x[:3]),model.predict(x[:3]))


if __name__=='__main__':unittest.main()
