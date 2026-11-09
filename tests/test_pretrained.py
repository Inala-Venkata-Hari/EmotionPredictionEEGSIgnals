from pathlib import Path
import unittest

import numpy as np

from eegis.data import CHANNELS
from eegis.labram_channels import get_input_chans
from eegis.pretrained_features import extract_mode,patches


class PretrainedTests(unittest.TestCase):
    def test_physical_channel_mapping_and_padding(self):
        self.assertEqual(get_input_chans([c.upper() for c in CHANNELS]),[0,7,16,18,28,38,60,81,83,68,46,34,22,24,11])
        x=np.random.default_rng(9).normal(size=(2,1008,14))
        out=patches(x,'physical')
        self.assertEqual(out.shape,(2,14,8,200))
        np.testing.assert_array_equal(out[:,:,-1,-25:],0.)
        np.testing.assert_allclose(out[:1],patches(x[:1],'physical'),rtol=1e-6,atol=1e-7)

    @unittest.skipUnless(Path('artifacts_pretrained/labram-base.pth').exists(),'Official checkpoint is excluded from source distribution')
    def test_frozen_encoder_features_are_example_local(self):
        x=np.random.default_rng(6).normal(size=(3,128,14))
        whole=extract_mode(x,'normalized');single=extract_mode(x[:1],'normalized')
        for key in ['cls','mean']:
            self.assertEqual(whole[key].shape,(3,200))
            self.assertTrue(np.isfinite(whole[key]).all())
            np.testing.assert_allclose(whole[key][:1],single[key],rtol=1e-4,atol=1e-4)


if __name__=='__main__':unittest.main()
