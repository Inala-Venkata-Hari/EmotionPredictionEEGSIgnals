from pathlib import Path
import unittest

import numpy as np
import torch

from eegis.finetune_features import LastBlockClassifier,prefix_tokens
from eegis.pretrained_features import encoder,patches


@unittest.skipUnless(Path('artifacts_pretrained/labram-base.pth').exists(),'Official checkpoint excluded from source')
class FineTuneTests(unittest.TestCase):
    def test_prefix_plus_original_last_block_matches_official_encoder(self):
        x=np.random.default_rng(3).normal(size=(3,128,14));model,device,channels=encoder()
        prefix=prefix_tokens(x)
        with torch.inference_mode():
            rebuilt=model.norm(model.blocks[-1](torch.tensor(prefix,device=device),rel_pos_bias=None))
            expected=model.forward_features(torch.tensor(patches(x,'normalized'),device=device),input_chans=channels,return_all_tokens=True)
        np.testing.assert_allclose(rebuilt.cpu().numpy(),expected.cpu().numpy(),rtol=1e-5,atol=1e-5)

    def test_task_adaptation_does_not_mutate_pretrained_encoder(self):
        base,_,_=encoder();before=base.blocks[-1].mlp.fc1.weight.detach().cpu().clone()
        x=np.random.default_rng(5).normal(size=(8,128,14));tokens=prefix_tokens(x);labels=np.array([6,8]*4)
        fitted=LastBlockClassifier(epochs=1).fit(tokens,labels)
        torch.testing.assert_close(base.blocks[-1].mlp.fc1.weight.detach().cpu(),before,rtol=0,atol=0)
        self.assertEqual(fitted.predict(tokens).shape,(8,))


if __name__=='__main__':unittest.main()
