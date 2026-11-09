"""Frozen official LaBraM representations; no encoder fitting on EEGIS."""
import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import signal
import torch

from .accuracy_features import validate
from .data import CHANNELS
from .labram_backbone import labram_base_patch200_200
from .labram_channels import get_input_chans
from .signal_models import broadband


ROOT=Path(__file__).resolve().parents[1]/'artifacts_pretrained'
MODES=['physical','normalized','car_physical','car_normalized']


def checkpoint_sha():
    path=ROOT/'labram-base.pth'
    data=path.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if len(data)!=96612769 or blob!='221148b5af403e9178629bb61e94b2e7d89ef172':
        raise ValueError('Official checkpoint hash differs')
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1)
def encoder():
    sha=checkpoint_sha()
    allowed=[(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,type(np.dtype('float64')),argparse.Namespace]
    with torch.serialization.safe_globals(allowed):
        checkpoint=torch.load(ROOT/'labram-base.pth',map_location='cpu',weights_only=True)
    state={k[len('student.'):]:v for k,v in checkpoint['model'].items() if k.startswith('student.')}
    rng=torch.random.get_rng_state()
    try:model=labram_base_patch200_200(num_classes=0,use_mean_pooling=False,init_values=.1)
    finally:torch.random.set_rng_state(rng)
    loaded=model.load_state_dict(state,strict=False)
    if loaded.missing_keys or set(loaded.unexpected_keys)!={'mask_token','lm_head.weight','lm_head.bias'}:
        raise ValueError('Encoder checkpoint does not match architecture: '+str(loaded))
    model.requires_grad_(False);model.eval()
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    channels=get_input_chans([c.upper() for c in CHANNELS])
    audit={'checkpoint_sha256':sha,'missing_encoder_keys':loaded.missing_keys,
           'unused_pretraining_keys':loaded.unexpected_keys,'encoder_parameters':sum(p.numel() for p in model.parameters()),
           'channel_names':CHANNELS,'position_indices_including_cls':channels,'frozen':True,'device':device,
           'learned_on_EEGIS':False,'pretraining_corpus_not_independently_audited':True}
    (ROOT/'encoder_loading_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return model,device,channels


def patches(x,mode):
    x=validate(x)
    if mode not in MODES:raise ValueError('Unknown frozen-encoder input mode')
    z=broadband(x,60.)
    if mode.startswith('car_'):z-=z.mean(axis=2,keepdims=True)
    if mode.endswith('normalized'):z/=z.std(axis=1,keepdims=True).clip(1e-9)
    else:z/=100. # Assumes native values are microvolts; independently uncalibrated.
    z=signal.resample_poly(z,25,16,axis=1)
    padding=(-z.shape[1])%200
    z=np.pad(z,[(0,0),(0,padding),(0,0)])
    return np.ascontiguousarray(z.transpose(0,2,1).reshape(len(z),14,-1,200),dtype=np.float32)


def extract_mode(x,mode):
    model,device,channels=encoder()
    x=validate(x);cls=[];means=[]
    torch.set_num_threads(4)
    with torch.inference_mode():
        for start in range(0,len(x),16):
            inputs=torch.from_numpy(patches(x[start:start+16],mode)).to(device)
            tokens=model.forward_features(inputs,input_chans=channels,return_all_tokens=True)
            cls.append(tokens[:,0].cpu().numpy());means.append(tokens[:,1:].mean(dim=1).cpu().numpy())
    return {'cls':np.concatenate(cls),'mean':np.concatenate(means)}


def pretrained_banks(x):
    out={}
    for mode in MODES:
        print('Frozen LaBraM:',mode,flush=True)
        for pooling,value in extract_mode(x,mode).items():out[mode+'__'+pooling]=value
    return out


@dataclass
class PretrainedModel:
    estimator:object
    bank:str
    samples:int

    def features(self,x):
        x=validate(x)
        if x.shape[1]!=self.samples:raise ValueError('Input duration differs')
        mode,pooling=self.bank.split('__')
        return extract_mode(x,mode)[pooling]

    def predict(self,x):return self.estimator.predict(self.features(x))
