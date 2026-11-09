"""Small Torch-only dependency adapters for frozen LaBraM inference."""
import torch
from torch.nn.init import trunc_normal_


def register_model(fn):return fn


def to_2tuple(x):return x if isinstance(x,tuple) else (x,x)


def drop_path(x,drop_prob=0.,training=False):
    if drop_prob==0. or not training:return x
    keep=1-drop_prob
    mask=(torch.rand((x.shape[0],)+(1,)*(x.ndim-1),dtype=x.dtype,device=x.device)<keep).to(x.dtype)
    return x*mask/keep


def rearrange(x,pattern):
    if pattern=='B N A T -> B (N A) T':
        b,n,a,t=x.shape;return x.reshape(b,n*a,t)
    if pattern=='B C NA T -> B NA (T C)':
        b,c,n,t=x.shape;return x.permute(0,2,3,1).reshape(b,n,t*c)
    raise ValueError('Unsupported reference reshape '+pattern)
