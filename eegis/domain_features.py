"""Low-dimensional topographic contrasts and training-only latent alignment.

Latent clusters describe signal/contact variation. They are NOT participant IDs.
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .accuracy_features import prepared, spectral_features, validate
from .data import BANDS


def topographic_banks(x):
    raw=validate(x);z=prepared(raw)
    f,p=signal.welch(z,fs=128,nperseg=min(256,z.shape[1]),axis=1)
    power=np.stack([p[:,(f>=lo)&(f<hi)].sum(axis=1)*(f[1]-f[0]) for lo,hi in BANDS],axis=1).clip(1e-12)
    log=np.log(power)
    # Channel-order homologues: AF3/AF4, F7/F8, F3/F4, FC5/FC6, T7/T8, P7/P8, O1/O2.
    left=np.arange(7);right=np.arange(13,6,-1)
    lateral=(log[:,:,left]-log[:,:,right]).reshape(len(z),-1)
    topography=(log-log.mean(axis=2,keepdims=True))/log.std(axis=2,keepdims=True).clip(1e-9)
    spectral_shape=np.log(p[:,(f>=1)&(f<40)].clip(1e-12))
    spectral_shape-=spectral_shape.mean(axis=2,keepdims=True)
    spectral_shape/=spectral_shape.std(axis=2,keepdims=True).clip(1e-9)
    basic=spectral_features(raw)[0]
    # Raw offsets/scale are reserved for training-only nuisance-domain inference.
    domain=np.concatenate([raw.mean(axis=1),np.log(raw.std(axis=1).clip(1e-9))],axis=1)
    return {'lateral':lateral,'topography':topography.reshape(len(z),-1),
            'relative_spectrum':spectral_shape.reshape(len(z),-1),
            'spectral':basic,'domain':np.concatenate([domain,basic],axis=1)}


class LatentAlignment(TransformerMixin,BaseEstimator):
    def __init__(self,clusters=5,mode='center',shrinkage=5.):
        self.clusters=clusters;self.mode=mode;self.shrinkage=shrinkage

    def fit(self,x,y=None):
        x=np.asarray(x,dtype=float)
        self.domain_scaler_=StandardScaler().fit(x[:,:28])
        domain=self.domain_scaler_.transform(x[:,:28])
        self.feature_scaler_=StandardScaler().fit(x[:,28:])
        values=self.feature_scaler_.transform(x[:,28:])
        self.clusterer_=KMeans(n_clusters=min(self.clusters,len(x)),random_state=42,n_init=10).fit(domain)
        self.centres_=np.stack([values[self.clusterer_.labels_==k].sum(axis=0)/
                               (np.sum(self.clusterer_.labels_==k)+self.shrinkage)
                               for k in range(self.clusterer_.n_clusters)])
        distance=self.clusterer_.transform(domain)**2
        self.temperature_=np.median(np.min(distance,axis=1)).clip(1e-6)
        return self

    def transform(self,x):
        check_is_fitted(self,'centres_')
        x=np.asarray(x,dtype=float)
        values=self.feature_scaler_.transform(x[:,28:])
        distance=self.clusterer_.transform(self.domain_scaler_.transform(x[:,:28]))**2
        weights=np.exp(-(distance-distance.min(axis=1,keepdims=True))/(2*self.temperature_))
        weights/=weights.sum(axis=1,keepdims=True)
        if self.mode=='center':return values-weights@self.centres_
        if self.mode=='interact':
            return np.concatenate([values,(values[:,None,:]*weights[:,:,None]).reshape(len(values),-1)],axis=1)
        raise ValueError('Unknown alignment mode')


@dataclass
class DomainModel:
    estimator:object
    bank:str
    samples:int

    def features(self,x):
        x=validate(x)
        if x.shape[1]!=self.samples:raise ValueError('Incorrect input duration')
        return topographic_banks(x)[self.bank]

    def predict(self,x):return self.estimator.predict(self.features(x))
