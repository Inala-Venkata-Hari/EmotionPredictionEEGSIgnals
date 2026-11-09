"""Phase/lag features and a small fixed-duration binary EEG power network."""
from dataclasses import dataclass

import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
from torch import nn

from .accuracy_features import validate
from .data import BANDS


def broadband(x, high=40.):
    z = signal.detrend(validate(x), axis=1, type='linear')
    if high > 50:
        b,a = signal.iirnotch(50.,30.,fs=128)
        z = signal.filtfilt(b,a,z,axis=1)
    return signal.sosfiltfilt(signal.butter(4,[.5,high],btype='bandpass',fs=128,output='sos'),z,axis=1)


def phase_connectivity(x, batch=32):
    """PLV, wPLI, imaginary coherence, and envelope correlation, per example."""
    x = validate(x)
    left,right = np.triu_indices(14,1)
    result = []
    for start in range(0,len(x),batch):
        z = signal.detrend(x[start:start+batch],axis=1,type='linear')
        features = []
        for lo,hi in BANDS:
            band = signal.sosfiltfilt(signal.butter(4,[lo,hi],btype='bandpass',fs=128,output='sos'),z,axis=1)
            analytic = signal.hilbert(band,axis=1)
            envelope = np.abs(analytic)
            phases = analytic/envelope.clip(1e-12)
            plv = np.abs(np.einsum('ntc,ntd->ncd',phases,phases.conj())/z.shape[1])[:,left,right]
            cross = analytic[:,:,left]*analytic[:,:,right].conj()
            imaginary = cross.imag
            wpli = np.abs(imaginary.mean(axis=1))/np.abs(imaginary).mean(axis=1).clip(1e-12)
            energy = (envelope**2).mean(axis=1)
            coherence = np.abs(imaginary.mean(axis=1))/np.sqrt(energy[:,left]*energy[:,right]).clip(1e-12)
            amplitude = envelope-envelope.mean(axis=1,keepdims=True)
            amplitude /= amplitude.std(axis=1,keepdims=True).clip(1e-12)
            correlation = (np.einsum('ntc,ntd->ncd',amplitude,amplitude)/z.shape[1])[:,left,right]
            features.extend([plv,wpli,coherence,correlation])
        result.append(np.concatenate(features,axis=1))
    return np.concatenate(result)


def lagged_features(x):
    z = broadband(x)
    vectors = []
    for lag in [1,2,4,8,16]:
        a,b = z[:,:-lag].copy(),z[:,lag:].copy()
        a -= a.mean(axis=1,keepdims=True);b -= b.mean(axis=1,keepdims=True)
        a /= a.std(axis=1,keepdims=True).clip(1e-12)
        b /= b.std(axis=1,keepdims=True).clip(1e-12)
        vectors.append((np.einsum('ntc,ntd->ncd',a,b)/a.shape[1]).reshape(len(z),-1))
    return np.concatenate(vectors,axis=1)


def high_frequency_features(x):
    z = broadband(x,60.)
    f,p = signal.welch(z,fs=128,nperseg=min(256,z.shape[1]),axis=1)
    bands = [(4,8),(8,13),(13,30),(30,40),(40,49),(51,60)]
    power = np.stack([p[:,(f>=lo)&(f<hi)].sum(axis=1)*(f[1]-f[0]) for lo,hi in bands],axis=1).clip(1e-12)
    shape = np.stack([np.log(p[:,(f>=hz-.5)&(f<hz+.5)].mean(axis=1).clip(1e-12))
                      for hz in [*range(1,49),*range(52,60)]],axis=1)
    return np.concatenate([np.log(power).reshape(len(z),-1),
                           (power/power.sum(axis=1,keepdims=True)).reshape(len(z),-1),
                           shape.reshape(len(z),-1)],axis=1)


def signal_banks(x):
    print('Computing phase connectivity',flush=True)
    connectivity = phase_connectivity(x)
    print('Computing lagged correlations and high frequencies',flush=True)
    lagged,high = lagged_features(x),high_frequency_features(x)
    return {'connectivity':connectivity,'lagged':lagged,'high_frequency':high,
            'joint':np.concatenate([connectivity,lagged,high],axis=1),'raw':x}


class Square(nn.Module):
    def forward(self,x):
        return x.square()


class SafeLog(nn.Module):
    def forward(self,x):
        return torch.log(x.clamp_min(1e-6))


class BinaryPowerCNN(ClassifierMixin,BaseEstimator):
    def __init__(self,normalization='window',high=40.,epochs=50,seed=42):
        self.normalization=normalization;self.high=high;self.epochs=epochs;self.seed=seed

    def _input(self,x):
        z=broadband(x,self.high)
        if self.normalization=='window':
            z-=z.mean(axis=1,keepdims=True)
            z/=z.std(axis=1,keepdims=True).clip(1e-9)
        elif self.normalization=='train':
            z=(z-self.mean_)/self.std_
        else:
            raise ValueError('Unknown normalization')
        return np.ascontiguousarray(np.clip(z,-8,8).transpose(0,2,1)[:,None],dtype=np.float32)

    def fit(self,x,y):
        torch.set_num_threads(4);torch.manual_seed(self.seed);torch.use_deterministic_algorithms(True)
        self.classes_,encoded=np.unique(y,return_inverse=True)
        if list(self.classes_)!=[6,8]:
            raise ValueError('Expected No/Sí labels 6/8')
        z=broadband(x,self.high)
        self.mean_=z.mean(axis=(0,1),keepdims=True)
        self.std_=z.std(axis=(0,1),keepdims=True).clip(1e-9)
        xx=torch.from_numpy(self._input(x));yy=torch.as_tensor(encoded,dtype=torch.long)
        self.network_=nn.Sequential(nn.Conv2d(1,8,(1,25),padding=(0,12),bias=False),
            nn.Conv2d(8,16,(14,1),groups=8,bias=False),nn.BatchNorm2d(16),Square(),
            nn.AdaptiveAvgPool2d((1,1)),SafeLog(),nn.Dropout(.5),nn.Flatten(),nn.Linear(16,2))
        self.training_device_='cuda' if torch.cuda.is_available() else 'cpu'
        device=torch.device(self.training_device_);self.network_.to(device)
        counts=np.bincount(encoded,minlength=2)
        loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(len(y)/(2*counts),dtype=torch.float32,device=device))
        optimizer=torch.optim.Adam(self.network_.parameters(),lr=.001,weight_decay=.01)
        rng=np.random.default_rng(self.seed);self.history_=[]
        for epoch in range(self.epochs):
            self.network_.train();loss_sum=0.;correct=0
            for indices in np.array_split(rng.permutation(len(xx)),max(1,int(np.ceil(len(xx)/32)))):
                a,b=xx[indices].to(device),yy[indices].to(device)
                a=torch.roll(a,int(rng.integers(-8,9)),dims=-1)
                a=a+.02*torch.randn_like(a)
                optimizer.zero_grad(set_to_none=True);logits=self.network_(a);loss=loss_fn(logits,b)
                loss.backward();optimizer.step()
                loss_sum+=float(loss.detach())*len(indices);correct+=int((logits.argmax(dim=1)==b).sum())
            self.history_.append({'epoch':epoch+1,'training_loss':loss_sum/len(xx),'training_accuracy':correct/len(xx)})
        # Portable checkpoint and measured inference on CPU, even after CUDA fitting.
        self.network_.to('cpu');self.network_.eval()
        return self

    def predict(self,x):
        self.network_.eval()
        xx=torch.from_numpy(self._input(x))
        with torch.no_grad():
            predicted=torch.cat([self.network_(batch).argmax(dim=1) for batch in xx.split(128)]).numpy()
        return self.classes_[predicted]


@dataclass
class SignalModel:
    estimator:object
    bank:str
    samples:int

    def features(self,x):
        x=validate(x)
        if x.shape[1]!=self.samples:
            raise ValueError('Input duration differs')
        if self.bank=='raw':return x
        if self.bank=='connectivity':return phase_connectivity(x)
        if self.bank=='lagged':return lagged_features(x)
        if self.bank=='high_frequency':return high_frequency_features(x)
        if self.bank=='joint':return np.concatenate([phase_connectivity(x),lagged_features(x),high_frequency_features(x)],axis=1)
        raise ValueError('Unknown feature bank')

    def predict(self,x):
        return self.estimator.predict(self.features(x))
