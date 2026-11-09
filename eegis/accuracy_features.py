"""New binary EEG representations; earlier reproduction modules are unchanged.

Random temporal kernels are an independent ROCKET-inspired implementation,
not MiniROCKET or a reproduction of its reference code. All random weights
and thresholds depend only on a fixed seed and input shape, never labels.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from scipy import linalg, signal, stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .data import BANDS


def validate(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != 14 or x.shape[1] < 16 or not np.isfinite(x).all():
        raise ValueError('Expected finite EEG (examples, time samples, 14 channels)')
    return x


def prepared(x, reference='none'):
    z = signal.detrend(validate(x), axis=1, type='linear')
    if reference == 'car':
        z -= z.mean(axis=2, keepdims=True)
    elif reference != 'none':
        raise ValueError('Unknown reference')
    sos = signal.butter(4, [.5, 40], btype='bandpass', fs=128, output='sos')
    return signal.sosfiltfilt(sos, z, axis=1)


def spectral_features(x, reference='none'):
    z = prepared(x, reference)
    f, psd = signal.welch(z, fs=128, nperseg=min(256, z.shape[1]), axis=1)
    df = f[1] - f[0]
    power = np.stack([psd[:, (f >= lo) & (f < hi)].sum(axis=1)*df
                      for lo, hi in BANDS], axis=1).clip(1e-12)
    relative = power / power.sum(axis=1, keepdims=True)
    var = z.var(axis=1).clip(1e-12)
    first = np.diff(z, axis=1).var(axis=1).clip(1e-12)
    second = np.diff(z, n=2, axis=1).var(axis=1).clip(1e-12)
    mobility = np.sqrt(first / var)
    p = psd[:, (f >= .5) & (f < 40)]
    p = p / p.sum(axis=1, keepdims=True).clip(1e-12)
    entropy = -(p*np.log(p.clip(1e-12))).sum(axis=1) / np.log(p.shape[1])
    basic = np.concatenate([np.log(power).reshape(len(z), -1),
                            relative.reshape(len(z), -1), np.log(var), mobility,
                            np.sqrt(second/first)/mobility, entropy,
                            stats.skew(z, axis=1), stats.kurtosis(z, axis=1)], axis=1)
    # Integer-Hz bins use 1-Hz windows around each centre. No population fit.
    shape = np.stack([np.log(psd[:, (f >= hz-.5) & (f < hz+.5)].mean(axis=1).clip(1e-12))
                      for hz in range(1, 40)], axis=1).reshape(len(z), -1)
    return np.nan_to_num(basic), np.concatenate([np.nan_to_num(basic), shape], axis=1)


def covariance_features(x, reference='none'):
    z = signal.detrend(validate(x), axis=1, type='linear')
    if reference == 'car':
        z -= z.mean(axis=2, keepdims=True)
    elif reference != 'none':
        raise ValueError('Unknown reference')
    matrices = []
    for lo, hi in [(.5, 40), *BANDS]:
        a = signal.sosfiltfilt(signal.butter(4, [lo, hi], btype='bandpass', fs=128, output='sos'), z, axis=1)
        a -= a.mean(axis=1, keepdims=True)
        c = np.einsum('nti,ntj->nij', a, a)/(a.shape[1]-1)
        scale = (np.trace(c, axis1=-2, axis2=-1)/14).clip(1e-12)
        matrices.append(.9*c+.1*scale[:, None, None]*np.eye(14))
    return np.stack(matrices, axis=1)


class BinaryCSP(TransformerMixin, BaseEstimator):
    """Supervised spatial projections learned from the fitting fold only."""
    def __init__(self, pairs=2):
        self.pairs = pairs

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2 or not 1 <= self.pairs <= 7:
            raise ValueError('CSP requires two classes and 1..7 spatial pairs')
        normalized = x / np.trace(x, axis1=-2, axis2=-1)[..., None, None].clip(1e-12)
        means = [normalized[y == label].mean(axis=0) for label in self.classes_]
        filters = []
        for positive, negative in zip(*means):
            _, vectors = linalg.eigh(positive, positive+negative)
            columns = [*range(self.pairs), *range(14-self.pairs, 14)]
            filters.append(vectors[:, columns])
        self.filters_ = np.stack(filters)
        return self

    def transform(self, x):
        check_is_fitted(self, 'filters_')
        variance = np.einsum('bif,nbij,bjf->nbf', self.filters_, x, self.filters_, optimize=True).clip(1e-12)
        return np.log(variance/variance.sum(axis=-1, keepdims=True)).reshape(len(x), -1)


@njit(cache=True, parallel=True)
def apply_kernels(x, weights, channels, counts, lengths, dilations, paddings, biases):
    n, time, _ = x.shape
    kernels = len(lengths)
    out = np.empty((n, 2*kernels), dtype=np.float32)
    for row in prange(n):
        for k in range(kernels):
            length, dilation, padding = lengths[k], dilations[k], paddings[k]
            size = time + 2*padding - (length-1)*dilation
            positive, maximum = 0, -np.inf
            for offset in range(size):
                value = biases[k]
                for a in range(counts[k]):
                    channel = channels[k, a]
                    for j in range(length):
                        t = offset-padding+j*dilation
                        if 0 <= t < time:
                            value += weights[k, a, j]*x[row, t, channel]
                positive += value > 0
                maximum = max(maximum, value)
            out[row, 2*k] = positive/size
            out[row, 2*k+1] = maximum
    return out


@dataclass
class RandomTemporalFeatures:
    kernels: int = 2048
    seed: int = 42
    derivative: bool = False

    def fit(self, x):
        x = validate(x)
        time = x.shape[1] - int(self.derivative)
        self.input_samples_ = x.shape[1]
        rng = np.random.default_rng(self.seed)
        self.weights_ = np.zeros((self.kernels, 14, 11), dtype=np.float32)
        self.channels_ = np.zeros((self.kernels, 14), dtype=np.int32)
        self.counts_ = np.empty(self.kernels, dtype=np.int32)
        self.lengths_ = rng.choice(np.array([7, 9, 11], dtype=np.int32), self.kernels)
        self.dilations_ = np.empty(self.kernels, dtype=np.int32)
        self.paddings_ = np.empty(self.kernels, dtype=np.int32)
        self.biases_ = rng.uniform(-1, 1, self.kernels).astype(np.float32)
        for k, length in enumerate(self.lengths_):
            count = min(14, int(2**rng.uniform(0, np.log2(15))))
            self.counts_[k] = count
            self.channels_[k, :count] = rng.choice(14, count, replace=False)
            weight = rng.normal(size=(count, length))
            weight -= weight.mean()
            weight /= np.sqrt((weight**2).sum()).clip(1e-12)
            self.weights_[k, :count, :length] = weight
            dilation = max(1, int(2**rng.uniform(0, np.log2((time-1)/(length-1)))))
            self.dilations_[k] = dilation
            self.paddings_[k] = ((length-1)*dilation)//2 if rng.random() < .5 else 0
        return self

    def transform(self, x):
        x = validate(x)
        if x.shape[1] != self.input_samples_:
            raise ValueError('Input duration differs from fitted kernel bank')
        z = prepared(x)
        if self.derivative:
            z = np.diff(z, axis=1)
        z -= z.mean(axis=1, keepdims=True)
        z /= z.std(axis=1, keepdims=True).clip(1e-9)
        return apply_kernels(np.ascontiguousarray(z, dtype=np.float32), self.weights_, self.channels_,
                             self.counts_, self.lengths_, self.dilations_, self.paddings_, self.biases_)


def make_banks(x, kernels=2048):
    basic, shape = spectral_features(x)
    car, _ = spectral_features(x, 'car')
    raw = RandomTemporalFeatures(kernels=kernels).fit(x)
    derivative = RandomTemporalFeatures(kernels=kernels, derivative=True).fit(x)
    print('Random temporal features: raw', flush=True)
    r = raw.transform(x)
    print('Random temporal features: derivatives', flush=True)
    d = derivative.transform(x)
    return {'spectral': basic, 'shape': shape, 'car': car,
            'covariance': covariance_features(x), 'car_covariance': covariance_features(x, 'car'),
            'rocket': r, 'derivative': d, 'combined': np.concatenate([r, d, basic], axis=1)}, (raw, derivative)


@dataclass
class BinaryAccuracyModel:
    estimator: object
    bank: str
    kernels: tuple
    samples: int

    def features(self, x):
        x = validate(x)
        if x.shape[1] != self.samples:
            raise ValueError('Incorrect input duration')
        if self.bank == 'spectral':
            return spectral_features(x)[0]
        if self.bank == 'shape':
            return spectral_features(x)[1]
        if self.bank == 'car':
            return spectral_features(x, 'car')[0]
        if self.bank in ['covariance', 'car_covariance']:
            return covariance_features(x, 'car' if self.bank == 'car_covariance' else 'none')
        if self.bank == 'rocket':
            return self.kernels[0].transform(x)
        if self.bank == 'derivative':
            return self.kernels[1].transform(x)
        if self.bank == 'combined':
            return np.concatenate([self.kernels[0].transform(x), self.kernels[1].transform(x), spectral_features(x)[0]], axis=1)
        raise ValueError('Unknown feature bank')

    def predict(self, x):
        return self.estimator.predict(self.features(x))
