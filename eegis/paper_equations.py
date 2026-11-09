"""EEGIS feature profile following the paper's printed equations.

Existing feature profiles are intentionally unchanged. Undefined expressions are
NaN, never silently replaced by another formula. The caller chooses error or a
documented imputation policy. Unspecified numerical choices are in specification().
"""
from dataclasses import asdict, dataclass
from importlib.metadata import version

import antropy as ant
import numpy as np
from scipy import signal, stats

from .data import BANDS, CHANNELS
from .paper_features import COLUMNS, FEATURE_NAMES, _line_histogram

TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


@dataclass(frozen=True)
class EquationConfig:
    higuchi: str = 'conventional'
    kmax: int = 10
    power_low: float = 0.
    power_high: float = 64.
    recurrence_std_fraction: float = .2

    def __post_init__(self):
        if self.higuchi not in ['literal', 'conventional']:
            raise ValueError('Higuchi must be literal or conventional')
        if not 2 <= self.kmax < 64:
            raise ValueError('Require 2 <= kmax < 64')
        if not 0 <= self.power_low < self.power_high <= 64:
            raise ValueError('Invalid total-power limits')
        if self.recurrence_std_fraction <= 0:
            raise ValueError('Recurrence threshold must be positive')


def specification(config):
    return {
        'profile': 'printed-paper-equations-v1', 'config': asdict(config),
        'paper': 'https://doi.org/10.3390/eng6040075',
        'sample_rate': 128, 'shape': [14,214], 'columns': COLUMNS,
        'bands': [list(band) for band in BANDS],
        'filter': 'Butterworth order 4, scipy b/a filtfilt on raw once, default odd padding',
        'spectrum': 'one-sided density periodogram, boxcar, no detrending, nfft=128',
        'coefficient_variation': 'population std / mean, A3',
        'quartiles': 'median of sorted lower/upper 64 samples, A8/A9',
        'hjorth_activity': 'mean square, A14',
        'entropy': 'natural logarithms; permutation order3 delay1; ApEn order2 r=.2*population_std',
        'dfa': 'AntroPy default; exact scales are absent from paper',
        'hurst': 'disjoint rescaled-range blocks, population std, scales2,4,8,16,32,64,128',
        'recurrence': 'scalar distance < configured fraction*std; identity included; maximal lines, minimum2',
        'petrosian': '1+log2(sign_changes/(128-sign_changes)), A43',
        'katz': 'log10(sum(abs(diff))/abs(last-first))/log10(128), A45',
        'relative_power': 'integrate RAW PSD over matching band / RAW PSD over configured total range, A46-A48',
        'amplitude': 'hypot(filtered_signal, time_derivative); trapezoid mean over sampled duration, A49-A50; no second filter',
        'undefined_values': 'NaN plus per-column audit; never silently clipped to epsilon',
        'remaining_assumptions': [
            'Paper does not give exact PSD estimator, padding, recurrence construction, embedding settings, scales or zero-denominator policy.',
            'Total-power frequency limits are configurable; default covers the complete 0..64Hz Nyquist range.',
            'First34 raw features omit the two band-dependent fields, consistent with feature count and table.',
            'Literal A22-A24 omit absolute increments and can require logarithms of nonpositive curve lengths.',
            'Conventional Higuchi is an explicit alternative, not a literal A22-A24 implementation.',
        ],
        'libraries': {name:version(name) for name in ['numpy','scipy','antropy']},
    }


def _divide(a,b):
    a,b = np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    out = np.full(a.shape,np.nan)
    return np.divide(a,b,out=out,where=b!=0)


def literal_higuchi(x,kmax=10):
    """Direct signed lengths and positive slope of A22-A24; undefined -> NaN."""
    x = np.asarray(x,dtype=float)
    lengths = []
    for k in range(1,kmax+1):
        components = []
        for start in range(k):
            values = x[start::k]
            components.append(np.diff(values).sum()/((len(values)-1)*k))
        lengths.append(np.mean(components))
    lengths = np.asarray(lengths)
    if not np.isfinite(lengths).all() or np.any(lengths<=0):
        return np.nan
    return float(np.polyfit(np.log(np.arange(1,kmax+1)),np.log(lengths),1)[0])


def _recurrence(x,fraction):
    matrix = np.abs(x[:,:,None]-x[:,None,:]) < (fraction*x.std(axis=1))[:,None,None]
    diagonal = _line_histogram(matrix,True)[:,2:]
    vertical = _line_histogram(matrix,False)[:,2:]
    lengths = np.arange(2,x.shape[1]+1)
    counts = diagonal.sum(axis=1)
    probability = _divide(diagonal,counts[:,None])
    entropy = -np.sum(np.where(probability>0,probability*np.log(np.maximum(probability,1e-300)),0),axis=1)
    entropy[counts==0] = 0
    determinism = _divide((diagonal*lengths).sum(axis=1),matrix.sum(axis=(1,2)))
    trapping = _divide((vertical*lengths).sum(axis=1),vertical.sum(axis=1))
    trapping[vertical.sum(axis=1)==0] = 0
    average = _divide((diagonal*lengths).sum(axis=1),counts)
    average[counts==0] = 0
    return determinism,trapping,entropy,average,matrix.mean(axis=(1,2))


def _hurst(x):
    values = []
    scales = 2**np.arange(1,8)
    for n in scales:
        blocks = x.reshape(len(x),-1,n)
        walk = np.cumsum(blocks-blocks.mean(axis=-1,keepdims=True),axis=-1)
        std = blocks.std(axis=-1)
        ratios = np.divide(np.ptp(walk,axis=-1),std,out=np.zeros_like(std),where=std!=0)
        values.append(ratios.mean(axis=1))
    with np.errstate(divide='ignore',invalid='ignore'):
        answer = np.polyfit(np.log(scales),np.log(np.stack(values)),1)[0]
    return answer


def series_features(x,config=None,batch_size=128):
    """34 core values; band-dependent columns are assembled with RAW context."""
    config = config or EquationConfig()
    x = np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=128 or not np.isfinite(x).all() or batch_size<1:
        raise ValueError('Expected finite rows of 128 EEG samples and positive batch size')
    output = []
    for start in range(0,len(x),batch_size):
        a = np.ascontiguousarray(x[start:start+batch_size])
        mean,std = a.mean(axis=1),a.std(axis=1)
        first,second = np.diff(a,axis=1),np.diff(a,n=2,axis=1)
        mobility = np.sqrt(_divide(first.var(axis=1),std**2))
        complexity = _divide(np.sqrt(_divide(second.var(axis=1),first.var(axis=1))),mobility)
        ordered = np.sort(a,axis=1)
        q1,q3 = np.median(ordered[:,:64],axis=1),np.median(ordered[:,64:],axis=1)
        frequency,psd = signal.periodogram(a,fs=128,window='boxcar',detrend=False,axis=1)
        probability = _divide(psd,psd.sum(axis=1,keepdims=True))
        spectrum_entropy = -np.sum(np.where(probability>0,probability*np.log(np.maximum(probability,1e-300)),0),axis=1)
        cumulative = np.cumsum(probability,axis=1)
        edges = [frequency[np.argmax(cumulative>=q,axis=1)] for q in [.25,.5,.75]]
        det,tt,dle,adll,rr = _recurrence(a,config.recurrence_std_fraction)
        changes = (np.sign(first[:,1:])!=np.sign(first[:,:-1])).sum(axis=1)
        with np.errstate(divide='ignore',invalid='ignore'):
            petrosian = 1+np.log2(_divide(changes,128-changes))
            katz = np.log10(_divide(np.abs(first).sum(axis=1),np.abs(a[:,-1]-a[:,0])))/np.log10(128)
        def apen(row):
            return ant.app_entropy(row,order=2,tolerance=.2*float(row.std())) if row.std()>0 else np.nan
        modes = []
        for row in a:
            unique,counts = np.unique(row,return_counts=True)
            modes.append(unique[np.argmax(counts)])
        values = [mean,std,_divide(std,mean),np.median(a,axis=1),modes,
            a.max(axis=1),a.min(axis=1),q1,q3,q3-q1,stats.kurtosis(a,axis=1),stats.skew(a,axis=1),
            [ant.detrended_fluctuation(row) if row.std()>0 else np.nan for row in a],
            np.mean(a*a,axis=1),mobility,complexity,
            [ant.perm_entropy(row,order=3,delay=1,normalize=False)*np.log(2) for row in a],
            [apen(row) for row in a],spectrum_entropy,
            [np.nan if row.std()==0 else (literal_higuchi(row,config.kmax) if config.higuchi=='literal'
                                         else ant.higuchi_fd(row,kmax=config.kmax)) for row in a],
            psd.sum(axis=1),_divide((psd*frequency).sum(axis=1),psd.sum(axis=1)),
            det,tt,dle,adll,rr,*edges,_hurst(a),
            [ant.svd_entropy(row,order=3,delay=1,normalize=False)*np.log(2) if row.std()>0 else np.nan for row in a],
            petrosian,katz]
        values = np.stack(values,axis=1)
        values[~np.isfinite(values)] = np.nan
        output.append(values)
    return np.concatenate(output)


def filter_layers(windows):
    x = np.asarray(windows,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(128,len(CHANNELS)) or not np.isfinite(x).all():
        raise ValueError('Expected finite (N,128,14) raw EEG windows')
    layers = [x]
    for band in BANDS:
        b,a = signal.butter(4,band,fs=128,btype='bandpass')
        layers.append(signal.filtfilt(b,a,x,axis=1))
    return layers


def band_amplitude(filtered):
    """A49/A50 on an already filtered series, without another filter."""
    filtered = np.asarray(filtered,dtype=float)
    envelope = np.hypot(filtered,np.gradient(filtered,1/128,axis=-1))
    return TRAPZ(envelope,dx=1/128,axis=-1)/(127/128)


def equation_features(windows,config=None,batch_size=128,progress=None):
    config = config or EquationConfig()
    layers = filter_layers(windows)
    raw = layers[0].transpose(0,2,1).reshape(-1,128)
    frequency,raw_psd = signal.periodogram(raw,fs=128,window='boxcar',detrend=False,axis=1)
    total = (frequency>=config.power_low)&(frequency<=config.power_high)
    denominator = TRAPZ(raw_psd[:,total],frequency[total],axis=1)
    parts = []
    for index,layer in enumerate(layers):
        if progress:
            progress(f'Printed-equation features: {index+1}/6 layers')
        a = layer.transpose(0,2,1).reshape(-1,128)
        core = series_features(a,config,batch_size)
        if index:
            band = BANDS[index-1]
            mask = (frequency>=band[0])&(frequency<=band[1])
            relative = _divide(TRAPZ(raw_psd[:,mask],frequency[mask],axis=1),denominator)
            core = np.column_stack([core,relative,band_amplitude(a)])
        parts.append(core)
    result = np.concatenate(parts,axis=1).reshape(len(windows),14,214)
    result[~np.isfinite(result)] = np.nan
    return result
