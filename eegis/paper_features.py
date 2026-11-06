"""EEGIS paper feature bank: 34 raw + 5*36 band features per channel.

Reference: Lara-Arellano et al., Eng 2025, 6, 75, Appendix A.
Unspecified parameters and numerical/formula decisions are explicit in FEATURE_SPEC.
Only existing NumPy/SciPy dependencies are used. No dataset-level fitting occurs here.
"""
import numpy as np
from scipy import signal, stats

from .data import BANDS, CHANNELS

FEATURE_NAMES = [
    'mean', 'standard_deviation', 'coefficient_variation', 'median', 'mode',
    'maximum', 'minimum', 'first_quartile', 'third_quartile', 'interquartile_range',
    'kurtosis', 'skewness', 'detrended_fluctuation', 'hjorth_activity',
    'hjorth_mobility', 'hjorth_complexity', 'permutation_entropy',
    'approximate_entropy', 'spectral_entropy', 'higuchi_fractal_dimension',
    'total_psd', 'centroid_psd', 'determinism', 'trapping_time',
    'diagonal_line_entropy', 'average_diagonal_line_length', 'recurrence_rate',
    'spectral_edge_25', 'spectral_edge_50', 'spectral_edge_75', 'hurst_exponent',
    'svd_entropy', 'petrosian_fractal_dimension', 'katz_fractal_dimension',
    'relative_band_power', 'band_amplitude',
]
LAYERS = ['raw', 'delta', 'theta', 'alpha', 'beta', 'gamma']
COLUMNS = [f'raw/{name}' for name in FEATURE_NAMES[:34]] + [
    f'{layer}/{name}' for layer in LAYERS[1:] for name in FEATURE_NAMES]
FEATURE_SPEC = {
    'paper_doi': '10.3390/eng6040075',
    'shape': [14, 214], 'sample_rate_hz': 128,
    'band_limits_hz': BANDS,
    'filter': '4th-order Butterworth SOS, forward-backward scipy.sosfiltfilt; no raw detrending/reference',
    'raw_omissions': 'relative_band_power and band_amplitude: assumed, not explicitly identified in paper',
    'spectrum': 'scipy.signal.periodogram, boxcar, nfft=128, detrend=False, density scaling',
    'entropy': 'natural logarithms, no normalization; permutation order=3 delay=1',
    'approximate_entropy': 'Chebyshev distance; m=2; r=0.2*within-series population std; self matches included',
    'recurrence': 'm=3 delay=1; Chebyshev distance <=0.2*within-series std; main diagonal excluded; min line=2',
    'dfa_hurst_scales': [4, 8, 16, 32],
    'dfa': 'integrated demeaned signal; disjoint blocks; linear detrending; log-log slope',
    'hurst': 'mean within-block rescaled range; sample standard deviation; log-log slope',
    'svd': 'delay embedding order=3 delay=1; entropy of normalized singular values',
    'higuchi': 'k=1..8; positive absolute curve increments; normalization (N-1)/(floor((N-1-m)/k)*k**2); negative log-length slope',
    'higuchi_departure': 'Appendix A22-A24 omit absolute increments/sign detail; conventional positive dimension is used',
    'hjorth_activity': 'mean square, as printed in A14 (not centered variance)',
    'petrosian': 'printed A43: 1+log2(N_sign_changes/(N-N_sign_changes)); not conventional Petrosian formula',
    'katz': 'printed A45: log10(curve_length/abs(last-first))/log10(N); not conventional Katz formula',
    'band_amplitude': 'printed A49-A50: mean hypot(signal, time_derivative); derivative=np.gradient(signal)*128; not Hilbert envelope',
    'relative_power': 'band periodogram sum divided by raw periodogram sum for the same example/channel',
    'edge_frequency': 'first discrete frequency with cumulative PSD >= specified fraction',
    'numerical_policy': 'epsilon=1e-12 in undefined ratios/logs; constant series has zero shape/complexity/entropy; nonfinite output raises',
    'mode': 'exact sample mode; smallest value breaks ties, as scipy.stats.mode',
    'fitted_population_statistics': False,
}


def _entropy(probability):
    p = np.asarray(probability)
    return -(p*np.log(np.maximum(p, 1e-12))).sum(axis=-1)


def _line_histogram(matrix, diagonal):
    """Histogram of Boolean line lengths, with zero delimiters between lines."""
    batch, n, _ = matrix.shape
    if diagonal:
        row, col = np.indices((n, n))
        lines = np.zeros((batch, 2*n-1, n+2), dtype=np.int8)
        lines[:, col-row+n-1, row+1] = matrix
    else:
        lines = np.pad(matrix.transpose(0, 2, 1), ((0, 0), (0, 0), (1, 1))).astype(np.int8)
    edges = np.diff(lines, axis=-1).reshape(-1)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    lengths = ends-starts
    owners = starts//((lines.shape[1])*(lines.shape[2]-1))
    return np.bincount(owners*(n+1)+lengths, minlength=batch*(n+1)).reshape(batch, n+1)


def _nonlinear(a):
    """Vectorized short-series entropy, recurrence, and scaling calculations."""
    nseries, n = a.shape
    std = a.std(axis=1)
    tolerance = .2*std
    phi = []
    distances3 = None
    for order in [2, 3]:
        embedded = np.lib.stride_tricks.sliding_window_view(a, order, axis=1)
        distances = np.max(np.abs(embedded[:, :, None, :]-embedded[:, None, :, :]), axis=-1)
        matches = distances <= tolerance[:, None, None]
        phi.append(np.log(matches.mean(axis=-1).clip(1e-12)).mean(axis=-1))
        if order == 3:
            distances3 = distances
    approximate = phi[0]-phi[1]
    embedded = np.lib.stride_tricks.sliding_window_view(a, 3, axis=1)
    patterns = np.argsort(embedded, axis=-1, kind='stable')
    codes = (patterns*np.array([1, 3, 9])).sum(axis=-1)
    owners = np.arange(nseries)[:, None]
    counts = np.zeros((nseries, 27))
    np.add.at(counts, (np.broadcast_to(owners, codes.shape), codes), 1)
    permutation = _entropy(counts/counts.sum(axis=-1, keepdims=True))
    singular = np.linalg.svd(embedded, compute_uv=False)
    svd = _entropy(singular/np.maximum(singular.sum(axis=-1, keepdims=True), 1e-12))
    recurrent = distances3 <= tolerance[:, None, None]
    diagonal_index = np.arange(recurrent.shape[1])
    recurrent[:, diagonal_index, diagonal_index] = False
    dh = _line_histogram(recurrent, diagonal=True)[:, 2:]
    vh = _line_histogram(recurrent, diagonal=False)[:, 2:]
    lengths = np.arange(2, recurrent.shape[1]+1)
    points = recurrent.sum(axis=(1, 2))
    determinism = (dh*lengths).sum(axis=-1)/np.maximum(points, 1)
    trapping = (vh*lengths).sum(axis=-1)/np.maximum(vh.sum(axis=-1), 1)
    diagonal_entropy = _entropy(dh/np.maximum(dh.sum(axis=-1, keepdims=True), 1))
    diagonal_length = (dh*lengths).sum(axis=-1)/np.maximum(dh.sum(axis=-1), 1)
    recurrence_rate = points/(recurrent.shape[1]**2)
    scales = np.array([4, 8, 16, 32])
    integrated = np.cumsum(a-a.mean(axis=-1, keepdims=True), axis=-1)
    fluctuation, rescaled = [], []
    for scale in scales:
        blocks = a[:, :n//scale*scale].reshape(nseries, -1, scale)
        profile = integrated[:, :n//scale*scale].reshape(nseries, -1, scale)
        residual = signal.detrend(profile, axis=-1, type='linear')
        fluctuation.append(np.sqrt(np.mean(residual**2, axis=(1, 2))))
        walk = np.cumsum(blocks-blocks.mean(axis=-1, keepdims=True), axis=-1)
        rs = np.ptp(walk, axis=-1)/np.maximum(blocks.std(axis=-1, ddof=1), 1e-12)
        rescaled.append(rs.mean(axis=-1))
    def slope(values, horizontal):
        horizontal = np.log(horizontal)
        horizontal -= horizontal.mean()
        return (np.log(np.maximum(np.stack(values, axis=-1), 1e-12))*horizontal).sum(axis=-1)/(horizontal**2).sum()
    dfa, hurst = slope(fluctuation, scales), slope(rescaled, scales)
    curve_lengths = []
    for k in range(1, 9):
        parts = []
        for m in range(k):
            sampled = a[:, m::k]
            steps = sampled.shape[1]-1
            parts.append(np.abs(np.diff(sampled, axis=-1)).sum(axis=-1)*(n-1)/(steps*k*k))
        curve_lengths.append(np.mean(parts, axis=0))
    higuchi = -slope(curve_lengths, np.arange(1, 9))
    return [dfa, permutation, approximate, svd, determinism, trapping,
            diagonal_entropy, diagonal_length, recurrence_rate, hurst, higuchi]


def series_features(a, raw_power=None, batch_size=96):
    """Calculate all 36 feature types for rows of finite 128-sample signals."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 128 or not np.isfinite(a).all():
        raise ValueError('Expected finite rows of 128 samples')
    output = []
    for start in range(0, len(a), batch_size):
        x = a[start:start+batch_size]
        mean, std = x.mean(axis=1), x.std(axis=1)
        var = std**2
        first, second = np.diff(x, axis=1), np.diff(x, n=2, axis=1)
        mobility = np.sqrt(first.var(axis=1)/np.maximum(var, 1e-12))
        complexity = np.sqrt(second.var(axis=1)/np.maximum(first.var(axis=1), 1e-12))/np.maximum(mobility, 1e-12)
        frequencies, psd = signal.periodogram(x, fs=128, window='boxcar', detrend=False, axis=1)
        power = psd.sum(axis=1)
        probability = psd/np.maximum(power[:, None], 1e-12)
        cumulative = np.cumsum(probability, axis=1)
        edges = [frequencies[np.argmax(cumulative >= q, axis=1)] for q in [.25, .5, .75]]
        dfa, pe, apen, svd, det, tt, dle, adll, rr, hurst, hfd = _nonlinear(x)
        centered = (x-mean[:, None])/np.maximum(std[:, None], 1e-12)
        sign_changes = (np.sign(first[:, 1:]) != np.sign(first[:, :-1])).sum(axis=1)
        petrosian = 1+np.log2(np.maximum(sign_changes, 1e-12)/np.maximum(128-sign_changes, 1e-12))
        length = np.abs(first).sum(axis=1)
        katz = np.log10(np.maximum(length, 1e-12)/np.maximum(np.abs(x[:, -1]-x[:, 0]), 1e-12))/np.log10(128)
        q1, q3 = np.quantile(x, [.25, .75], axis=1)
        amplitude = np.hypot(x, np.gradient(x, axis=1)*128).mean(axis=1)
        denominator = power if raw_power is None else raw_power[start:start+len(x)]
        features = np.stack([
            mean, std, std/np.where(np.abs(mean) > 1e-12, mean, 1e-12),
            np.median(x, axis=1), stats.mode(x, axis=1, keepdims=False).mode,
            x.max(axis=1), x.min(axis=1), q1, q3, q3-q1,
            (centered**4).mean(axis=1)-3, (centered**3).mean(axis=1), dfa,
            (x*x).mean(axis=1), mobility, complexity, pe, apen, _entropy(probability),
            hfd, power, (probability*frequencies).sum(axis=1), det, tt, dle, adll, rr,
            *edges, hurst, svd, petrosian, katz,
            power/np.maximum(denominator, 1e-12), amplitude,
        ], axis=1)
        constant = std <= 1e-12
        if constant.any():
            # Fixed convention for mathematically undefined constant-series descriptors.
            features[np.ix_(constant, [2, 10, 11, 12, 14, 15, 16, 17, 18, 19, 27, 28, 29, 30, 31, 32, 33])] = 0
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite paper features')
        output.append(features)
    return np.concatenate(output)


def paper_features(windows, batch_size=96, progress=None):
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (128, len(CHANNELS)) or not np.isfinite(x).all():
        raise ValueError('Expected finite raw windows with shape (N,128,14)')
    layers = [x]
    for band in BANDS:
        sos = signal.butter(4, band, btype='bandpass', fs=128, output='sos')
        layers.append(signal.sosfiltfilt(sos, x, axis=1))
    raw = x.transpose(0, 2, 1).reshape(-1, 128)
    _, psd = signal.periodogram(raw, fs=128, window='boxcar', detrend=False, axis=1)
    raw_power = psd.sum(axis=1)
    parts = []
    for layer, values in zip(LAYERS, layers):
        if progress:
            progress(f'Extracting {layer}: {len(x)} windows x 14 channels')
        a = values.transpose(0, 2, 1).reshape(-1, 128)
        f = series_features(a, raw_power, batch_size=batch_size)
        parts.append(f[:, :34] if layer == 'raw' else f)
    result = np.concatenate(parts, axis=1).reshape(len(x), 14, 214)
    if not np.isfinite(result).all():
        raise ValueError('Invalid feature bank')
    return result
