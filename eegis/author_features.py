"""Reconstruct the paper's 34+5*36 bank using the author's Signalytica code.

Signalytica 0.1.20 is MIT licensed: https://github.com/Edgar-La/signalytica
The optimized kernels below are numerically tested against its public functions.
Eight AntroPy functions follow the commented calls in that source. The full
paper training program, exact versions and GA rates are not public/pinned.
"""
from collections import Counter
import hashlib
import importlib
from importlib.metadata import version
from pathlib import Path

import antropy as ant
import numpy as np
from scipy import signal, stats

from .data import BANDS
from .epochs import read_window
from .paper_features import COLUMNS, FEATURE_NAMES, _line_histogram

TRAPEZOID = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

PUBLISHED_GENOME = (
    '100010111101100111100010010101110011000000110001110111010101111011110101110010011001'
    '001001100101111110101000000001001101010011000101100111010000100001000101100100011101'
    '011110100100111011100011100001100000100010001111101110010001'
)


def published_genome():
    g = np.array(list(PUBLISHED_GENOME), dtype=np.uint8)
    if len(g) != 228 or g[:14].sum() != 8 or g[14:].sum() != 103:
        raise AssertionError('Published genome transcription failed')
    return g


class _LegacyNumpy:
    """Adapt removed np.trapz within the upstream module, without changing NumPy."""
    def __getattr__(self, name):
        return TRAPEZOID if name == 'trapz' else getattr(np, name)


def reference_library():
    module = importlib.import_module('signalytica.signalytica')
    module.np = _LegacyNumpy()
    return module.SignaLytica()


def source_spec():
    module = importlib.import_module('signalytica.signalytica')
    return {
        'paper': 'https://doi.org/10.3390/eng6040075',
        'author_library': 'https://github.com/Edgar-La/signalytica',
        'author_thesis': 'https://ri-ng.uaq.mx/bitstream/123456789/12198/1/IGMAC-320366.pdf',
        'library_version': version('signalytica'), 'antropy_version': version('antropy'),
        'numeric_libraries': {name: version(name) for name in ['numpy', 'scipy', 'numba', 'scikit-learn']},
        'library_source_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        'feature_names': FEATURE_NAMES, 'columns': COLUMNS,
        'bank_shape': [14, 214],
        'feature_order_evidence': 'Paper feature list and Section 3 genome; checked against Table 2 activations',
        'bank_rule': '34 non-band-specific features on raw; same 34 plus matching-band relative power and RMS amplitude for each supplied band',
        'signal_source': 'Original raw/delta/theta/alpha/beta/gamma CSVs; no signal transformation before feature extraction',
        'author_behavior_retained': {
            'coeff_var': 'np.var, not std/mean', 'hjorth_activity': 'np.var, not mean square',
            'mode': 'Counter, first occurrence wins ties',
            'psd': 'Welch fs=128 nperseg=128, defaults including Hann window and constant detrending',
            'recurrence': 'scalar absolute differences, epsilon=0.2*std, strict <, identity diagonal included',
            'determinism_diagonal_entropy_length': 'all suffixes of diagonal runs >=2, as upstream (not corrected maximal-run convention)',
            'recurrence_rate': 'scalar absolute difference <0.1, including identity diagonal',
            'hurst': 'normalized series; disjoint scales 2,4,8,16,32,64,128; population std; upstream polynomial slope',
            'band_amplitude': 'matching-band filtfilt then RMS, including filtering an already filtered supplied band',
            'relative_power': 'inclusive Welch integral in matching band / integral 0.5..100 Hz of that same input series',
            'antropy': 'DFA, permutation(normalize=True), app_entropy(defaults), spectral_entropy(sf=128,fft,normalize=True), Higuchi(defaults), SVD(normalize=True), Petrosian(defaults), Katz(defaults)',
        },
        'compatibility': 'Upstream-module-only np.trapz alias to np.trapezoid; vectorization verified against public source',
        'remaining_unknowns': [
            'The paper misspells Signalytica; identity established through author thesis and package author metadata.',
            'AntroPy 0.1.9 predates the paper but the original version is not specified.',
            'AntroPy methods are commented out in public Signalytica; their documented calls are reconstructed here.',
            'The full author GA/training source, split seed, mutation/crossover rates and exact feature assembly are not available.',
            'Mapping matching-band relative power/amplitude to the final two columns is inferred from paper feature count and Table 2.',
            'Table 2 bottom counts swap theta/alpha (14/16); published genome/row activations yield theta=16 and alpha=14.',
            'Published genome was selected by authors on this dataset; fixed-mask scores are not an independently selected feature test.',
        ],
    }


def _recurrence_features(x):
    distance = np.abs(x[:, :, None]-x[:, None, :])
    rp = distance < (.2*x.std(axis=1))[:, None, None]
    maximum = _line_histogram(rp, diagonal=True)
    # Each maximal diagonal run of length L contributes suffixes L,L-1,...,2.
    suffix = np.cumsum(maximum[:, ::-1], axis=1)[:, ::-1][:, 2:]
    lengths = np.arange(2, x.shape[1]+1)
    line_count = suffix.sum(axis=1)
    line_points = (suffix*lengths).sum(axis=1)
    determinant = line_points/np.maximum(rp.sum(axis=(1, 2)), 1)
    probability = suffix/np.maximum(line_count[:, None], 1)
    entropy = -(probability*np.log(np.maximum(probability, 1e-300))).sum(axis=1)
    diagonal_length = line_points/np.maximum(line_count, 1)
    vertical = _line_histogram(rp, diagonal=False)[:, 2:]
    trapping = (vertical*lengths).sum(axis=1)/np.maximum(vertical.sum(axis=1), 1)
    recurrence_rate = (distance < .1).mean(axis=(1, 2))
    return determinant, trapping, entropy, diagonal_length, recurrence_rate


def _hurst(x):
    z = (x-x.mean(axis=1, keepdims=True))/x.std(axis=1, keepdims=True)
    scales = 2**np.arange(1, 8)
    values = []
    for n in scales:
        blocks = z.reshape(len(x), -1, n)
        walk = np.cumsum(blocks-blocks.mean(axis=-1, keepdims=True), axis=-1)
        std = blocks.std(axis=-1)
        values.append((np.ptp(walk, axis=-1)/np.where(std == 0, 1e-4, std)).mean(axis=1))
    return np.polyfit(np.log(scales), np.log(np.stack(values)), 1)[0]


def author_series_features(x, band=None, batch_size=128):
    """34 core values, plus two matching-band values when band is supplied."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 128 or not np.isfinite(x).all():
        raise ValueError('Expected finite 128-sample rows')
    if np.any(x.std(axis=1) == 0):
        raise ValueError('Upstream descriptors are undefined on constant series; do not silently impute')
    rows = []
    for start in range(0, len(x), batch_size):
        a = x[start:start+batch_size]
        mean, std, var = a.mean(axis=1), a.std(axis=1), a.var(axis=1)
        diff, second = np.diff(a, axis=1), np.diff(a, n=2, axis=1)
        mobility = np.sqrt(diff.var(axis=1)/var)
        complexity = np.sqrt(second.var(axis=1)/diff.var(axis=1))/mobility
        f, psd = signal.welch(a, fs=128, nperseg=128, axis=1)
        total = TRAPEZOID(psd, f, axis=1)
        centroid = (psd*f).sum(axis=1)/psd.sum(axis=1)
        cumulative = np.cumsum(psd, axis=1)
        cumulative /= cumulative[:, -1:]
        edges = [f[np.argmax(cumulative >= q, axis=1)] for q in [.25, .5, .75]]
        det, tt, dle, adll, rr = _recurrence_features(a)
        modes = [Counter(series).most_common(1)[0][0] for series in a]
        q1, q3 = np.percentile(a, [25, 75], axis=1)
        values = [mean, std, var, np.median(a, axis=1), modes, a.max(axis=1), a.min(axis=1),
                  q1, q3, q3-q1, stats.kurtosis(a, axis=1), stats.skew(a, axis=1),
                  [ant.detrended_fluctuation(series) for series in a], var, mobility, complexity,
                  [ant.perm_entropy(series, normalize=True) for series in a],
                  [ant.app_entropy(series) for series in a],
                  [ant.spectral_entropy(series, sf=128, method='fft', normalize=True) for series in a],
                  [ant.higuchi_fd(series.astype(float)) for series in a], total, centroid,
                  det, tt, dle, adll, rr, *edges, _hurst(a),
                  [ant.svd_entropy(series, normalize=True) for series in a],
                  [ant.petrosian_fd(series) for series in a], [ant.katz_fd(series) for series in a]]
        if band is not None:
            selected = (f >= band[0]) & (f <= band[1])
            full = (f >= .5) & (f <= 100)
            relative = TRAPEZOID(psd[:, selected], f[selected], axis=1)/TRAPEZOID(psd[:, full], f[full], axis=1)
            b, denominator = signal.butter(4, band, fs=128, btype='band')
            filtered = signal.filtfilt(b, denominator, a, axis=1)
            values += [relative, np.sqrt(np.mean(filtered**2, axis=1))]
        result = np.stack(values, axis=1)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite author features; inspect source instead of hiding invalid values')
        rows.append(result)
    return np.concatenate(rows)


def author_features(layers, batch_size=128, progress=None):
    """layers is (N,6,128,14), exactly aligned to the raw manifest."""
    layers = np.asarray(layers, dtype=np.float64)
    if layers.ndim != 4 or layers.shape[1:] != (6, 128, 14):
        raise ValueError('Expected aligned (N,6,128,14) signal layers')
    parts = []
    for i, name in enumerate(['raw', 'delta', 'theta', 'alpha', 'beta', 'gamma']):
        if progress:
            progress(f'Author features: {name}, {len(layers)} windows')
        values = layers[:, i].transpose(0, 2, 1).reshape(-1, 128)
        part = author_series_features(values, None if i == 0 else BANDS[i-1], batch_size)
        parts.append(part)
    return np.concatenate(parts, axis=1).reshape(len(layers), 14, 214)


def load_layers(data, manifest):
    layers = []
    for band in ['raw', 'delta', 'theta', 'alpha', 'beta', 'gamma']:
        layers.append(np.stack([read_window(Path(data)/p.replace('raw/', band+'/', 1)) for p in manifest.path]))
    return np.stack(layers, axis=1)


class AuthorBinaryModel:
    def __init__(self, genome, scaler, estimator):
        self.genome, self.scaler, self.estimator = genome, scaler, estimator
        self.classes_ = np.array([6, 8])

    def predict_features(self, features):
        from .paper_ga import genome_columns
        a = features.reshape(len(features), -1)[:, genome_columns(self.genome)]
        return self.estimator.predict(self.scaler.transform(a))

    def predict_layers(self, layers):
        return self.predict_features(author_features(layers))

    def predict(self, raw):
        raise ValueError('This author-source checkpoint requires all six supplied layers. Use predict_layers or predict_features.')
