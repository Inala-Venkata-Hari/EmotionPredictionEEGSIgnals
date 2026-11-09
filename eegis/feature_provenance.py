"""Fingerprint feature code dependencies and filter settings for cache reuse."""
import hashlib
import json
from pathlib import Path


DEPENDENCIES = ('author_features.py', 'paper_features.py', 'data.py', 'epochs.py',
                'feature_provenance.py')


def feature_fingerprint(module_dir=None, band_limits=None):
    if module_dir is None:
        module_dir = Path(__file__).parent
    if band_limits is None:
        from .author_features import BANDS
        band_limits = BANDS
    return {
        'schema_version': 1,
        'files_sha256': {name: hashlib.sha256((Path(module_dir)/name).read_bytes()).hexdigest()
                         for name in DEPENDENCIES},
        'band_limits_hz': [list(band) for band in band_limits],
    }


def fingerprint_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
