"""Reload the recording model and independently recompute saved evaluation metrics."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from .data import audit_shared_samples
from .epochs import load_recordings, predict_recording
from .improve import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', default='artifacts_epochs')
    args = parser.parse_args()
    out = Path(args.artifacts)
    r = json.loads((out/'results.json').read_text())
    meta = json.loads((out/'best_model.json').read_text())
    source = pd.read_csv(r['arguments']['manifest'])
    x, m = load_recordings(r['arguments']['data'], source)
    pd.testing.assert_frame_equal(m, pd.read_csv(out/'manifest.csv'))
    audit_shared_samples(x, m)
    winner = max(r['training_candidates'], key=lambda n: (
        r['training_candidates'][n]['training_cv_accuracy'],
        r['training_candidates'][n]['training_cv_macro_f1']))
    assert winner == r['best_model'] == meta['model']
    model = joblib.load(out/meta['file'])
    train = m.loc[m.split == 'train', 'label'].to_numpy()
    majority = int(np.bincount(train, minlength=9).argmax())
    with threadpool_limits(limits=4):
        for split in ['validation', 'test']:
            indices = np.flatnonzero(m.split == split)
            saved = pd.read_csv(out/f'{split}_predictions.csv')
            a = x[indices].reshape(len(indices), -1) if meta['flatten'] else x[indices]
            p = model.predict(a)
            np.testing.assert_array_equal(saved.overlap_group, m.loc[indices, 'overlap_group'])
            np.testing.assert_array_equal(saved.label, m.loc[indices, 'label'])
            np.testing.assert_array_equal(saved.prediction, p)
            np.testing.assert_array_equal(saved.majority, np.full(len(indices), majority))
            for name, predicted in [('selected', p), ('majority', saved.majority)]:
                measured = score(saved.label, predicted)
                for metric, value in measured.items():
                    assert abs(value-r['scores'][split][name][metric]) < 1e-12
            if split == 'test':
                cm = confusion_matrix(saved.label, p, labels=range(9))
                np.testing.assert_array_equal(cm, np.loadtxt(out/'confusion.csv', delimiter=','))
                paths = [Path(r['arguments']['data'])/p for p in json.loads(m.loc[indices[0], 'source_paths'])]
                # Filenames and caller order do not determine the recovered signal.
                inference = predict_recording(list(reversed(paths)), out)
                assert inference['class_id'] == int(p[0])
    verification = {'reconstruction_replayed': True, 'identical_source_split': True,
                    'cross_split_shared_samples': 0, 'checkpoint_predictions_match': True,
                    'metrics_recomputed': True, 'confusion_matrix_verified': True,
                    'selection_uses_training_cv_only': True, 'shuffled_window_inference_matches': True}
    (out/'verification.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
