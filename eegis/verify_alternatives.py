"""Reload all alternative models and recompute predictions, metrics, and audits."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from .data import audit_shared_samples
from .epochs import read_window
from .improve import score
from .predict import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', default='artifacts_alternatives')
    args = parser.parse_args(); out = Path(args.artifacts)
    r = json.loads((out/'results.json').read_text())
    m = pd.read_csv(out/'manifest.csv')
    pd.testing.assert_frame_equal(m, pd.read_csv(r['arguments']['manifest']))
    folds = pd.read_csv(out/'training_cv_manifest.csv')
    source_training = m.loc[m.split == 'train', ['path', 'overlap_group', 'label']].reset_index(drop=True)
    pd.testing.assert_frame_equal(folds[['path', 'overlap_group', 'label']], source_training)
    assert set(folds.cv_fold) == {0, 1, 2}
    assert folds.groupby('overlap_group').cv_fold.nunique().max() == 1
    x = np.stack([read_window(Path(r['arguments']['data'])/path) for path in m.path])
    audit_shared_samples(x, m)
    expected = max(r['models'], key=lambda n: (r['models'][n]['training_cv_accuracy'], r['models'][n]['training_cv_macro_f1']))
    assert expected == r['best_model']
    with threadpool_limits(limits=4):
        for name, detail in r['models'].items():
            model = joblib.load(out/f'{name}.joblib')
            for split in ['validation', 'test']:
                indices = np.flatnonzero(m.split == split)
                saved = pd.read_csv(out/f'{split}_predictions.csv')
                assert saved.path.tolist() == m.loc[indices, 'path'].tolist()
                np.testing.assert_array_equal(saved.label, m.loc[indices, 'label'])
                p = model.predict(x[indices])
                np.testing.assert_array_equal(p, saved[name])
                for metric, value in score(saved.label, p).items():
                    assert abs(value-detail[split][metric]) < 1e-12
                if split == 'test':
                    cm = confusion_matrix(saved.label, p, labels=range(9))
                    np.testing.assert_array_equal(cm, np.loadtxt(out/f'{name}_confusion.csv', delimiter=','))
            print(name, 'reloaded predictions and metrics verified', flush=True)
        first = pd.read_csv(out/'test_predictions.csv').iloc[0]
        assert predict(Path(r['arguments']['data'])/first.path, out)['class_id'] == int(first[expected])
    report = {'original_split_unchanged': True, 'training_cv_groups_disjoint': True,
              'shared_samples_across_outer_partitions': 0, 'all_model_predictions_reproduced': True,
              'metrics_recomputed': True, 'confusion_matrices_verified': True,
              'training_cv_selection_verified': True, 'single_window_inference_verified': True}
    (out/'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
