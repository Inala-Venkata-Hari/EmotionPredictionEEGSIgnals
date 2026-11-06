"""Audit nested group exclusion and replay saved ensemble predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import confusion_matrix

from .ensemble import MEMBERS, METHODS, fit_meta
from .epochs import read_window
from .improve import score
from .predict import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', default='artifacts_ensembles')
    args = parser.parse_args(); out = Path(args.artifacts)
    r = json.loads((out/'results.json').read_text()); plan = json.loads((out/'plan.json').read_text())
    m = pd.read_csv(out/'manifest.csv')
    pd.testing.assert_frame_equal(m, pd.read_csv('artifacts/manifest.csv'))
    assignments = pd.read_csv(out/'nested_cv_group_assignments.csv')
    audit = json.loads((out/'fit_audit.json').read_text())
    training = m.loc[m.split == 'train'].reset_index(drop=True)
    expected_groups = set(training.overlap_group)
    assert len(assignments) == 3*len(expected_groups) and len(audit) == 36
    held_counts = assignments.loc[assignments.outer_role == 'heldout'].groupby('overlap_group').size()
    assert set(held_counts.index) == expected_groups and (held_counts == 1).all()
    for fold in range(3):
        a = assignments.loc[assignments.outer_fold == fold]
        assert set(a.overlap_group) == expected_groups and not a.overlap_group.duplicated().any()
        tr = set(a.loc[a.outer_role == 'train', 'overlap_group'])
        va = set(a.loc[a.outer_role == 'heldout', 'overlap_group'])
        entries = [item for item in audit if item['outer_fold'] == fold]
        assert len(entries) == 12
        for entry in entries:
            fit, pred = set(entry['fit_groups']), set(entry['prediction_groups'])
            assert not fit & pred
            if entry['phase'] == 'outer':
                assert fit == tr and pred == va and not entry['forbidden_outer_groups']
            else:
                inner = int(entry['phase'].split('_')[1])
                expected_pred = set(a.loc[(a.outer_role == 'train') & (a.inner_validation_fold == inner), 'overlap_group'])
                assert pred == expected_pred and fit == tr-expected_pred
                assert set(entry['forbidden_outer_groups']) == va and not (fit | pred) & va
        assert {(e['phase'], e['member']) for e in entries} == {(phase, member) for phase in ['inner_0', 'inner_1', 'outer'] for member in MEMBERS}
    saved = pd.read_csv(out/'training_oof_predictions.csv')
    pd.testing.assert_frame_equal(saved[['path', 'label', 'overlap_group']], training[['path', 'label', 'overlap_group']])
    for name in METHODS:
        for metric, value in score(saved.label, saved[name]).items():
            assert abs(value-r['models'][name]['training_cv'][metric]) < 1e-12
    winner = max(r['models'], key=lambda n: (r['models'][n]['training_cv']['accuracy'], r['models'][n]['training_cv']['macro_f1']))
    assert winner == r['best_model']
    ensemble = joblib.load(out/'best_model.joblib')
    assert ensemble.method == winner
    for name, sha in plan['base_checkpoint_sha256'].items():
        assert hashlib.sha256((Path(r['arguments']['base'])/f'{name}.joblib').read_bytes()).hexdigest() == sha
    matrix = np.load(out/'training_oof_scores.npz')
    np.testing.assert_array_equal(matrix['labels'], training.label)
    assert matrix['scores'].shape == (len(training), 4, 9) and np.isfinite(matrix['scores']).all()
    np.testing.assert_allclose(ensemble.prior, np.bincount(training.label, minlength=9)/len(training))
    with threadpool_limits(limits=4):
        calibrators, stacker = fit_meta(matrix['scores'], matrix['labels'])
        for expected, fitted in [*zip(calibrators, ensemble.calibrators), (stacker, ensemble.stacker)]:
            np.testing.assert_allclose(expected[-1].coef_, fitted[-1].coef_, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(expected[-1].intercept_, fitted[-1].intercept_, rtol=1e-6, atol=1e-8)
        for split in ['validation', 'test']:
            frame = m.loc[m.split == split]
            x = np.stack([read_window(Path(r['arguments']['data'])/p) for p in frame.path])
            predictions = pd.read_csv(out/f'{split}_predictions.csv')
            assert predictions.path.tolist() == frame.path.tolist()
            np.testing.assert_array_equal(predictions.label, frame.label)
            for name, probability in ensemble.predict_all(x).items():
                p = probability.argmax(axis=1)
                np.testing.assert_array_equal(p, predictions[name])
                for metric, value in score(predictions.label, p).items():
                    assert abs(value-r['models'][name][split][metric]) < 1e-12
                if split == 'test':
                    np.testing.assert_array_equal(confusion_matrix(predictions.label, p, labels=range(9)), np.loadtxt(out/f'{name}_confusion.csv', delimiter=','))
            if split == 'test':
                assert predict(Path(r['arguments']['data'])/frame.path.iloc[0], out)['class_id'] == int(predictions[winner].iloc[0])
    result = {'source_split_unchanged': True, 'all_36_fits_have_disjoint_groups': True,
              'outer_groups_excluded_from_inner_meta_features': True,
              'training_cv_scores_recomputed': True, 'meta_models_reproduced_from_group_oof_scores': True,
              'base_checkpoint_hashes_verified': True, 'all_holdout_predictions_and_metrics_reproduced': True,
              'confusion_matrices_verified': True, 'inference_verified': True}
    (out/'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
