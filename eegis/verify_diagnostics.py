"""Audit diagnostic predictions and exact sample leakage without refitting models."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from .diagnose_leakage import leakage_counts
from .epochs import read_window
from .improve import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', default='artifacts_diagnostics')
    args = parser.parse_args()
    out = Path(args.artifacts)
    report = json.loads((out/'results.json').read_text())
    source = pd.read_csv(report['arguments']['manifest'])
    m = source.loc[source.split == 'train'].copy().reset_index(drop=True)
    pd.testing.assert_frame_equal(m, pd.read_csv(out/'source_training_manifest.csv'))
    predictions = pd.read_csv(out/'diagnostic_predictions.csv')
    measured = pd.read_csv(out/'leakage_control.csv')
    assert len(measured) == 12
    x = np.stack([read_window(Path(report['arguments']['data'])/p) for p in m.path])
    y = m.label.to_numpy(); g = m.overlap_group.to_numpy()
    for seed in [42, 43, 44]:
        random_split = next(StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(x, y))
        group_split = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(x, y, g))
        unique = np.unique(g)
        values = np.array([y[g == group][0] for group in unique])
        mapping = dict(zip(unique, np.random.default_rng(seed).permutation(values)))
        shuffled_y = np.array([mapping[group] for group in g])
        for protocol, (train, heldout) in [('invalid_random_window', random_split), ('valid_overlap_group', group_split)]:
            counts = leakage_counts(x, g, train, heldout)
            for target, labels in [('real_labels', y), ('shuffled_group_labels', shuffled_y)]:
                p = predictions.loc[(predictions.seed == seed) & (predictions.protocol == protocol) & (predictions.target == target)]
                row = measured.loc[(measured.seed == seed) & (measured.protocol == protocol) & (measured.target == target)].iloc[0]
                assert p.path.tolist() == m.loc[heldout, 'path'].tolist()
                np.testing.assert_array_equal(p.label, labels[heldout])
                np.testing.assert_array_equal(p.overlap_group, g[heldout])
                assert len(train) == row.training_windows and len(heldout) == row.heldout_windows
                for key, value in counts.items():
                    assert value == row[key]
                for metric, value in score(p.label, p.prediction).items():
                    assert abs(value-row[metric]) < 1e-12
    for item in report['summary']:
        rows = measured.loc[(measured.protocol == item['protocol']) & (measured.target == item['target'])]
        for metric in ['accuracy', 'macro_f1']:
            assert abs(rows[metric].mean()-item[f'{metric}_mean']) < 1e-12
            assert abs(rows[metric].std()-item[f'{metric}_std']) < 1e-12
    assert len(predictions) == int(measured.heldout_windows.sum())
    assert set(predictions.path) <= set(m.path)
    assert set(predictions.path).isdisjoint(set(source.loc[source.split != 'train', 'path']))
    verification = {'original_validation_and_test_excluded': True,
                    'all_twelve_control_predictions_audited': True,
                    'metrics_and_summary_recomputed': True,
                    'sample_overlap_counts_recomputed': True,
                    'scores_are_diagnostic_only': True}
    (out/'verification.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
