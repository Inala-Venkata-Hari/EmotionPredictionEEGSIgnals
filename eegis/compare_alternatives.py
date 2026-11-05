"""Descriptive paired recording-group bootstrap on the previously reported holdout."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', default='artifacts_alternatives')
    args = parser.parse_args(); out = Path(args.artifacts)
    report = json.loads((out/'results.json').read_text())
    p = pd.read_csv(out/'test_predictions.csv')
    old = pd.read_csv('artifacts_improved/test_predictions.csv')
    assert p.path.tolist() == old.path.tolist()
    np.testing.assert_array_equal(p.label, old.label)
    winner = report['best_model']
    rng = np.random.default_rng(42)
    values = []
    for name, reference in [('previous_TangentLogistic', old.TangentLogistic), ('majority', p.Majority)]:
        difference = (p[winner].to_numpy() == p.label.to_numpy()).astype(int) - (reference.to_numpy() == p.label.to_numpy()).astype(int)
        groups = pd.DataFrame({'group': p.overlap_group, 'difference': difference}).groupby('group').difference.agg(['sum', 'count'])
        indices = rng.integers(0, len(groups), size=(5000, len(groups)))
        sums = groups['sum'].to_numpy(); counts = groups['count'].to_numpy()
        draws = sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
        interval = (np.quantile(draws, [.025, .975])*100).tolist()
        values.append({'reference': name, 'selected_model': winner,
                       'accuracy_difference_percentage_points': float(difference.mean()*100),
                       'paired_group_bootstrap_95_percent_interval_pp': interval})
    result = {'comparisons': values, 'groups': int(p.overlap_group.nunique()), 'replicates': 5000,
              'limitations': 'Descriptive uncertainty on a reused holdout; no model selection by these differences; not participant-independent evidence'}
    (out/'paired_comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
