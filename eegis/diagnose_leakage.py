"""Training-only controls: intentionally invalid random-window split vs group split.

Diagnostic scores from this module are NOT deployable test performance.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

from .data import digest, features, preprocess
from .epochs import read_window
from .improve import score


def leakage_counts(x, groups, train, heldout):
    samples = {digest(row) for window in x[train] for row in window}
    shared_per_window = [sum(digest(row) in samples for row in window) for window in x[heldout]]
    cross_groups = set(groups[train]) & set(groups[heldout])
    return {'shared_overlap_groups': len(cross_groups),
            'heldout_windows_with_shared_samples': int(np.count_nonzero(shared_per_window)),
            'shared_samples_counting_window_occurrences': int(np.sum(shared_per_window))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='EEGIS - Electroencephalogram Imagined Speech dataset')
    parser.add_argument('--manifest', default='artifacts/manifest.csv')
    parser.add_argument('--out', default='artifacts_diagnostics')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    source = pd.read_csv(args.manifest)
    m = source.loc[source.split == 'train'].copy().reset_index(drop=True)
    if m.overlap_group.nunique() != 205 or set(m.split) != {'train'}:
        raise ValueError('This diagnostic requires the frozen original training partition')
    m.to_csv(out/'source_training_manifest.csv', index=False)
    x = np.stack([read_window(Path(args.data)/p) for p in m.path])
    f = features(preprocess(x)); y = m.label.to_numpy(); groups = m.overlap_group.to_numpy()
    records, predictions = [], []
    # Three repeated, prespecified train-only controls. No original validation/test data used.
    with threadpool_limits(limits=4):
        for seed in [42, 43, 44]:
            random_train, random_heldout = next(StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(f, y))
            group_train, group_heldout = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(f, y, groups))
            splits = [('invalid_random_window', random_train, random_heldout),
                      ('valid_overlap_group', group_train, group_heldout)]
            unique = np.unique(groups)
            true_group_labels = np.array([y[groups == group][0] for group in unique])
            shuffled = np.random.default_rng(seed).permutation(true_group_labels)
            mapping = dict(zip(unique, shuffled))
            shuffled_y = np.array([mapping[group] for group in groups])
            for split_name, train, heldout in splits:
                counts = leakage_counts(x, groups, train, heldout)
                if split_name == 'valid_overlap_group' and any(counts.values()):
                    raise AssertionError('The valid diagnostic split leaks source samples')
                for target, labels in [('real_labels', y), ('shuffled_group_labels', shuffled_y)]:
                    model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=seed, n_jobs=4)
                    model.fit(f[train], labels[train])
                    p = model.predict(f[heldout])
                    row = {'seed': seed, 'protocol': split_name, 'target': target,
                           'training_windows': len(train), 'heldout_windows': len(heldout),
                           **counts, **score(labels[heldout], p)}
                    records.append(row)
                    saved = m.loc[heldout, ['path', 'overlap_group']].copy()
                    saved['seed'] = seed
                    saved['protocol'] = split_name
                    saved['target'] = target
                    saved['label'] = labels[heldout]
                    saved['prediction'] = p
                    predictions.append(saved)
                    print(row, flush=True)
    frame = pd.DataFrame(records); frame.to_csv(out/'leakage_control.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out/'diagnostic_predictions.csv', index=False)
    summary = []
    for (protocol, target), rows in frame.groupby(['protocol', 'target']):
        summary.append({'protocol': protocol, 'target': target, 'runs': len(rows),
                        'accuracy_mean': float(rows.accuracy.mean()), 'accuracy_std': float(rows.accuracy.std()),
                        'macro_f1_mean': float(rows.macro_f1.mean()), 'macro_f1_std': float(rows.macro_f1.std())})
    report = {'purpose': 'Demonstrate overlap leakage and recording fingerprints, not estimate test-set classification accuracy',
              'original_partition_used': 'train only', 'original_validation_and_test_used': False,
              'model': 'RandomForest: 300 trees, balanced class weights, original 238 features',
              'group_shuffle': 'Permutation of the 205 training component labels, consistent across all windows of each component',
              'summary': summary, 'elapsed_seconds': time.time()-started,
              'arguments': vars(args), 'subject_independent': False}
    (out/'results.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    names = [f"{r['protocol']}\n{r['target']}" for r in summary]
    ax.bar(names, [r['accuracy_mean'] for r in summary], yerr=[r['accuracy_std'] for r in summary],
           color=['#b94d4d' if r['protocol'].startswith('invalid') else '#438a65' for r in summary], capsize=4)
    ax.axhline(1/9, color='black', linestyle='--', label='Uniform-guess reference')
    ax.set_ylim(0, 1); ax.set_ylabel('Diagnostic validation accuracy')
    ax.set_title('TRAINING-ONLY CONTROL: random-window protocol is invalid')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'leakage_control.png', dpi=160); plt.close(fig)
    verification = {'original_validation_and_test_excluded': set(m.path).isdisjoint(set(source.loc[source.split != 'train', 'path'])),
                    'valid_splits_have_zero_shared_samples': bool((frame.loc[frame.protocol == 'valid_overlap_group', 'shared_samples_counting_window_occurrences'] == 0).all()),
                    'all_prespecified_controls_completed': len(frame) == 12,
                    'scores_are_diagnostic_only': True}
    if not all(verification.values()):
        raise AssertionError('Diagnostic verification failed')
    (out/'verification.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
