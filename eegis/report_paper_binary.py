"""Write a measured, explicitly qualified report for the binary paper reproduction."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .train_paper_binary import dump, plot_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', default='artifacts_paper_binary')
    args = parser.parse_args()
    root = Path(args.artifacts)
    paths = [p for p in [root, root/'paper_window_diagnostic'] if (p/'results.json').exists()]
    if not paths:
        raise ValueError('Run the experiment first')
    environment = json.loads((root/'environment.json').read_text())
    report = [
        '# Binary paper-method reproduction', '',
        'Source: Lara-Arellano et al., *Feature Generation with Genetic Algorithms for Imagined Speech '
        'Electroencephalogram Signal Classification*, Eng 2025, 6, 75. '
        '[Paper](https://doi.org/10.3390/eng6040075).', '',
        'This experiment uses only **No (original class 6)** and **Sí (original class 8)**. '
        'It loads 924 original one-second windows, representing 77 exact-overlap groups. '
        'It does not train on the reconstructed 7.875-second segments.', '',
        'The paper uses a genetic algorithm, not a neural epoch loop. '
        'The smoke run uses 50 GA generations, plus initialization at generation 0, with 100 genomes. '
        'The smoke run tests kNN. The pipeline also supports a separate search for each of the ten paper classifiers. '
        'The ten-classifier comparison has not been run unless those names appear in results.json.', '',
        'Each raw window produces 14 × 214 values: 34 raw features and 36 features from each of five bands. '
        'A 228-bit genome contains 14 channel bits and 214 shared feature bits. '
        'Selected channel-feature matrices are flattened. StandardScaler is fitted only on the current fitting partition. '
        'kNN searches k in {5, 7, 13, 15, 17, 21, 37} for every candidate. '
        'Rank-weighted mating, uniform bit crossover, bit-flip mutation at 1/228, and top-100 survival implement the search.', '',
        '**This is a transparent implementation of the stated method, not a claim of an exact numerical 1:1 reproduction.** '
        'The paper describes an initial 80/20 split and a separate 20/80 validation experiment; '
        'these can be different stages. It reports k=7 for the GA output and k=9 for validation, '
        'although 9 is absent from its stated search candidates. It omits several feature and GA parameters. '
        '`feature_spec.json`, `feature_columns.csv`, and `plan.json` record the assumptions. '
        'The printed Hjorth activity, Petrosian, Katz, and band-amplitude definitions are retained. '
        'Higuchi uses conventional absolute curve increments and a positive dimension because the printed equations lack necessary details. '
        'The two raw feature omissions are assumed to be relative band power and band amplitude. '
        'Fitness uses a fixed inner validation partition; the paper does not clearly specify this partition.', '',
        '| Protocol | Model | Fitness validation accuracy | Test accuracy | Test macro-F1 | Majority test accuracy |',
        '|---|---|---:|---:|---:|---:|',
    ]
    summary = []
    for path in paths:
        results = json.loads((path/'results.json').read_text())
        plan = json.loads((path/'plan.json').read_text())
        if not (path/'verification.json').exists():
            raise ValueError(f'Independent verification is required before reporting: {path}')
        plot_run(path, results, {name: pd.read_csv(path/f'{name}_history.csv').to_dict('records')
                                for name in results['models']})
        for name, values in results['models'].items():
            report.append(f'| {results["protocol"]} | {name} | {values["fitness_validation_accuracy"]:.2%} | '
                          f'{values["test"]["accuracy"]:.2%} | {values["test"]["macro_f1"]:.4f} | '
                          f'{results["majority"]["accuracy"]:.2%} |')
        summary.append({'protocol': results['protocol'], 'diagnostic_only': results['diagnostic_only'],
                        'selected_model': results['selected_model'], 'models': results['models'],
                        'majority': results['majority'], 'overlap_audit': plan['overlap_audit']})
    report += ['',
        '**Grouped result:** all windows from an overlap group stay together. '
        'The outer training set combines the original training and validation groups; the original test groups stay fixed. '
        'An inner group split supplies fitness labels. After search, the model is refitted on all outer training windows. '
        'Neither test labels nor test features enter genetic search. '
        'The holdout has been reported in earlier experiments, so this result is exploratory. '
        'Repeated optimization can overfit the inner fitness split.', '',
        '**Window-split diagnostic:** stratified random rows approximate a conventional interpretation of the paper\'s 80/20 split. '
        'The paper does not establish this exact split. Overlapping EEG samples cross these partitions. '
        'Its score is a leakage diagnostic and must not be presented as valid speech-decoder accuracy. '
        'No winner is selected by comparing the two protocols.', '',
        '**Hardware:** installed PyTorch is '+environment['libraries']['torch']+'. '
        'The executed kNN device is '+environment['resolved_device']+'. '
        'CUDA execution: '+str(environment['cuda_executed'])+'. '
        'PyTorch handles Euclidean distance and stable neighbor sorting on CUDA when available. '
        'Feature extraction, scaling, voting, and the other nine scikit-learn classifiers run on CPU. '
        'The current CPU-only PyTorch build and unavailable NVIDIA driver prevent GPU validation. '
        'No CUDA timing or accuracy is claimed.', '',
        'The authors report 96% binary accuracy. That value is a paper result, not a result of this implementation. '
        'There is no verified participant mapping. Neither protocol establishes subject-independent performance.', '',
        '![Grouped genetic-search graph](training_graph.png)', '',
        '![Grouped kNN confusion matrix](knn_confusion.png)', '',
        '## Run', '', '```bash',
        'MPLCONFIGDIR=/tmp/eegis-matplotlib .venv/bin/python -u -m eegis.train_paper_binary --generations 50 --population 100 --device auto > paper_binary.log 2>&1',
        '.venv/bin/python -m eegis.verify_paper_binary',
        '.venv/bin/python -m eegis.report_paper_binary',
        '.venv/bin/python -m unittest discover -s tests -v',
        '```', '',
        'For a CUDA-enabled environment, use the same command with `--device cuda`. '
        'That option raises a clear error if CUDA is unavailable. '
        'Install a compatible CUDA-enabled PyTorch build and NVIDIA driver before that run. '
        'No cuML or CuPy dependency is required.', '',
        'For all ten classifiers, add `--classifiers all --out artifacts_paper_binary_all`. '
        'Each classifier receives its own complete GA search. This takes much longer than the kNN smoke run. '
        'Scikit-learn defaults follow the installed version; random seeds and Random Forest thread limits are explicit.', '',
        'For inference with the grouped checkpoint:', '', '```bash',
        '.venv/bin/python -m eegis.predict --artifacts artifacts_paper_binary "EEGIS - Electroencephalogram Imagined Speech dataset/raw/class_8/0.csv"',
        '```', '',
        'The best checkpoint is `best_model.joblib`. It includes raw-window feature extraction, '
        'the selected mask, training scaler, and classifier. Original label IDs 6 and 8 are retained. '
        'Caches, populations, and binary checkpoints remain local. Re-run training after cloning to reproduce them.', '',
        'Saved files include source and split manifests, feature specification, candidate fitness scores, '
        'all-generation histories, selected genomes, predictions, metrics, confusion matrices, plots, and independent verification.', '',
    ]
    report += ['## Selected configurations', '']
    for path in paths:
        results = json.loads((path/'results.json').read_text())
        selected = json.loads((path/f'{results["selected_model"]}_selection.json').read_text())
        report += [f'**{results["protocol"]}:** selected `{results["selected_model"]}`, k={selected["k"]}; '
                   f'{len(selected["selected_channels"])} channels × {len(selected["selected_features"])} feature types '
                   f'= {selected["flattened_features"]} input values. '
                   f'The search evaluated {selected["unique_evaluations"]:,} distinct genomes. '
                   f'Selected channels: {", ".join(selected["selected_channels"])}. '
                   f'Protocol runtime after feature preparation: {results["elapsed_seconds"]:.1f} seconds.', '']
    if (root/'results.json').exists():
        grouped = json.loads((root/'results.json').read_text())
        score = grouped['models'][grouped['selected_model']]['test']['accuracy']
        if score <= grouped['majority']['accuracy']:
            report += ['The grouped GA-selected model did not beat the majority reference in test accuracy. '
                       'The adaptive fitness score is not an independent generalization estimate. '
                       'No reproduction of the paper\'s 96% accuracy is claimed.', '']
    (root/'README.md').write_text('\n'.join(report))
    dump(root/'comparison_summary.json', summary)
    main_readme = Path('README.md')
    marker = '\n## Binary paper-method reproduction\n'
    text = main_readme.read_text()
    section = marker+'\nA separate **Sí/No-only** implementation follows the paper\'s 214-feature-per-channel and 228-bit genetic-search design. '
    section += 'The test uses **50 GA generations**, not neural epochs. '
    section += 'See [the detailed binary report](artifacts_paper_binary/README.md) for assumptions, hardware, commands, and leakage audits.\n\n'
    section += '\n'.join(report[report.index('| Protocol | Model | Fitness validation accuracy | Test accuracy | Test macro-F1 | Majority test accuracy |'):][:2+sum(len(json.loads((p/'results.json').read_text())['models']) for p in paths)])
    section += '\n\nThe grouped score is exploratory. The window-split score knowingly shares EEG samples across partitions and is **diagnostic only**. '
    section += 'Neither establishes subject-independent performance. The paper\'s 96% is not claimed as a reproduced result. '
    section += f'The executed device is {environment["resolved_device"]}; CUDA-capable code is provided, but GPU execution could not be tested on this host.\n'
    if marker in text:
        # This is currently the final section. Preserve future sections if they are added later.
        before, tail = text.split(marker, 1)
        next_section = tail.find('\n## ')
        suffix = tail[next_section:] if next_section >= 0 else ''
        main_readme.write_text(before+section+suffix)
    else:
        main_readme.write_text(text+'\n'+section)
    print('Wrote measured binary report and root README section')


if __name__ == '__main__':
    main()
