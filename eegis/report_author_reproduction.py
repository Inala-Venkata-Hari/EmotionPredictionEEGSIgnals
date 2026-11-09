"""Report all measured author-source reproduction attempts without selecting on test scores."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .train_paper_binary import dump,plot_run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',default='artifacts_author_reproduction')
    args=p.parse_args();out=Path(args.artifacts)
    paths=[out,out/'within_window_grouped',out/'paper_window_diagnostic']
    for target in paths:
        if not (target/'verification.json').exists():
            raise ValueError('Verify completed experiments before reporting')
    spec=json.loads((out/'feature_spec.json').read_text())
    plan=json.loads((out/'experiment_plan.json').read_text())
    fixed=pd.read_csv(out/'published_mask_results.csv')
    results=[json.loads((p/'results.json').read_text()) for p in paths]
    for path,result in zip(paths,results):
        plot_run(path,result,{'knn':pd.read_csv(path/'knn_history.csv').to_dict('records')})
    rows=[]
    for r in results:
        candidate=r['models']['knn']
        name=r['protocol']+(' / within-window scaling' if r.get('normalization')=='within-window' else '')
        rows.append({'protocol':name,'diagnostic_only':r['diagnostic_only'],
                     'fitness_accuracy':candidate['fitness_validation_accuracy'],
                     'test_accuracy':candidate['test']['accuracy'],'test_macro_f1':candidate['test']['macro_f1'],
                     'majority_accuracy':r['majority']['accuracy'],'k':candidate['k'],
                     'channels':len(candidate['channels']),'feature_types':candidate['feature_types']})
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    grouped_rows=[r for r in rows if not r['diagnostic_only']]
    chosen=max(range(len(grouped_rows)),key=lambda i:grouped_rows[i]['fitness_accuracy'])
    recommended=grouped_rows[chosen]
    dump(out/'recommendation.json',{'selection':'highest predeclared grouped inner fitness accuracy only',
         'selected_protocol':recommended['protocol'],'model_directory':str(out if chosen==0 else out/'within_window_grouped'),
         'test_scores_not_used_for_selection':True,'all_grouped_results_reported':True})
    reached=any(r['test_accuracy']>=.96 for r in grouped_rows)
    dump(out/'target_status.json',{'paper_target':.96,'grouped_test_accuracies':[r['test_accuracy'] for r in grouped_rows],
         'reproduced_on_any_grouped_test':reached,
         'no_protocol_selected_by_test_score':True,'all_fixed_mask_cases_reported':len(fixed),
         'exact_original_experiment_still_not_identifiable':True})
    text=['# Author-source reproduction attempt', '',
      'This experiment attempts the 96% binary accuracy reported by '
      '[Lara-Arellano et al.](https://doi.org/10.3390/eng6040075). '
      'It preserves earlier results and uses only No (class 6) and Sí (class 8).', '',
      '## Recovered source and environment', '',
      'The paper spells the feature library "signalityca". The author\'s '
      '[thesis](https://ri-ng.uaq.mx/bitstream/123456789/12198/1/IGMAC-320366.pdf) identifies '
      '[Signalytica](https://pypi.org/project/signalytica/). '
      'Its [public source](https://github.com/Edgar-La/signalytica) supplies concrete functions that differ from the paper equations. '
      'The exact installed source hash is in feature_spec.json.', '',
      'Signalytica 0.1.20 declares NumPy 1.21.6 and SciPy 1.7.3. '
      'An isolated **Python 3.10** environment uses these exact two pins. '
      'AntroPy 0.1.9 and scikit-learn 1.6.1 are explicit compatible paper-era choices; '
      'the paper does not identify their exact original versions. '
      'See requirements-author.txt and requirements-author-lock.txt. '
      'Both the main and isolated environments pass dependency checks. '
      'The old library is isolated rather than forcing old NumPy/SciPy onto Python 3.14.', '',
      '| Detail | Author implementation retained |', '|---|---|',
      '| Coefficient variation | Returns variance |',
      '| Hjorth activity | Returns variance |',
      '| Mode | First occurrence wins ties |',
      '| Power spectrum | Welch, 128 Hz, 128 samples, Hann window, constant detrending |',
      '| Recurrence | Scalar differences, identity diagonal included, source suffix-count convention |',
      '| Band amplitude | Matching-band filter followed by RMS |',
      '| Relative band power | Inclusive Welch integral divided by same-input 0.5–100 Hz integral |',
      '| Complexity functions | Exact AntroPy calls shown in commented source, including normalization flags |', '',
      'Vectorized recurrence and statistical calculations are tested against the installed author functions. '
      'No corrected textbook formula silently replaces an unusual author implementation. '
      'The six original supplied signal layers are used directly. '
      'The feature bank is 14 × 214: 34 raw values and 36 values per filtered band.', '',
      '## Published genome checks', '',
      'The 228-bit genome is transcribed from Section 3. '
      'It selects AF3, T7, O1, O2, P8, T8, F4, F8 and 103 feature positions. '
      'These yield 824 flattened values. Table 2 row activations agree with the feature order. '
      'Its bottom theta/alpha counts appear swapped: the genome yields 16/14, not 14/16.', '',
      'The paper reports k=7 for the GA output and k=9 for a separate validation experiment; '
      '9 is absent from its stated k-search candidates. '
      'Section 2.1 describes an initial 80/20 split and Section 3 a separate 20/80 validation experiment. '
      'These can be different stages, so the two ratios alone are not a contradiction. '
      'The following fixed checks report all declared cases, with seed 42. '
      'The global-scaler case tests a possible interpretation of standardization before splitting. '
      '**These are diagnostics:** the published feature mask may already have used all released data. '
      'Window splits additionally share EEG samples; global scaling also uses held-out statistics. '
      'No case is selected as a winner using its test score.', '',
      '| Fixed-mask case | k | Fit/test windows | Accuracy | Macro-F1 | Shared test windows |',
      '|---|---:|---:|---:|---:|---:|']
    for r in fixed.to_dict('records'):
        text.append(f'| {r["case"]} | {r["k"]} | {r["train_windows"]}/{r["test_windows"]} | '
                    f'{r["accuracy"]:.2%} | {r["macro_f1"]:.4f} | {r["heldout_windows_with_shared_samples"]} |')
    text+=['','## Full genetic searches','',
      f'The three runs use **{plan["generations"]} generations**, plus initialization, '
      f'with **{plan["population"]} genomes**. '
      'Each chromosome has 14 channel bits and 214 feature bits. '
      'Rank-weighted mating, uniform crossover, bit-flip mutation and parent-plus-offspring survival are used. '
      'The paper does not state the crossover/mutation probabilities or random seed; '
      'p=0.5 crossover bits and mutation 1/228 are declared implementation choices.', '',
      'kNN searches {5,7,13,15,17,21,37} for every genome. '
      'The optimized evaluator is tested against the original sklearn-metric evaluator. '
      'Fitness labels belong only to an inner validation partition. '
      'Neither outer test features nor outer test labels enter the search. '
      'Selection is frozen before refitting the training scaler and classifier on outer training windows. '
      'There is no target-based stopping or seed search.', '',
      'A third, predeclared grouped run resolves a textual ambiguity: standardization before flattening may mean '
      'z-scoring each feature column across selected electrodes within each example. '
      'This variant has no population-scaler statistics. Its addition was recorded before either first full-run test score was inspected. '
      'It uses the same frozen outer and inner group assignments. '
      'The two grouped interpretations are compared by inner fitness only, and all test scores are reported.', '',
      'The within-example variant uses actual scikit-learn scaling and an independent prediction for each k. '
      'Two-electrode masks produce many distance ties. A preliminary Torch-neighbor implementation failed parity '
      '(9 of 180 predictions differed) and was rejected before reporting any test metric. '
      'The corrected variant reran all 100 generations; the incomplete run is archived locally. '
      'A two-electrode tie regression test checks exact sklearn prediction parity. '
      'This variant uses CPU to preserve the reference library\'s tie behavior.', '',
      'All numerical thread pools are fixed at four threads. '
      'Reference predictions repeat at that setting. Thread count can affect equal-distance selection in sklearn; '
      'the author\'s original setting is unknown. It is not searched using accuracy.', '',
      '| Protocol | Fitness accuracy | Test accuracy | Test macro-F1 | Majority test accuracy | Selected k |',
      '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f'| {r["protocol"]} | {r["fitness_accuracy"]:.2%} | {r["test_accuracy"]:.2%} | '
                    f'{r["test_macro_f1"]:.4f} | {r["majority_accuracy"]:.2%} | {r["k"]} |')
    text+=['', '**Outcome:** '+('The grouped score reached the numeric target, subject to the limitations below.' if reached else
          'The paper\'s 96% was not reproduced on the grouped test.'), '',
      'The window-split number is an overlap-contaminated diagnostic, not valid decoder accuracy. '
      'Fitness accuracy is adaptively optimized and is not independent test performance. '
      'The original grouped test has been used before; new scores remain exploratory. '
      'Participant identifiers are unavailable, so subject-independent testing is not established.', '',
      '## Remaining reproduction limits', '',
      'The complete author GA/training program and exact split indices were not located. '
      'The public Signalytica package comments out eight AntroPy methods; this code restores the shown calls. '
      'The original AntroPy/sklearn versions, feature assembly details and GA rates remain unspecified. '
      'Thus, source behavior and two dependency pins are recovered, but a complete numerical 1:1 claim is not justified. '
      'No authors or teammates were messaged.', '',
      '## Reproduce','',
      'The historical grouped and window-search results used Torch kNN on CPU. '
      'The command below explicitly selects that legacy backend. '
      'The default CPU backend has since been corrected to scikit-learn defaults; '
      'use a different output directory for a new experiment. '
      'See the [adversarial review](../artifacts_adversarial_review/README.md) for the bug, scope of verification, and cache fixes.',
      '', '```bash',
      'python3.10 -m venv .venv-paper',
      '.venv-paper/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-author.txt',
      'NUMBA_CACHE_DIR=/tmp/eegis-numba-paper MPLCONFIGDIR=/tmp/eegis-matplotlib-paper .venv-paper/bin/python -u -m eegis.author_reproduction --generations 100 --population 100 --device cpu --knn-backend legacy-torch > author_reproduction.log 2>&1',
      'NUMBA_CACHE_DIR=/tmp/eegis-numba-paper MPLCONFIGDIR=/tmp/eegis-matplotlib-paper .venv-paper/bin/python -u -m eegis.author_window_scaling --generations 100 --population 100 > author_window_scaling.log 2>&1',
      'NUMBA_CACHE_DIR=/tmp/eegis-numba-paper .venv-paper/bin/python -m eegis.verify_author_reproduction',
      '.venv-paper/bin/python -m eegis.report_author_reproduction',
      '.venv-paper/bin/python -m unittest discover -s tests -p test_author_reproduction.py -v',
      '.venv-paper/bin/python -m eegis.predict_author --window class_8/0.csv',
      '```', '',
      'Use the isolated environment for the new checkpoints. '
      'The best_model.joblib accepts a complete six-layer example through predict_layers; '
      'predict_author loads matching supplied CSVs automatically. '
      'The original raw-only predictor is not used for this checkpoint. '
      'Torch kNN retains CPU/CUDA/auto support, but the measured environment is CPU-only. '
      'No GPU result is claimed.', '',
      '![Grouped genetic search](training_graph.png)', '',
      '![Grouped test confusion matrix](knn_confusion.png)', '',
      '![Within-example scaling search](within_window_grouped/training_graph.png)', '',
      '![Window-split diagnostic search](paper_window_diagnostic/training_graph.png)', '',
      'Checkpoints, feature caches and complete population states stay local. '
      'Manifests, plans, masks, fitness tables, predictions, scores, plots and verification are shareable.', '']
    (out/'README.md').write_text('\n'.join(text))
    root=Path('README.md'); marker='\n## Author-source 96% reproduction attempt\n'
    section=marker+'\nRecovered the author\'s Signalytica functions and exact NumPy/SciPy pins in an isolated Python 3.10 environment. '
    section+=f'Ran three {plan["generations"]}-generation, {plan["population"]}-genome searches and eight fixed published-mask checks. '
    section+='The previous 50-generation implementation is retained. See [the full reproduction report](artifacts_author_reproduction/README.md).\n\n'
    section+='| Protocol | Fitness accuracy | Test accuracy | Test macro-F1 | Majority test accuracy |\n|---|---:|---:|---:|---:|\n'
    for r in rows:
        section+=f'| {r["protocol"]} | {r["fitness_accuracy"]:.2%} | {r["test_accuracy"]:.2%} | {r["test_macro_f1"]:.4f} | {r["majority_accuracy"]:.2%} |\n'
    section+='\n'+('The numeric grouped target was reached.' if reached else 'The paper\'s 96% was not reproduced on the grouped test.')
    section+=' Window-split scores are leakage diagnostics; fitness is not independent test accuracy. '
    section+='Original test reuse and missing subject IDs remain limitations. Unpublished author settings prevent a complete 1:1 claim.\n'
    original=root.read_text()
    if marker in original:
        before,after=original.split(marker,1);next_section=after.find('\n## ')
        root.write_text(before+section+(after[next_section:] if next_section>=0 else ''))
    else:
        root.write_text(original+'\n'+section)
    print('Wrote all measured cases and target status')


if __name__=='__main__':
    main()
