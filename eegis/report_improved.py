"""Append measured follow-up results to the project README."""
import json
from pathlib import Path
import pandas as pd

def main():
    out=Path('artifacts_improved');r=json.loads((out/'results.json').read_text());old=json.loads(Path('artifacts/results.json').read_text());paired=json.loads((out/'paired_comparison.json').read_text())
    selected=r['best_model'];s=r['models'][selected]
    previous=max(old['models'],key=lambda n:(old['models'][n]['validation']['accuracy'],old['models'][n]['validation']['macro_f1']))
    prior=old['models'][previous]
    use_new=(s['validation']['accuracy'],s['validation']['macro_f1'])>(prior['validation']['accuracy'],prior['validation']['macro_f1'])
    recommendation={'model':selected if use_new else previous,'artifacts':'artifacts_improved' if use_new else 'artifacts',
        'criterion':'highest validation accuracy across original and new candidates; macro-F1 tie-break',
        'test_used_for_selection':False}
    (out/'recommendation.json').write_text(json.dumps(recommendation,indent=2))
    fits=sum(len(pd.read_csv(f)) for f in out.glob('*_cv.csv'))*3
    lines=['## Accuracy improvement experiment', '',
    'This follow-up preserves the original manifest exactly: 205 training, 66 validation, and 66 test overlap groups. Participant identities remain unavailable. The previously reported test set is reused, so these are exploratory follow-up results and require an untouched dataset for confirmation. No split seed was changed to obtain higher accuracy.', '',
    'New spatial features use linearly detrended signals and five channel covariance matrices: broadband and 4–8, 8–13, 13–30, and 30–40 Hz. Each covariance matrix receives 10% shrinkage toward an isotropic matrix with matching trace. The training partition determines the covariance-whitening reference; symmetric matrix logarithms and upper-triangle vectorization yield 525 features. Trace normalization is tested both enabled and disabled. Log-covariance SVM omits reference whitening. Every fitted transform and scaler stays inside the cross-validation pipeline.', '',
    f'The search performed {fits} fits across three stratified group folds inside the training partition. Each family first selects its hyperparameters by training cross-validation accuracy. The final candidate is selected by validation accuracy, with macro-F1 breaking ties, matching the request to improve accuracy. Candidates include tangent-space RBF SVM, ridge classification, tangent logistic regression, log-covariance RBF SVM, Extra Trees, and histogram gradient boosting. Tree candidates use the original 238 features. See the saved `*_cv.csv` files for all configurations and fold scores.', '',
    '| New candidate | Training CV accuracy | Validation accuracy | Test accuracy | Test macro-F1 |',
    '|---|---:|---:|---:|---:|']
    for name,v in r['models'].items():
        lines.append(f"| {name} | {v['training_cv_accuracy']:.2%} | {v['validation']['accuracy']:.2%} | {v['test']['accuracy']:.2%} | {v['test']['macro_f1']:.4f} |")
    lo,hi=paired['paired_group_bootstrap_95_percent_interval_pp']
    lines += ['',f"The new validation-selected candidate is **{selected}**, with {s['validation']['accuracy']:.2%} validation accuracy and {s['test']['accuracy']:.2%} test accuracy (macro-F1 {s['test']['macro_f1']:.4f}). Its checkpoint is `artifacts_improved/best_model.joblib`. The original validation-selected Random Forest scored 8.08% on the same test windows; the new candidate's measured difference is {paired['accuracy_difference_percentage_points']:+.2f} percentage points. The 95% paired overlap-group bootstrap interval for this difference is [{lo:+.2f}, {hi:+.2f}] points (5,000 resamples, 66 groups). This interval does not establish subject-independent improvement.", '',
    'For context, the original EEGNet scored 11.24% test accuracy, and the training-majority predictor scored 12.12%. Compare against these as well as the original selected Random Forest. Low scores remain insufficient for a useful speech decoder; a small gain on a reused holdout should not be treated as a validated deployment result.', '',
    f"Across the original and new one-second models, validation accuracy recommends **{recommendation['model']}** from `{recommendation['artifacts']}`. This recommendation does not use test scores. The selected new model parameters are `{json.dumps(s['parameters'])}`. The final resumed execution took {r['elapsed_seconds']:.1f} seconds on CPU; this excludes interrupted solver attempts and the earlier completed nonlinear SVM search. The original model files and results are preserved. Seven automated tests passed, including spatial-feature validity, inference independence, group aggregation, and amplitude invariance under trace normalization. `eegis.verify_improved` independently reloads all new models, recomputes predictions and metrics, checks the identical split, and verifies the selected-model inference path.", '',
    '![Follow-up model comparison](artifacts_improved/comparison.png)', '',
    'Reproduce and verify this experiment:', '', '```bash',
    'MPLCONFIGDIR=/tmp/eegis-matplotlib .venv/bin/python -u -m eegis.improve > improvement.log 2>&1',
    'MPLCONFIGDIR=/tmp/eegis-matplotlib .venv/bin/python -m eegis.verify_improved',
    '.venv/bin/python -m eegis.compare_runs',
    '.venv/bin/python -m eegis.report_improved',
    '.venv/bin/python -m eegis.predict --artifacts artifacts_improved "EEGIS - Electroencephalogram Imagined Speech dataset/raw/class_0/0.csv"',
    '```', '',
    'All new results, candidate hyperparameters, cross-validation details, confusion matrices, test predictions, checkpoint metadata, leakage audit, and verification results are under `artifacts_improved/`. The original CNN training curves remain in `artifacts/`; no new neural training is claimed in this follow-up.']
    if (out/'context_results.json').exists():
        context=json.loads((out/'context_results.json').read_text());name=context['winner'];metrics=context['models'][name]
        lines += ['', '### Optional longer-context decisions', '',
            'A separate experiment averages class probabilities (or decision scores for models without probability output) across all 12 connected windows in a component. This uses 1,008 distinct raw samples, or 7.875 seconds, for one decision. Group membership comes from raw overlap matches; labels do not determine predictions. Entire components remain in their original partition. This is a different latency and evaluation unit from the one-second experiment, so its accuracy is not a direct one-second improvement.', '',
            f"The validation-selected context model is **{name}**: validation accuracy {metrics['validation']['accuracy']:.2%}; test accuracy **{metrics['test']['accuracy']:.2%}**, macro-F1 **{metrics['test']['macro_f1']:.4f}** across 66 test components. The checkpoint is `artifacts_improved/best_context_model.joblib`, with aggregation metadata in `context_selection.json`. `context_test_predictions.csv` saves one prediction per component. Single-chain inference was checked against the recorded predictions.", '',
            'Reproduce with `.venv/bin/python -m eegis.context`. To predict a new complete chain, run `.venv/bin/python -m eegis.context --predict <12 CSV paths>`; all windows must form one valid connected chain. The code rejects duplicates and incomplete or mixed chains. Re-run `eegis.report_improved` after this optional evaluation to refresh these measured results.']
    p=Path('README.md');text=p.read_text().split('\n## Accuracy improvement experiment')[0]
    text=text.replace('No extra tuning was performed after viewing test results.','No extra tuning was performed after viewing test results during this initial experiment; the separate follow-up below was requested afterward.')
    p.write_text(text+'\n\n'+'\n'.join(lines)+'\n')

if __name__=='__main__':main()
