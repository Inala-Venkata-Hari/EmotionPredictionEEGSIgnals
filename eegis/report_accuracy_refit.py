"""Report all accuracy research and the separately frozen final refit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate_accuracy import STUDIES


def main():
    root=Path('artifacts_accuracy_refit')
    assert (root/'verification.json').exists()
    results=json.loads((root/'results.json').read_text())['models']
    plan=json.loads((root/'selection_plan.json').read_text())
    nested=[];configurations=0
    for study,_ in STUDIES:
        base=Path(study);study_plan=json.loads((base/'plan.json').read_text())
        configurations+=sum(r['configurations'] for r in study_plan['candidates'])
        for unit in ['window','recording']:
            r=json.loads((base/unit/'nested_results.json').read_text())
            nested.append({'study':study,'unit':unit,**r['nested']})
    nested=pd.DataFrame(nested);nested.to_csv(root/'nested_comparison.csv',index=False)
    rows=[{'unit':r['unit'],'duration_seconds':r['duration_seconds'],'source_study':r['source_study'],
        'family':r['family'],**r['metrics'],'majority_accuracy':r['majority']['accuracy'],
        'majority_balanced_accuracy':r['majority']['balanced_accuracy'],
        'balanced_accuracy_ci95_low':r['uncertainty']['balanced_accuracy_ci95'][0],
        'balanced_accuracy_ci95_high':r['uncertainty']['balanced_accuracy_ci95'][1]} for r in results]
    pd.DataFrame(rows).to_csv(root/'comparison.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    for row in rows:
        directory=root/row['unit'];fig,ax=plt.subplots(figsize=(5,4))
        cm=np.loadtxt(directory/'confusion.csv',delimiter=',',dtype=int)
        ConfusionMatrixDisplay(cm,display_labels=['No','Sí']).plot(ax=ax,colorbar=False)
        ax.set_title(row['unit']+': reused exploratory test');fig.tight_layout();fig.savefig(directory/'confusion.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    names=[Path(s).name.replace('artifacts_','') for s,_ in STUDIES]
    for offset,unit in [(-.18,'window'),(.18,'recording')]:
        scores=nested.set_index(['study','unit']).loc[[(s,unit) for s,_ in STUDIES],'balanced_accuracy']
        ax.bar(np.arange(len(names))+offset,scores,width=.36,label=unit)
    ax.axhline(.5,color='black',linestyle='--',label='Chance reference')
    ax.axhline(.7,color='green',linestyle=':',label='Prototype target')
    ax.set(xticks=np.arange(len(names)),xticklabels=names,ylim=(0,1),ylabel='Nested balanced accuracy',title='Original training groups only')
    ax.tick_params(axis='x',rotation=20);ax.legend();fig.tight_layout();fig.savefig(root/'nested_comparison.png',dpi=140);plt.close(fig)
    best=float(nested.balanced_accuracy.max())
    lines=['# Accuracy-improvement research: target not reached','',
        f'Six exploratory batches tested **{configurations} hyperparameter configurations** with '
        'training-only nested grouping. Methods included random temporal convolutions, covariance/tangent models, '
        'CSP, spectral/spatial contrasts, phase connectivity, CNNs, latent alignment, reference MiniRocket, '
        'frozen LaBraM and last-block adaptation. No batch met the declared prototype gate.','',
        f'The best observed nested balanced accuracy was **{best:.2%}**, below the 70% target. '
        'These are repeated exploratory comparisons on the same training groups; this maximum is not an '
        'independent estimate of a cross-study selection procedure.','',
        '## Separately frozen final refit','',
        'For each input duration, the candidate with the highest original-training inner-CV balanced accuracy '
        'was selected, with macro-F1 and declared order breaking ties. The complete selection was saved before '
        'any new test prediction. Models were then refitted on the original training plus validation partitions '
        '(62 overlap components) and evaluated once on the previously reported test set (15 components). '
        'All training-only study checkpoints are preserved.','',
        '| Input duration | Test accuracy | Balanced accuracy | Macro-F1 | Majority accuracy | Approximate balanced-accuracy interval |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["duration_seconds"]:g} seconds | {row["accuracy"]:.2%} | {row["balanced_accuracy"]:.2%} | '
            f'{row["macro_f1"]:.4f} | {row["majority_accuracy"]:.2%} | '
            f'{row["balanced_accuracy_ci95_low"]:.2%}–{row["balanced_accuracy_ci95_high"]:.2%} |')
    lines+=['','The majority balanced-accuracy reference is 50%. **These are not passable decoder results.** '
        'The primary saved model is the predeclared one-second model, regardless of which duration scored higher. '
        'No test score selected its source method, parameters, seed, normalization or input duration.','',
        'The one-second winner is `'+plan['winners']['window']['name']+'` from '+plan['winners']['window']['study']+
        '; the longer-context winner is `'+plan['winners']['recording']['name']+'` from '+plan['winners']['recording']['study']+'.','',
        '## What the data support','',
        'The release reconstructs into only 77 No/Sí overlap components, with 47 in the original training partition. '
        'These are not verified complete trials or participants. Window counts include correlated observations. '
        'The release has no verified participant/session identifiers, so subject-independent accuracy cannot be measured. '
        'Every example appeared in earlier research, and all follow-up scores remain exploratory.','',
        'A separate training-only diagnostic tested spectral, spectral-shape, topographic, phase and raw-offset '
        'representations with 999 class-label permutations. None showed significant kernel class separation '
        'after Holm correction. This limited diagnostic does not prove that the data lack information or that '
        'labels are incorrect. Synthetic controls confirm the algorithms can learn clean frequency/spatial '
        'patterns, but those controls are not EEGIS performance measurements.','',
        'The useful next input is more labeled recordings and verified participant/trial metadata, followed by '
        'a genuinely untouched evaluation. Further trials on these same development splits cannot establish '
        'a reliable 70% decoder. No claimed success or fabricated metric replaces the negative results.','',
        '## Verification and reuse','',
        'Each research batch independently refits selected inner and outer models, checks group boundaries and '
        'recomputes all saved predictions and metrics. The final refit has a second independent fit for each '
        'duration and exact checkpoint/raw-inference checks. Earlier measurement files remain unchanged. '
        'Approximate intervals resample complete overlap components; they do not certify participant independence '
        'or account for all adaptive research choices.','',
        'The final model files remain local under window/best_model.joblib and recording/best_model.joblib. '
        'The root best_model.joblib is the predeclared one-second model. To classify one original raw CSV:', '',
        '```bash',
        '.venv/bin/python -m eegis.predict_accuracy "EEGIS - Electroencephalogram Imagined Speech dataset/raw/class_6/0.csv"',
        '```','',
        'Use --unit recording with twelve overlapping raw CSVs to classify a reconstructed component. '
        'This is a longer-context decision, not a one-second score.','',
        'Full study reports: '+', '.join('['+s+'](../'+s+'/README.md)' for s,_ in STUDIES)+'.','',
        '![All nested comparisons](nested_comparison.png)','',
        '![One-second final confusion](window/confusion.png)','',
        '![Longer-context final confusion](recording/confusion.png)','']
    (root/'README.md').write_text('\n'.join(lines))
    (root/'prototype_assessment.json').write_text(json.dumps({'target_balanced_accuracy':.7,'target_macro_f1':.7,
        'target_reached':False,'best_observed_nested_balanced_accuracy':best,'configurations':configurations,
        'all_results_exploratory':True,'independent_confirmation_available':False},indent=2)+'\n')


if __name__=='__main__':main()
