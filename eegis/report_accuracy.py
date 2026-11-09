"""Publish measured train-only nested results after independent verification."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def report(root):
    root=Path(root);verification=json.loads((root/'verification.json').read_text())
    assert all(r['all_outer_predictions_and_metrics_match'] for r in verification['units'])
    plan=json.loads((root/'plan.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    rows=[]
    for unit in ['window','recording']:
        directory=root/unit;result=json.loads((directory/'nested_results.json').read_text())
        interval=json.loads((directory/'uncertainty.json').read_text())
        selected=json.loads((directory/'best_model.json').read_text())
        row={'unit':unit,'duration_seconds':result['duration_seconds'],**result['nested'],
             'majority_balanced_accuracy':result['majority']['balanced_accuracy'],
             'balanced_accuracy_ci95_low':interval['balanced_accuracy_ci95'][0],
             'balanced_accuracy_ci95_high':interval['balanced_accuracy_ci95'][1],
             'training_groups':result['independent_overlap_groups'],'final_selected_family':selected['winner']['name']}
        row['prototype_gate_met']=bool(row['balanced_accuracy']>=.7 and row['macro_f1']>=.7 and row['balanced_accuracy_ci95_low']>.5)
        rows.append(row)
        fig,ax=plt.subplots(figsize=(5,4))
        cm=np.loadtxt(directory/'nested_confusion.csv',delimiter=',',dtype=int)
        ConfusionMatrixDisplay(cm,display_labels=['No','Sí']).plot(ax=ax,colorbar=False)
        ax.set_title(unit+': nested training-group predictions');fig.tight_layout();fig.savefig(directory/'nested_confusion.png',dpi=140);plt.close(fig)
        final=json.loads((directory/'final_search/selection.json').read_text())
        (directory/'final_selection.json').write_text(json.dumps(final,indent=2)+'\n')
        candidate_tables=[]
        for path in sorted(directory.glob('outer_*/*_cv.csv'))+sorted((directory/'final_search').glob('*_cv.csv')):
            frame=pd.read_csv(path);frame.insert(0,'family',path.name[:-7]);frame.insert(0,'scope',path.parent.name)
            candidate_tables.append(frame)
        pd.concat(candidate_tables,ignore_index=True).to_csv(directory/'inner_cv_results.csv',index=False)
        table=pd.DataFrame(final['families'])
        fig,ax=plt.subplots(figsize=(max(7,len(table)*.42),4))
        ax.bar(table.name,table.inner_balanced_accuracy);ax.axhline(.5,color='black',linestyle='--')
        ax.set(ylim=(0,1),ylabel='Inner selection balanced accuracy',title=unit+' final training-only search')
        ax.tick_params(axis='x',rotation=70);fig.tight_layout();fig.savefig(directory/'inner_search.png',dpi=140);plt.close(fig)
        histories=list(directory.glob('outer_*/cnn_training_history.csv'))
        if histories:
            fig,axes=plt.subplots(1,2,figsize=(9,4))
            all_histories=[]
            for path in histories:
                h=pd.read_csv(path);axes[0].plot(h.epoch,h.training_loss,label=path.parent.name)
                axes[1].plot(h.epoch,h.training_accuracy,label=path.parent.name)
                h.insert(0,'scope',path.parent.name);all_histories.append(h)
            pd.concat(all_histories,ignore_index=True).to_csv(directory/'cnn_training_histories.csv',index=False)
            axes[0].set(ylabel='Training loss');axes[1].set(ylabel='Training accuracy',ylim=(0,1))
            for ax in axes:ax.set_xlabel('Epoch');ax.legend()
            fig.suptitle(unit+': CNN candidate refits; training-only curves')
            fig.tight_layout();fig.savefig(directory/'cnn_training.png',dpi=140);plt.close(fig)
    pd.DataFrame(rows).to_csv(root/'comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar([r['unit'] for r in rows],[r['balanced_accuracy'] for r in rows])
    ax.axhline(.5,color='black',linestyle='--',label='Chance reference')
    ax.axhline(.7,color='green',linestyle=':',label='Prototype target')
    ax.set(ylim=(0,1),ylabel='Nested balanced accuracy',title='Original training groups only')
    ax.legend();fig.tight_layout();fig.savefig(root/'comparison.png',dpi=140);plt.close(fig)
    lines=['# Verified binary accuracy research','',
        f'This batch compares {len(plan["candidates"])} model families with '
        f'{sum(r["configurations"] for r in plan["candidates"])} hyperparameter configurations per inner search. '
        'Only the original training partition is used. All results below were measured and independently verified.','',
        '| Decision | Accuracy | Balanced accuracy | Macro-F1 | Approximate balanced-accuracy 95% interval |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["duration_seconds"]:g} seconds | {r["accuracy"]:.2%} | {r["balanced_accuracy"]:.2%} | '
            f'{r["macro_f1"]:.4f} | {r["balanced_accuracy_ci95_low"]:.2%}–{r["balanced_accuracy_ci95_high"]:.2%} |')
    lines+=['','The initial prototype gate requires at least 70% balanced accuracy and 0.70 macro-F1, '
        'with the approximate lower interval above chance. '+('A provisional gate is met; independent confirmation is still required.'
        if any(r['prototype_gate_met'] for r in rows) else '**This batch does not meet that gate.**'),'',
        'These scores estimate the complete inner-selection procedure, not the performance of one fixed classifier. '
        'Five outer folds keep complete overlap groups separate. Three inner folds select the representation, '
        'feature selector, scalers and classifier using only each outer fitting partition. The final exported model '
        'is selected by a separate three-fold search over all original training groups, without using outer scores '
        'to pick its parameters. The original validation and test partitions were not evaluated by this batch.','',
        'There are only **47 original training overlap components**: 564 correlated one-second windows, or '
        '47 reconstructed components of 1,008 samples each. The two decision durations differ and cannot be '
        'reported as interchangeable accuracy measurements. Chance balanced accuracy is 50% for two classes.','',
        'The saved field independent_overlap_groups counts components with disjoint observed samples. It does '
        'not certify independence of participants, sessions or complete experimental trials.','',
        'Every EEGIS example appeared in earlier project analyses. These follow-ups are exploratory, '
        'including repeated use of training folds while developing later batches. No verified participant IDs '
        'exist in the release, and no subject-independent or deployment-quality claim is made. The paired '
        'overlap-group bootstrap describes variation in saved out-of-fold predictions; it does not account '
        'for every adaptive research choice or all dependence caused by overlapping CV training sets.','',
        'Verification independently refits five selected outer models and 15 selected inner models per decision '
        'duration, checks exact predictions and all reported metrics, audits all group boundaries, and verifies '
        'raw versus cached inference and the final exported model. All stored candidate-score tables are checked '
        'for selection consistency; every candidate fit is not independently repeated.','',
        'Final models and feature caches stay local. Recreate them with the matching training module '
        f'and run `python -m eegis.verify_accuracy --artifacts {root}` before reporting results. The frozen plan '
        'records source/data/runtime hashes and every candidate.','',
        '![Nested comparison](comparison.png)','',
        '![One-second confusion](window/nested_confusion.png)','',
        '![Longer-context confusion](recording/nested_confusion.png)','']
    if (root/'vendor_provenance.json').exists():
        lines+=['The MiniRocket numerical helpers are adapted from the BSD-licensed '
            '[Aeon implementation](https://github.com/aeon-toolkit/aeon), with unchanged helper bodies and full '
            'copyright/license terms retained in eegis/aeon_minirocket.py. Source commit and function hashes '
            'are recorded in vendor_provenance.json. Its learned thresholds are fitted inside the current '
            'training fold, unlike the separate earlier fixed-random-kernel implementation.','']
    elif 'kernels_per_representation' in plan:
        lines+=['The random convolution bank is an independent '
            '[ROCKET-inspired](https://arxiv.org/abs/1910.13051) implementation with fixed, label-free thresholds; '
            'it is not the reference MiniRocket algorithm.','']
    (root/'README.md').write_text('\n'.join(lines))
    (root/'prototype_assessment.json').write_text(json.dumps({'target_balanced_accuracy':.7,'target_macro_f1':.7,
        'results':rows,'independent_confirmation_available':False},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',default='artifacts_accuracy')
    report(p.parse_args().artifacts)


if __name__=='__main__':main()
