"""Plot and report independently verified full kNN experiments."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def report(root):
    verification=json.loads((root/'verification.json').read_text())
    assert all(item['all_final_metrics_and_confusions_recomputed'] for item in verification['modes'])
    plan=json.loads((root/'study_plan.json').read_text())
    table=pd.read_csv(root/'comparison.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig,ax=plt.subplots(figsize=(9,5))
    for mode in plan['modes']:
        history=pd.read_csv(root/mode/'history.csv')
        ax.plot(history.generation,history.best_validation_accuracy,label=mode+' best fitness')
        ax.plot(history.generation,history.mean_survivor_accuracy,linestyle='--',alpha=.5,label=mode+' survivor mean')
        fig2,ax2=plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay(np.loadtxt(root/mode/'confusion.csv',delimiter=',',dtype=int),display_labels=['No','Sí']).plot(ax=ax2,colorbar=False)
        ax2.set_title(mode+(' diagnostic' if mode=='paper' else ' grouped exploratory test'))
        fig2.tight_layout();fig2.savefig(root/mode/'confusion.png',dpi=140);plt.close(fig2)
    ax.set(xlabel='GA generation (0=initialization)',ylabel='Adaptive inner/search fitness',ylim=(0,1))
    ax.legend();fig.tight_layout();fig.savefig(root/'training_graph.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,row in zip(axes,table.to_dict('records')):
        ax.bar(['kNN','Majority'],[row['accuracy'],row['majority_accuracy']])
        ax.set(ylim=(0,1),ylabel='Accuracy',title=row['mode']+(' diagnostic' if row['diagnostic_only'] else ' exploratory holdout'))
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=140);plt.close(fig)
    lines=['# Full kNN searches: paper-style and grouped evaluation','',
        f'Each search used population **{plan["population"]}** and **{plan["generations"]} generations**, '
        'plus initialization. All values were measured and independently verified.','',
        '| Protocol | Search fitness | Final accuracy | Macro-F1 | Majority accuracy | k | Evaluation windows |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in table.to_dict('records'):
        lines.append(f'| {row["mode"]} | {row["fitness_accuracy"]:.2%} | {row["accuracy"]:.2%} | '
            f'{row["macro_f1"]:.4f} | {row["majority_accuracy"]:.2%} | {row["k"]} | {row["evaluation_windows"]} |')
    lines+=['','The paper-style result is a **diagnostic**, because EEG overlap and reuse of selection labels '
        'prevent independent validation. The grouped result keeps entire overlap components together and excludes '
        'the original180-window holdout from its entire genetic search. That holdout was reported in earlier work, '
        'so this remains **exploratory**. Participant identifiers are unavailable; neither result establishes subject independence.','',
        'Both searches and their settings were declared before training. Both masks were frozen before any final '
        'predictions. The paper experiment uses the later184/740 split and k=9; the grouped experiment uses '
        '600fit/144fitness inside744training windows and the frozen180-window test, with k selected on inner fitness. '
        'Seeds, normalization and feature settings were not chosen using final accuracy.','',
        'The feature bank is unchanged from the printed-equation functional run. It includes the disclosed conventional '
        'Higuchi alternative and within-example handling of131 undefined raw Katz values. Exact author settings remain '
        'unavailable, and no unqualified numerical1:1 claim is made.','',
        'The optimization only avoids computing medians for nonmissing features. Transforms were checked bit for bit '
        'against the reference implementation; actual sklearn prediction is still called independently for every k. '
        'The GA follows the same population/RNG transitions and writes an atomic checkpoint after every generation.','',
        'The primary saved checkpoint is the predeclared grouped model. The paper diagnostic score never determines '
        'which checkpoint is saved as the primary model.','',
        '## Replay','',
        '```bash',
        'NUMBA_CACHE_DIR=/tmp/eegis-numba-paper MPLCONFIGDIR=/tmp/eegis-matplotlib-paper OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-paper/bin/python -u -m eegis.train_knn100',
        'NUMBA_CACHE_DIR=/tmp/eegis-numba-paper OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-paper/bin/python -m eegis.verify_knn100',
        '.venv-paper/bin/python -m eegis.report_knn100',
        '```','',
        'The training command resumes existing generation checkpoints only when code/data/configuration hashes match. '
        'Use a different --out for a different experiment. Only kNN received full searches here; the other nine '
        'classifiers retain the earlier functional-test results.','',
        '![GA histories](training_graph.png)','',
        '![Separate protocol comparisons](comparison.png)','',
        '![Grouped confusion](grouped/confusion.png)','',
        '![Paper diagnostic confusion](paper/confusion.png)','']
    (root/'README.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=Path('artifacts_paper_method/knn100'))
    args=parser.parse_args();report(args.artifacts)


if __name__=='__main__':
    main()
