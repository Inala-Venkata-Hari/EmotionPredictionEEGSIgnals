"""Write a measured report only after the paper-protocol verifier succeeds."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .train_paper_binary import dump


def report(out):
    verified=json.loads((out/'verification.json').read_text())
    if not verified['all_accuracy_macro_f1_confusions_recomputed']:
        raise ValueError('Verify all measured scores before reporting them')
    plan=json.loads((out/'plan.json').read_text())
    results=json.loads((out/'results.json').read_text())
    audit=json.loads((out/'feature_audit.json').read_text())
    literal=json.loads((out/'literal_higuchi_audit.json').read_text())
    comparison=pd.read_csv(out/'comparison.csv')
    dump(out/'best_model.json',{'model':results['selected_model'],'file':'best_model.joblib',
        'selection':'highest stage1 GA fitness; never stage2 accuracy',
        'input':'original raw windows, shape (N,128,14), no exported CSV index',
        'class_ids':[6,8],'class_names':['No','Sí'],'diagnostic_only':True,
        'feature_config':plan['feature_config'],'undefined_policy':plan['undefined_policy'],
        'full_100_generation_study':results['full_100_generation_study']})
    smoke=not results['full_100_generation_study']
    lines=['# Printed-paper feature and two-stage pipeline','',
        ('This is a **small functional test**, not the full published experiment.' if smoke else
         'This run uses the published population/generation counts with all ten classifiers.'),'',
        f'The run uses {plan["generations"]} GA generations plus initialization, '
        f'{plan["population"]} genomes, and {len(plan["classifiers"])} classifiers. '
        'All numbers below were measured and independently verified.','',
        '**Every reported score is diagnostic.** Random window splits share EEG samples, '
        'and the later evaluation reuses examples whose labels can already have influenced GA selection. '
        'These are not independent decoder accuracy estimates.','',
        '| Classifier | Stage1 GA fitness | Later 20/80 accuracy | Macro-F1 |',
        '|---|---:|---:|---:|']
    for row in comparison.to_dict('records'):
        lines.append(f'| {row["classifier"]} | {row["stage1_fitness"]:.2%} | '
                     f'{row["stage2_accuracy"]:.2%} | {row["stage2_macro_f1"]:.4f} |')
    lines+=['',f'Majority reference on the later split: **{results["majority"]["accuracy"]:.2%}**. '
        f'The saved model is **{results["selected_model"]}**, selected using stage1 fitness alone. '
        'No model was selected using the later accuracy.','',
        '## Method now implemented','',
        '- Only the original 480 No and 444 Sí one-second, 14-channel, 128 Hz examples are used.',
        '- Raw EEG is filtered once into the five fourth-order Butterworth bands.',
        '- The 14×214 bank follows printed CV, median-of-halves quartiles, Hjorth activity, Petrosian and Katz formulas.',
        '- Relative band power uses the original raw spectrum as context. Band amplitude uses the already filtered signal and its time derivative, without a second band filter.',
        '- Each classifier has its own 228-bit electrode/feature GA. All ten sklearn estimators retain their default parameters; kNN searches all seven published k values independently.',
        '- Stage1 fits on739 windows and uses185 for GA fitness. There is no additional inner split.',
        '- Every selected genome is frozen before stage2. The later experiment fits on184 windows and evaluates740; kNN uses k=9 as stated for that stage in Section3.',
        '- Standard accuracy, macro-F1, per-class metrics, confusion matrices, convergence graphs and checkpoints are saved. The paper’s printed precision formula is recorded separately from standard precision.','',
        '## Limits of literal reproduction','',
        f'Literal Higuchi equations A22–A24 were undefined in **{literal["undefined_cases"]} of '
        f'{len(literal["cases"])} declared real EEG checks**, because signed curve lengths are nonpositive before taking logarithms. '
        'The strict-literal mode records this and stops without training. '
        f'This runnable experiment explicitly selects **{plan["feature_config"]["higuchi"]} Higuchi**.','',
        f'There are **{audit["undefined_values"]} undefined values** in the base feature bank, '
        f'with per-column counts in undefined_features.csv. The selected policy is **{plan["undefined_policy"]}**. '
        'Within-window imputation uses only electrodes from that example; a completely missing feature column receives zero. '
        'The optional train-population mode fits imputation and scaling only on fitting examples.','',
        'The paper does not provide the exact seed, split indices, GA probabilities, normalization axis, '
        'complete spectral/recurrence configuration or missing-value policy. '
        'These choices are recorded in plan.json and feature_spec.json. '
        'This is an executable reconstruction of the stated workflow; it is not proof of a numerical 1:1 replica of the unpublished author program.','',
        '## Reproduce','',
        'Use the existing pinned Python 3.10 environment (.venv-paper). No new dependencies are required.','',
        '```bash',
        'NUMBA_CACHE_DIR=/tmp/eegis-numba-paper MPLCONFIGDIR=/tmp/eegis-matplotlib-paper OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-paper/bin/python -u -m eegis.paper_protocol --generations 2 --population 4 --out artifacts_paper_method/smoke_replay',
        'NUMBA_CACHE_DIR=/tmp/eegis-numba-paper OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-paper/bin/python -m eegis.verify_paper_protocol --artifacts artifacts_paper_method/smoke_replay',
        '.venv-paper/bin/python -m eegis.report_paper_protocol --artifacts artifacts_paper_method/smoke_replay',
        '```','',
        'The default command below runs all ten classifiers with population100 and100 generations '
        'in a separate output directory. It can take substantially longer than the functional test.','',
        '```bash',
        'NUMBA_CACHE_DIR=/tmp/eegis-numba-paper MPLCONFIGDIR=/tmp/eegis-matplotlib-paper OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-paper/bin/python -u -m eegis.paper_protocol',
        '```','',
        'Use --resume with the same code and configuration to resume saved classifier searches. '
        'The full 100-generation run has not been executed unless results.json states full_100_generation_study=true.','',
        'Strict literal check:','',
        '```bash',
        'NUMBA_CACHE_DIR=/tmp/eegis-numba-paper .venv-paper/bin/python -m eegis.paper_protocol --strict-literal --prepare-only --out artifacts_paper_method/literal_check',
        '```','',
        'This command is expected to stop with the mathematical-domain diagnostic; it does not silently replace Higuchi.','',
        'Single raw-window prediction:','',
        '```bash',
        '.venv-paper/bin/python -m eegis.predict_paper_protocol "EEGIS - Electroencephalogram Imagined Speech dataset/raw/class_8/0.csv"',
        '```','',
        'The reference runner uses CPU sklearn defaults. Earlier CUDA-capable alternatives remain available; '
        'no GPU execution or exact CUDA tie parity is claimed here.','',
        '![GA fitness](training_graph.png)','',
        '![Later-stage comparison](comparison.png)','',
        'Feature caches, checkpoints and population arrays stay local. '
        'Source, plans, manifests, measured scores, predictions, graphs and verification receipts can be shared.','']
    if (out/'checkpoint_migration.json').exists():
        lines+=['## Checkpoint provenance','',
            'A CLI serialization reference was repaired after training. Original code snapshots, '
            'the original plan and original checkpoints are retained. All numerical code outside '
            'the CLI guard is AST-identical, and every prediction was unchanged before/after repair. '
            'The verifier checks the documented migration as well as independently refitting all models. '
            'See checkpoint_migration.json. Future CLI checkpoints use a canonical importable module.','']
    (out/'README.md').write_text('\n'.join(lines))
    parent=out.parent
    if parent.resolve()==Path('artifacts_paper_method').resolve():
        (parent/'README.md').write_text('# Printed-paper method implementation\n\n'
            'The new pipeline implements the printed feature profile and the two-stage, ten-classifier workflow. '
            'Earlier experiments are preserved. See [the verified run report]('+out.name+'/README.md) '
            'for measured results and assumptions. Literal Higuchi and unspecified author settings prevent '
            'an unqualified numerical 1:1 claim.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=Path('artifacts_paper_method/smoke'))
    args=parser.parse_args()
    report(args.artifacts)


if __name__=='__main__':
    main()
