"""Verify full kNN searches, checkpoints, grouped isolation, and final metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from .data import audit_shared_samples
from .diagnose_leakage import leakage_counts
from .paper_equations import EquationConfig, equation_features
from .paper_ga import CLASS_IDS, CLASS_NAMES, K_VALUES, genome_columns
from .paper_protocol import EquationTransform, score
from .train_knn100 import fingerprint, load_bank, partitions
from .train_paper_binary import dump
from .verify_paper_protocol import independent_transform


def metric_scores(labels,predicted):
    return {'accuracy':float(np.mean(labels==predicted)),
            'macro_f1':float(f1_score(labels,predicted,labels=CLASS_IDS,average='macro',zero_division=0))}


def reference_scores(bank,labels,genome,fit,validation):
    z=independent_transform(bank,genome,fit,'within-window')
    records=[]
    for k in K_VALUES:
        model=KNeighborsClassifier(n_neighbors=k).fit(z[fit],labels[fit])
        predicted=model.predict(z[validation])
        records.append({'k':k,**metric_scores(labels[validation],predicted)})
    return records


def verify(root,data,manifest):
    plan=json.loads((root/'study_plan.json').read_text())
    assert plan['source_code_sha256']==fingerprint(),'Training source changed'
    x,source,bank,config,audit=load_bank(plan['feature_run'],data,manifest)
    assert plan['feature_array_sha256']==audit['array_sha256']
    assert plan['raw_source_sha256']==audit['source_sha256']
    split=partitions(source,plan['seed'],plan['validation_seed'])
    for mode,item in split.items():
        assert {key:list(map(int,value)) for key,value in item.items()}==plan['partitions'][mode]
    study_sha=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    frozen=json.loads((root/'frozen_selections.json').read_text())
    assert frozen['study_sha256']==study_sha and frozen['frozen_before_final_predictions'] is True
    subset=np.array([0,479,480,923])
    np.testing.assert_allclose(equation_features(x[subset],config),bank[subset],rtol=1e-10,atol=1e-10,equal_nan=True)
    labels,groups=source.label.to_numpy(),source.overlap_group.to_numpy()
    comparison=pd.read_csv(root/'comparison.csv').set_index('mode')
    receipt=[]
    for mode,item in split.items():
        out=root/mode
        chosen=json.loads((out/'selection.json').read_text())
        assert chosen==frozen['selections'][mode]
        state=joblib.load(out/'search_state.joblib')
        assert state['generation']==plan['generations']
        assert state['config']['guard']=={'study_sha256':study_sha,'mode':mode}
        history=pd.read_csv(out/'history.csv')
        np.testing.assert_array_equal(history.generation,np.arange(plan['generations']+1))
        evaluated=pd.read_csv(out/'evaluations.csv',dtype={'genome':str})
        assert len(evaluated)==chosen['unique_evaluations']==len(state['cache'])
        assert evaluated.genome.nunique()==len(evaluated)
        lookup=dict(zip(evaluated.genome,evaluated.accuracy))
        with np.load(out/'populations.npz',allow_pickle=False) as file:
            populations,accuracies=file['genomes'],file['accuracies']
        assert populations.shape==(plan['generations']+1,plan['population'],228)
        for index,(population,values) in enumerate(zip(populations,accuracies)):
            np.testing.assert_allclose(values,[lookup[''.join(map(str,g))] for g in population],rtol=0,atol=1e-12)
            np.testing.assert_array_equal(population,state['states'][index]['genomes'])
            assert np.all(np.diff(values)<=0)
            assert abs(values[0]-history.best_validation_accuracy.iloc[index])<1e-12
            assert abs(values.mean()-history.mean_survivor_accuracy.iloc[index])<1e-12
        np.testing.assert_array_equal(populations[-1,0],chosen['genome'])
        for key,value in state['cache'].items():
            assert abs(lookup[''.join(map(str,key))]-value['accuracy'])<1e-12
        # Independent sklearn + per-window transform for the selected genome.
        records=reference_scores(bank,labels,chosen['genome'],item['fit'],item['fitness'])
        best=max(records,key=lambda row:row['accuracy'])
        assert best['k']==chosen['k'] and best['accuracy']==chosen['accuracy']
        assert abs(best['macro_f1']-chosen['macro_f1'])<1e-12
        for left,right in zip(records,chosen['all_k_scores']):
            assert left['k']==right['k'] and left['accuracy']==right['accuracy']
            assert abs(left['macro_f1']-right['macro_f1'])<1e-12
        # A declared sample checks historical fitness; this is not an assertion
        # that every candidate was independently re-evaluated.
        sample=np.unique(np.linspace(0,len(evaluated)-1,min(32,len(evaluated)),dtype=int))
        for index in sample:
            row=evaluated.iloc[index]
            genome=np.array(list(row.genome),dtype=np.uint8)
            reference=reference_scores(bank,labels,genome,item['fit'],item['fitness'])
            candidate=max(reference,key=lambda value:value['accuracy'])
            assert candidate['k']==int(row.k)
            assert abs(candidate['accuracy']-row.accuracy)<1e-12
            assert abs(candidate['macro_f1']-row.macro_f1)<1e-12
        saved=pd.read_csv(out/'predictions.csv')
        np.testing.assert_array_equal(saved.path,source.path.iloc[item['final_evaluation']])
        np.testing.assert_array_equal(saved.label,labels[item['final_evaluation']])
        stored=joblib.load(out/'model.joblib')
        np.testing.assert_array_equal(stored.genome,chosen['genome'])
        z=independent_transform(bank,chosen['genome'],item['final_fit'],'within-window')
        k=9 if mode=='paper' else chosen['k']
        model=KNeighborsClassifier(n_neighbors=k).fit(z[item['final_fit']],labels[item['final_fit']])
        # Separate per-example scaler calls can change the last few floating
        # point bits for large-offset features. Require numerical agreement,
        # exact prediction parity, and bitwise equality to the original batch
        # transform whose operations this optimization preserves.
        np.testing.assert_allclose(stored.estimator._fit_X,model._fit_X,rtol=1e-10,atol=1e-10)
        selected_flat=bank.reshape(len(bank),-1)[:,genome_columns(chosen['genome'])]
        original=EquationTransform(sum(chosen['genome'][:14]),sum(chosen['genome'][14:])).fit(selected_flat[item['final_fit']])
        original_z=original.transform(selected_flat)
        np.testing.assert_array_equal(stored.estimator._fit_X,original_z[item['final_fit']])
        independent_transform_error=float(np.max(np.abs(stored.estimator._fit_X-model._fit_X)))
        predicted=model.predict(z[item['final_evaluation']])
        np.testing.assert_array_equal(predicted,saved.prediction)
        np.testing.assert_array_equal(stored.predict_features(bank[item['final_evaluation']]),predicted)
        np.testing.assert_array_equal(stored.predict(x[item['final_evaluation'][:1]]),predicted[:1])
        result=json.loads((out/'results.json').read_text())
        for key,value in metric_scores(labels[item['final_evaluation']],predicted).items():
            assert abs(result['final'][key]-value)<1e-12
            column='accuracy' if key=='accuracy' else 'macro_f1'
            assert abs(comparison.loc[mode,column]-value)<1e-12
        cm=confusion_matrix(labels[item['final_evaluation']],predicted,labels=CLASS_IDS)
        np.testing.assert_array_equal(cm,np.loadtxt(out/'confusion.csv',delimiter=',',dtype=int))
        detail=classification_report(labels[item['final_evaluation']],predicted,labels=CLASS_IDS,
            target_names=CLASS_NAMES,output_dict=True,zero_division=0)
        for label in CLASS_NAMES:
            for metric,value in detail[label].items():
                assert abs(result['report'][label][metric]-value)<1e-12
        majority=int(pd.Series(labels[item['final_fit']]).mode().iloc[0])
        np.testing.assert_array_equal(saved.majority,np.full(len(saved),majority))
        assert result['majority']==metric_scores(labels[item['final_evaluation']],saved.majority.to_numpy())
        overlap={'search':leakage_counts(x,groups,item['fit'],item['fitness']),
                 'final':leakage_counts(x,groups,item['final_fit'],item['final_evaluation'])}
        assert overlap==json.loads((out/'overlap_audit.json').read_text())
        if mode=='grouped':
            assert not any(value for counts in overlap.values() for value in counts.values())
            m=source.copy();m['split']='fit';m.loc[item['final_evaluation'],'split']='evaluation'
            audit_shared_samples(x,m)
            assert not np.intersect1d(item['search_rows'],item['final_evaluation']).size
        receipt.append({'mode':mode,'population_states_checked':len(history),
            'stored_candidate_score_records_consistency_checked':len(evaluated),
            'historical_genomes_independently_checked':len(sample),
            'selected_fitness_all_seven_k_independently_recomputed':True,
            'all_candidates_independently_recomputed':False,
            'optimized_training_values_bitwise_equal_to_original_batch_transform':True,
            'independent_per_example_transform_max_absolute_error':independent_transform_error,
            'raw_and_cached_inference_match':True,'final_model_independently_refitted':True,
            'all_final_metrics_and_confusions_recomputed':True,'overlap_audited':True})
        print(mode+': independent verification passed',flush=True)
    root_model=joblib.load(root/'best_model.joblib')
    np.testing.assert_array_equal(root_model.genome,frozen['selections']['grouped']['genome'])
    grouped_predictions=pd.read_csv(root/'grouped'/'predictions.csv').prediction.to_numpy()
    np.testing.assert_array_equal(root_model.predict_features(bank[split['grouped']['final_evaluation']]),
                                  grouped_predictions)
    np.testing.assert_array_equal(root_model.predict(x[split['grouped']['final_evaluation'][:1]]),
                                  grouped_predictions[:1])
    metadata=json.loads((root/'best_model.json').read_text())
    assert metadata['mode']=='grouped' and metadata['class_ids']==[6,8]
    assert metadata['exploratory'] is True and metadata['subject_independent'] is False
    dump(root/'verification.json',{'modes':receipt,'all_source_and_feature_hashes_match':True,
        'four_raw_examples_rebuilt':True,'both_masks_frozen_before_final_predictions':True,
        'root_model_is_predeclared_grouped_model':True,'generations':plan['generations'],
        'root_model_predictions_and_metadata_verified':True,
        'population':plan['population'],'subject_independent':False,
        'grouped_result_exploratory':True,'paper_result_diagnostic_only':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=Path('artifacts_paper_method/knn100'))
    parser.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset')
    parser.add_argument('--manifest',default='artifacts/manifest.csv')
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        verify(args.artifacts,args.data,args.manifest)


if __name__=='__main__':
    main()
