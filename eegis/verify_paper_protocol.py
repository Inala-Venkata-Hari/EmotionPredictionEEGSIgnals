"""Independently replay selected masks, both stages, and saved paper-method models."""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import digest
from .diagnose_leakage import leakage_counts
from .paper_equations import COLUMNS, EquationConfig, equation_features, specification
from .paper_ga import CLASS_IDS, CLASS_NAMES, K_VALUES
from .paper_protocol import code_fingerprint, estimator, stage_splits
from .migrate_paper_checkpoints import check_source
from .train_paper_binary import array_sha, binary_data, dump


def independent_transform(bank,genome,fit,normalization):
    channels=np.flatnonzero(np.asarray(genome[:14]))
    columns=np.flatnonzero(np.asarray(genome[14:]))
    selected=bank[:,channels][:,:,columns]
    if normalization=='within-window':
        rows=[]
        for row in selected:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='All-NaN slice encountered')
                medians=np.nanmedian(row,axis=0)
            medians=np.nan_to_num(medians,nan=0.)
            filled=np.where(np.isnan(row),medians[None],row)
            rows.append(StandardScaler().fit_transform(filled).ravel())
        return np.stack(rows)
    selected=selected.reshape(len(bank),-1)
    filled=SimpleImputer(strategy='median',keep_empty_features=True).fit(selected[fit]).transform(selected)
    return StandardScaler().fit(filled[fit]).transform(filled)


def reference_fit(name,k,z,labels,fit,seed):
    model=estimator(name,k or 7)
    state=np.random.get_state()
    try:
        np.random.seed(seed)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(z[fit],labels[fit])
    finally:
        np.random.set_state(state)
    return model


def verify(out,data,manifest_path):
    plan=json.loads((out/'plan.json').read_text())
    audit=json.loads((out/'feature_audit.json').read_text())
    result=json.loads((out/'results.json').read_text())
    config=EquationConfig(**plan['feature_config'])
    current=code_fingerprint()
    if plan['code_sha256']!=current:
        migration=json.loads((out/'checkpoint_migration.json').read_text())
        assert migration['original_training_code_sha256']==plan['code_sha256']
        assert migration['current_code_sha256']==current
        assert migration['new_training_executed'] is False
        check_source(out,plan,current)
    assert json.loads((out/'feature_spec.json').read_text())==specification(config)
    x,source=binary_data(data,manifest_path)
    pd.testing.assert_frame_equal(source,pd.read_csv(out/'source_manifest.csv'))
    assert digest(x)==audit['source_sha256']
    with np.load(audit['cache_path'],allow_pickle=False) as cache:
        bank=cache['features']
        assert str(cache['key'])==audit['cache_key']
        assert str(cache['array_sha'])==array_sha(bank)==audit['array_sha256']
    assert list(bank.shape)==audit['shape']
    undefined=pd.read_csv(out/'undefined_features.csv')
    np.testing.assert_array_equal(undefined.column,COLUMNS)
    np.testing.assert_array_equal(undefined.undefined_values,np.isnan(bank).sum(axis=(0,1)))
    assert int(np.isnan(bank).sum())==audit['undefined_values']
    subset=np.array([0,479,480,923])
    np.testing.assert_allclose(equation_features(x[subset],config),bank[subset],rtol=1e-10,atol=1e-10,equal_nan=True)
    labels,groups=source.label.to_numpy(),source.overlap_group.to_numpy()
    fit,search,later_fit,later_test=stage_splits(labels,plan['seed'],plan['validation_seed'])
    m=pd.read_csv(out/'manifest.csv')
    np.testing.assert_array_equal(m.path,source.path)
    np.testing.assert_array_equal(np.flatnonzero(m.stage1=='fit'),fit)
    np.testing.assert_array_equal(np.flatnonzero(m.stage1=='fitness'),search)
    np.testing.assert_array_equal(np.flatnonzero(m.stage2=='fit'),later_fit)
    np.testing.assert_array_equal(np.flatnonzero(m.stage2=='evaluation'),later_test)
    overlap=json.loads((out/'split_audit.json').read_text())
    assert overlap['stage1_overlap']==leakage_counts(x,groups,fit,search)
    assert overlap['stage2_overlap']==leakage_counts(x,groups,later_fit,later_test)
    saved=pd.read_csv(out/'predictions.csv')
    comparison=pd.read_csv(out/'comparison.csv').set_index('classifier')
    selected={}
    checked_candidates=0
    for name in plan['classifiers']:
        directory=out/name
        chosen=json.loads((directory/'selection.json').read_text())
        selected[name]=chosen
        z=independent_transform(bank,chosen['genome'],fit,plan['normalization'])
        candidate_scores=[]
        for k in K_VALUES if name=='knn' else [None]:
            model=reference_fit(name,k,z,labels,fit,plan['seed'])
            predicted=model.predict(z[search])
            candidate_scores.append({'k':k,'accuracy':float(np.mean(predicted==labels[search])),
                'macro_f1':float(f1_score(labels[search],predicted,labels=CLASS_IDS,average='macro',zero_division=0))})
        best=max(candidate_scores,key=lambda row:row['accuracy'])
        assert best['k']==chosen['k'] and best['accuracy']==chosen['accuracy']
        assert abs(best['macro_f1']-chosen['macro_f1'])<1e-12
        records=pd.read_csv(directory/'evaluations.csv',dtype={'genome':str})
        assert len(records)==chosen['unique_evaluations'] and records.genome.nunique()==len(records)
        lookup=dict(zip(records.genome,records.accuracy))
        checked_candidates+=len(records)
        history=pd.read_csv(directory/'history.csv')
        np.testing.assert_array_equal(history.generation,np.arange(plan['generations']+1))
        with np.load(directory/'populations.npz',allow_pickle=False) as data_file:
            states,scores=data_file['genomes'],data_file['accuracies']
        assert states.shape==(plan['generations']+1,plan['population'],228)
        for i,(population,values) in enumerate(zip(states,scores)):
            np.testing.assert_allclose(values,[lookup[''.join(map(str,row))] for row in population],rtol=0,atol=1e-12)
            assert np.all(np.diff(values)<=0)
            assert abs(values[0]-history.best_validation_accuracy.iloc[i])<1e-12
            assert abs(values.mean()-history.mean_survivor_accuracy.iloc[i])<1e-12
        np.testing.assert_array_equal(states[-1,0],chosen['genome'])
        stored=joblib.load(directory/'model.joblib')
        np.testing.assert_array_equal(stored.genome,chosen['genome'])
        z=independent_transform(bank,chosen['genome'],later_fit,plan['normalization'])
        reference=reference_fit(name,9 if name=='knn' else None,z,labels,later_fit,plan['seed'])
        assert stored.estimator.get_params(deep=False)==reference.get_params(deep=False)
        predicted=reference.predict(z[later_test])
        part=saved[saved.classifier==name]
        np.testing.assert_array_equal(part.path,source.path.iloc[later_test])
        np.testing.assert_array_equal(part.label,labels[later_test])
        np.testing.assert_array_equal(part.prediction,predicted)
        np.testing.assert_array_equal(stored.predict_features(bank[later_test]),predicted)
        np.testing.assert_array_equal(stored.predict(x[later_test[:1]]),predicted[:1])
        measured={'accuracy':float(np.mean(predicted==labels[later_test])),
            'macro_f1':float(f1_score(labels[later_test],predicted,labels=CLASS_IDS,average='macro',zero_division=0))}
        for key,value in measured.items():
            assert abs(value-result['models'][name]['stage2'][key])<1e-12
        assert abs(comparison.loc[name,'stage2_accuracy']-measured['accuracy'])<1e-12
        assert abs(comparison.loc[name,'stage2_macro_f1']-measured['macro_f1'])<1e-12
        detail=classification_report(labels[later_test],predicted,labels=CLASS_IDS,
            target_names=CLASS_NAMES,output_dict=True,zero_division=0)
        for label in CLASS_NAMES:
            for metric,value in detail[label].items():
                assert abs(value-result['models'][name]['stage2']['report'][label][metric])<1e-12
        cm=confusion_matrix(labels[later_test],predicted,labels=CLASS_IDS)
        np.testing.assert_array_equal(cm,
                                      np.loadtxt(directory/'confusion.csv',delimiter=',',dtype=int))
        tn,fp,fn,tp=cm.ravel()
        precision=float(tp/(tp+tn)) if tp+tn else 0.
        recall=float(tp/(tp+fn)) if tp+fn else 0.
        printed_f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        assert result['models'][name]['stage2']['paper_printed_precision_for_Si']==precision
        assert result['models'][name]['stage2']['paper_printed_f1_for_Si']==printed_f1
        assert result['models'][name]['stage2_k']==(9 if name=='knn' else None)
        print(name+': both stages, checkpoint and metrics verified',flush=True)
    winner=max(plan['classifiers'],key=lambda name:selected[name]['accuracy'])
    selection=json.loads((out/'selection.json').read_text())
    assert selection['winner']==result['selected_model']==winner
    assert selection['frozen_before_stage2'] is True and result['selection_uses_stage2'] is False
    best_model=joblib.load(out/'best_model.joblib')
    np.testing.assert_array_equal(best_model.genome,selected[winner]['genome'])
    winning_predictions=saved[saved.classifier==winner].prediction.to_numpy()
    np.testing.assert_array_equal(best_model.predict_features(bank[later_test]),winning_predictions)
    majority=int(pd.Series(labels[later_fit]).mode().iloc[0])
    assert result['majority']['accuracy']==float(np.mean(labels[later_test]==majority))
    for filename in ['training_graph.png','comparison.png']:
        assert (out/filename).stat().st_size>1000
    receipt={'classifiers_verified':len(plan['classifiers']),'source_and_feature_hashes_checked':True,
        'four_raw_examples_rebuilt':True,'selected_fitness_independently_recomputed_for_all_classifiers':True,
        'all_seven_knn_values_independently_recomputed':True,
        'stored_candidate_score_records_checked_for_consistency':checked_candidates,
        'all_candidate_fitness_predictions_recomputed':False,
        'both_stage_splits_and_overlap_recomputed':True,'all_stage2_models_refitted_independently':True,
        'raw_inference_and_cached_feature_predictions_match':True,'all_accuracy_macro_f1_confusions_recomputed':True,
        'winner_uses_stage1_only':True,'diagnostic_only':True,'full_100_generation_study':result['full_100_generation_study'],
        'exact_unpublished_author_execution_reproduced':False}
    dump(out/'verification.json',receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=Path('artifacts_paper_method/smoke'))
    parser.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset')
    parser.add_argument('--manifest',default='artifacts/manifest.csv')
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        verify(args.artifacts,args.data,args.manifest)


if __name__=='__main__':
    main()
