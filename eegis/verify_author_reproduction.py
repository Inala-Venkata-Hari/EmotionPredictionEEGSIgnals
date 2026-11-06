"""Replay the author-source feature bank, published masks, and 100-generation runs."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .author_features import author_features, load_layers, published_genome, source_spec
from .author_reproduction import fixed_cases
from .data import audit_shared_samples, digest
from .diagnose_leakage import leakage_counts
from .paper_ga import CLASS_IDS, fitness_predictions, genome_columns
from .train_paper_binary import array_sha, binary_data, dump, metrics, split_binary


def check_model(model, features, labels, train, test, expected, scaler_rows=None):
    cols = genome_columns(model.genome)
    flat = features.reshape(len(features), -1)[:, cols]
    rows = train if scaler_rows is None else scaler_rows
    if getattr(model.scaler,'per_example',False):
        matrices=flat.reshape(len(flat),model.scaler.channels,model.scaler.features)
        independently_scaled=np.stack([StandardScaler().fit_transform(a).ravel() for a in matrices])
        np.testing.assert_allclose(model.scaler.transform(flat),independently_scaled,rtol=1e-10,atol=1e-10)
    else:
        np.testing.assert_allclose(model.scaler.mean_, flat[rows].mean(axis=0), rtol=1e-10, atol=1e-10)
        independently_scaled=model.scaler.transform(flat)
    stored=getattr(model.estimator,'x_',None)
    if stored is None:
        stored=model.estimator._fit_X
    np.testing.assert_allclose(stored, independently_scaled[train])
    model.estimator.device = 'cpu'
    np.testing.assert_array_equal(model.predict_features(features[test]), expected)
    reference = KNeighborsClassifier(n_neighbors=model.estimator.n_neighbors, algorithm='brute')
    reference.fit(independently_scaled[train], labels[train])
    np.testing.assert_array_equal(reference.predict(independently_scaled[test]), expected)


def verify_ga(out, x, source, layers, features):
    plan = json.loads((out/'plan.json').read_text())
    result = json.loads((out/'results.json').read_text())
    selected = json.loads((out/'knn_selection.json').read_text())
    m, train, test, fit, val = split_binary(source, plan['protocol'], plan['seed'])
    pd.testing.assert_frame_equal(m, pd.read_csv(out/'manifest.csv'))
    y, groups = m.label.to_numpy(), m.overlap_group.to_numpy()
    for a, b, name in [(train,test,'outer'),(fit,val,'inner')]:
        assert leakage_counts(x, groups, a, b) == plan['overlap_audit'][name]
    if not plan['diagnostic_only']:
        audit_shared_samples(x, m.rename(columns={'outer_split':'split'}))
        inner=m.iloc[train].copy();inner['split']=inner.search_role
        audit_shared_samples(x[train], inner)
    local_fit,local_val=np.flatnonzero(np.isin(train,fit)),np.flatnonzero(np.isin(train,val))
    # Independent original scorer (sklearn metrics), rather than the optimized callback.
    if plan.get('normalization','train')=='within-window':
        from .paper_ga import K_VALUES
        flat=features.reshape(len(features),-1)[:,genome_columns(selected['genome'])]
        dimensions=(len(selected['selected_channels']),len(selected['selected_features']))
        z=np.stack([StandardScaler().fit_transform(a.reshape(*dimensions)).ravel() for a in flat])
        candidates=[]
        for k in K_VALUES:
            reference=KNeighborsClassifier(n_neighbors=k,algorithm='brute').fit(z[fit],y[fit])
            candidates.append({'k':k,**metrics(y[val],reference.predict(z[val]))})
        score=max(candidates,key=lambda r:r['accuracy'])
    else:
        score=fitness_predictions(features[train],y[train],local_fit,local_val,np.array(selected['genome']),
                                  'knn',plan['seed'],'cpu')
    assert score['k']==selected['k'] and score['accuracy']==selected['accuracy']
    assert abs(score['macro_f1']-selected['macro_f1'])<1e-12
    evaluated=pd.read_csv(out/'knn_search_evaluations.csv',dtype={'genome':str})
    assert len(evaluated)==selected['unique_evaluations']
    assert evaluated.genome.nunique()==len(evaluated)
    assert abs(evaluated.accuracy.max()-selected['accuracy'])<1e-12
    lookup=dict(zip(evaluated.genome,evaluated.accuracy))
    history=pd.read_csv(out/'knn_history.csv')
    np.testing.assert_array_equal(history.generation,np.arange(plan['generations']+1))
    assert np.all(np.diff(history.best_validation_accuracy)>=-1e-12)
    with np.load(out/'knn_populations.npz',allow_pickle=False) as data:
        populations,accuracies=data['genomes'],data['accuracies']
        assert populations.shape==(plan['generations']+1,plan['population_size'],228)
        for index,(genomes,values) in enumerate(zip(populations,accuracies)):
            np.testing.assert_allclose(values,[lookup[''.join(map(str,g))] for g in genomes],atol=1e-12)
            assert np.all(np.diff(values)<=0)
            assert abs(values[0]-history.best_validation_accuracy.iloc[index])<1e-12
            assert abs(values.mean()-history.mean_survivor_accuracy.iloc[index])<1e-12
        np.testing.assert_array_equal(populations[-1,0],selected['genome'])
    saved=pd.read_csv(out/'test_predictions.csv')
    np.testing.assert_array_equal(saved.path,m.path.iloc[test])
    np.testing.assert_array_equal(saved.label,y[test])
    np.testing.assert_array_equal(saved.overlap_group,groups[test])
    model=joblib.load(out/'best_model.joblib')
    check_model(model,features,y,train,test,saved.knn)
    np.testing.assert_array_equal(model.predict_layers(layers[test[:1]]),saved.knn.to_numpy()[:1])
    measured=metrics(y[test],saved.knn)
    for key,value in measured.items():
        assert abs(value-result['models']['knn']['test'][key])<1e-12
    majority=int(pd.Series(y[train]).mode().iloc[0])
    np.testing.assert_array_equal(saved.majority,np.full(len(test),majority))
    assert metrics(y[test],saved.majority)==result['majority']
    np.testing.assert_array_equal(confusion_matrix(y[test],saved.knn,labels=CLASS_IDS),
                                  np.loadtxt(out/'knn_confusion.csv',delimiter=','))
    assert result['target_met_on_grouped_test']==(not plan['diagnostic_only'] and measured['accuracy']>=.96)
    verification={'protocol':plan['protocol'],'diagnostic_only':plan['diagnostic_only'],
                  'normalization':plan.get('normalization','train'),
                  'source_split_and_overlap_counts_reproduced':True,
                  'all_population_states_checked':len(history),'candidate_fitness_scores_checked':len(evaluated),
                  'selected_fitness_recomputed_with_original_scorer':True,'scaler_fitted_to_training_only':True,
                  'scaler_policy':'independent within-example electrode statistics' if plan.get('normalization')=='within-window' else 'outer training population only',
                  'checkpoint_and_sklearn_predictions_match':True,'six_layer_inference_matches':True,
                  'all_test_metrics_and_confusions_recomputed':True,'target_status_verified':True,
                  'cuda_execution_verified':False}
    dump(out/'verification.json',verification)
    print(json.dumps(verification,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',default='artifacts_author_reproduction')
    p.add_argument('--completed-only',action='store_true',help='Verify completed runs while another predeclared run is still training')
    args=p.parse_args();out=Path(args.artifacts)
    audit=json.loads((out/'feature_audit.json').read_text())
    spec=source_spec()
    assert hashlib.sha256(Path(audit['manifest']).read_bytes()).hexdigest()==audit['source_manifest_sha256']
    x,source=binary_data(audit['data'],audit['manifest'])
    pd.testing.assert_frame_equal(source,pd.read_csv(out/'source_manifest.csv'))
    layers=load_layers(audit['data'],source)
    assert digest(layers)==audit['source_layers_sha256']
    assert hashlib.sha256(Path(__file__).with_name('author_features.py').read_bytes()).hexdigest()==audit['feature_code_sha256']
    assert hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()==audit['feature_spec_sha256']
    with np.load(out/'feature_cache.npz',allow_pickle=False) as data:
        features=data['features']
    assert array_sha(features)==audit['feature_array_sha256']
    with threadpool_limits(limits=4):
        subset=[0,len(features)//2,len(features)-1]
        np.testing.assert_allclose(author_features(layers[subset]),features[subset],rtol=1e-10,atol=1e-10)
        saved=pd.read_csv(out/'published_mask_predictions.csv')
        scores=pd.read_csv(out/'published_mask_results.csv')
        stored_audit=json.loads((out/'published_mask_audit.json').read_text())
        seed=json.loads((out/'experiment_plan.json').read_text())['seed']
        y,groups=source.label.to_numpy(),source.overlap_group.to_numpy()
        for index,(case,train,test,global_scaling) in enumerate(fixed_cases(source,seed)):
            assert leakage_counts(x,groups,train,test)==stored_audit[index]['overlap']
            for k in [7,9]:
                prediction=saved[(saved['case']==case)&(saved.k==k)]
                np.testing.assert_array_equal(prediction.path,source.path.iloc[test])
                model=joblib.load(out/f'published_{case}_k{k}.joblib')
                np.testing.assert_array_equal(model.genome,published_genome())
                check_model(model,features,y,train,test,prediction.prediction,
                            scaler_rows=np.arange(len(source)) if global_scaling else None)
                row=scores[(scores['case']==case)&(scores.k==k)].iloc[0]
                for metric,value in metrics(y[test],prediction.prediction).items():
                    assert abs(row[metric]-value)<1e-12
        dump(out/'published_mask_verification.json',{'cases_verified':8,'all_predictions_and_metrics_recomputed':True,
              'all_scaler_fit_rows_checked':True,'all_masks_match_published_genome':True,
              'sklearn_prediction_parity':True,'all_cases_diagnostic_only':True})
        for target in [out,out/'paper_window_diagnostic',out/'within_window_grouped']:
            if args.completed_only and not (target/'results.json').exists():
                continue
            verify_ga(target,x,source,layers,features)


if __name__=='__main__':
    main()
