"""Independent refits and uncertainty estimates for binary accuracy research."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numba
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from .accuracy_features import BinaryAccuracyModel
from .data import digest
from .epochs import load_recordings
from .train_accuracy import candidates, fingerprint, folds, score, write
from .train_paper_binary import binary_data


def grouped_uncertainty(frame, seed=42, draws=10000):
    """Paired group bootstrap of OOF predictions; approximate, exploratory CI."""
    matrices, baseline = [], []
    for _, group in frame.groupby('overlap_group', sort=True):
        matrices.append(confusion_matrix(group.label, group.prediction, labels=[6,8]))
        baseline.append(confusion_matrix(group.label, group.majority, labels=[6,8]))
    rng = np.random.default_rng(seed)
    ix = rng.integers(len(matrices), size=(draws, len(matrices)))
    def calculate(values):
        cm = np.asarray(values)[ix].sum(axis=1)
        totals = cm.sum(axis=2)
        ok = (totals > 0).all(axis=1)
        return .5*(cm[ok,0,0]/totals[ok,0]+cm[ok,1,1]/totals[ok,1]), ok
    values, ok = calculate(matrices)
    other, other_ok = calculate(baseline)
    np.testing.assert_array_equal(ok, other_ok)
    return {'method':'paired overlap-group bootstrap of nested OOF predictions; exploratory approximate intervals',
            'draws':draws, 'seed':seed, 'balanced_accuracy_ci95':np.percentile(values,[2.5,97.5]).tolist(),
            'difference_from_fold_majority_ci95':np.percentile(values-other,[2.5,97.5]).tolist(),
            'does_not_account_for_all_adaptive_research_choices':True}


def verify(root, data, manifest):
    root = Path(root)
    plan = json.loads((root/'plan.json').read_text())
    if 'window_data_sha256' in plan:
        catalogue, source_hashes, factory = candidates, fingerprint, BinaryAccuracyModel
    else:
        from .train_signal_models import candidates as signal_candidates, fingerprint as signal_fingerprint
        from .signal_models import SignalModel
        catalogue, source_hashes = signal_candidates, signal_fingerprint
        factory = lambda estimator,bank,kernels,samples: SignalModel(estimator,bank,samples)
    assert plan['source_sha256'] == source_hashes()
    assert all(version(name)==value for name,value in plan['libraries'].items())
    assert hashlib.sha256(Path(manifest).read_bytes()).hexdigest() == plan['manifest_sha256']
    assert (root/'search_completed.json').exists()
    windows, wm = binary_data(data, manifest)
    wm = wm.rename(columns={'original_split':'split'})
    recordings, rm = load_recordings(data, wm)
    assert digest(windows) == plan.get('window_data_sha256',plan.get('window_sha256'))
    lookup = {name:(bank, model) for name,bank,model,_ in catalogue()}
    receipts = []
    for unit,x,m in [('recording',recordings,rm),('window',windows,wm)]:
        directory = root/unit
        cache = joblib.load(directory/'features.joblib')
        assert cache['code'] == source_hashes() and cache['source_sha'] == digest(x)
        banks,kernels = cache['banks'],cache.get('kernels')
        training = np.flatnonzero(m.split == 'train')
        original_holdout_groups = set(m.overlap_group[m.split != 'train'])
        saved = pd.read_csv(directory/'nested_predictions.csv')
        np.testing.assert_array_equal(saved.label,m.label.iloc[training])
        np.testing.assert_array_equal(saved.overlap_group,m.overlap_group.iloc[training])
        y = m.label.to_numpy()
        membership = json.loads((directory/'cv_membership.json').read_text())
        actual_outer = folds(m,training,5,42)
        selected_refits = 0
        for fold,(fit,val) in enumerate(actual_outer):
            a,b = training[fit],training[val]
            fit_groups=set(m.overlap_group.iloc[a]); val_groups=set(m.overlap_group.iloc[b])
            assert fit_groups.isdisjoint(val_groups | original_holdout_groups)
            assert val_groups.isdisjoint(original_holdout_groups)
            assert set(membership[fold]['fit_groups'])==fit_groups
            assert set(membership[fold]['validation_groups'])==val_groups
            directory_fold=directory/('outer_'+str(fold))
            selection=json.loads((directory_fold/'selection.json').read_text())
            rows=selection['families']
            independently_selected=max(rows,key=lambda r:(r['inner_balanced_accuracy'],r['inner_macro_f1']))
            chosen=selection['winner'];assert chosen==independently_selected
            bank,model=lookup[chosen['name']]
            inner=folds(m,a,3,100+fold)
            inner_scores=[]
            for index,(i,j) in enumerate(inner):
                inner_fit=set(m.overlap_group.iloc[a[i]]);inner_val=set(m.overlap_group.iloc[a[j]])
                assert inner_fit.isdisjoint(inner_val | val_groups | original_holdout_groups)
                assert inner_val.isdisjoint(val_groups | original_holdout_groups)
                assert set(membership[fold]['inner'][index]['fit_groups'])==inner_fit
                assert set(membership[fold]['inner'][index]['validation_groups'])==inner_val
                estimator=clone(model).set_params(**chosen['parameters']).fit(banks[bank][a[i]],y[a[i]])
                pred=estimator.predict(banks[bank][a[j]])
                inner_scores.append(score(y[a[j]],pred));selected_refits+=1
            for metric,field in [('balanced_accuracy','inner_balanced_accuracy'),('macro_f1','inner_macro_f1')]:
                assert abs(np.mean([s[metric] for s in inner_scores])-chosen[field])<1e-10
            estimator=clone(model).set_params(**chosen['parameters']).fit(banks[bank][a],y[a])
            pred=estimator.predict(banks[bank][b])
            np.testing.assert_array_equal(pred,saved.prediction.iloc[val])
            np.testing.assert_array_equal(saved.outer_fold.iloc[val],np.full(len(val),fold))
            majority=int(pd.Series(y[a]).mode().iloc[0])
            np.testing.assert_array_equal(saved.majority.iloc[val],np.full(len(val),majority))
            # Check raw-vs-cached values and exact predicted labels independently.
            raw_model=factory(estimator,bank,kernels,x.shape[1])
            rebuilt=raw_model.features(x[b[:2]])
            np.testing.assert_allclose(rebuilt,banks[bank][b[:2]],rtol=1e-5,atol=1e-6)
            np.testing.assert_array_equal(raw_model.predict(x[b[:2]]),pred[:2])
            # Every stored candidate table's reported selection is checked for
            # consistency, distinct from independently repeating all candidates.
            for family in rows:
                table=pd.read_csv(directory_fold/(family['name']+'_cv.csv'))
                index=max(range(len(table)),key=lambda i:(table.mean_test_balanced_accuracy.iloc[i],table.mean_test_macro_f1.iloc[i],-i))
                assert abs(table.mean_test_balanced_accuracy.iloc[index]-family['inner_balanced_accuracy'])<1e-10
        results=json.loads((directory/'nested_results.json').read_text())
        assert results['nested']==score(saved.label,saved.prediction)
        assert results['majority']==score(saved.label,saved.majority)
        chosen=json.loads((directory/'best_model.json').read_text())
        assert set(chosen['training_groups'])==set(m.overlap_group.iloc[training])
        assert set(chosen['training_groups']).isdisjoint(original_holdout_groups)
        model=joblib.load(directory/'best_model.joblib')
        bank,estimator=lookup[chosen['winner']['name']]
        refit=clone(estimator).set_params(**chosen['winner']['parameters']).fit(banks[bank][training],y[training])
        reference=refit.predict(banks[bank][training])
        np.testing.assert_array_equal(model.predict(x[training[:2]]),reference[:2])
        np.testing.assert_array_equal(model.estimator.predict(banks[bank][training]),reference)
        uncertainty=grouped_uncertainty(saved)
        write(directory/'uncertainty.json',uncertainty)
        np.savetxt(directory/'nested_confusion.csv',confusion_matrix(saved.label,saved.prediction,labels=[6,8]),fmt='%d',delimiter=',')
        receipts.append({'unit':unit,'outer_models_independently_refitted':5,
                         'selected_inner_models_independently_refitted':selected_refits,
                         'all_outer_predictions_and_metrics_match':True,'raw_cached_inference_match':True,
                         'all_fold_groups_audited':True,'stored_candidate_score_consistency_checked':True,
                         'all_candidate_fits_independently_repeated':False,'final_model_independently_refitted':True})
        print('Verified',unit,results['nested'],flush=True)
    write(root/'verification.json',{'source_data_runtime_plan_checked':True,'units':receipts,
                                   'original_validation_test_not_evaluated':True,'subject_independent':False,
                                   'all_results_exploratory':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',default='artifacts_accuracy')
    parser.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset')
    parser.add_argument('--manifest',default='artifacts/manifest.csv')
    args=parser.parse_args();numba.set_num_threads(4)
    with threadpool_limits(limits=4):
        verify(args.artifacts,args.data,args.manifest)


if __name__=='__main__':
    main()
