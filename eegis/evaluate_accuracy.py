"""Freeze train-CV winners, refit with validation groups, evaluate reused test once.

This is a separately declared final refit stage. All source studies' original
training-only models and measurements are preserved. Test scores never choose
the study, representation, parameters, duration, seed or primary checkpoint.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report,confusion_matrix
from threadpoolctl import threadpool_limits

from .accuracy_features import BinaryAccuracyModel
from .data import audit_shared_samples,digest
from .epochs import load_recordings
from .train_accuracy import score,write
from .train_paper_binary import binary_data
from .verify_accuracy import grouped_uncertainty


STUDIES=[('artifacts_accuracy','train_accuracy'),('artifacts_signal_models','train_signal_models'),
         ('artifacts_domain_models','train_domain_models'),('artifacts_minirocket','train_minirocket'),
         ('artifacts_pretrained','train_pretrained'),('artifacts_finetune','train_finetune')]


def exported(study,estimator,bank,samples,cache):
    if study=='artifacts_accuracy':return BinaryAccuracyModel(estimator,bank,cache['kernels'],samples)
    if study=='artifacts_signal_models':
        from .signal_models import SignalModel
        return SignalModel(estimator,bank,samples)
    if study=='artifacts_domain_models':
        from .domain_features import DomainModel
        return DomainModel(estimator,bank,samples)
    if study=='artifacts_minirocket':
        from .train_minirocket import MiniRocketModel
        return MiniRocketModel(estimator,bank,samples)
    if study=='artifacts_finetune':
        from .finetune_features import FineTuneModel
        return FineTuneModel(estimator,bank,samples)
    from .pretrained_features import PretrainedModel
    return PretrainedModel(estimator,bank,samples)


def run(args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    windows,wm=binary_data(args.data,args.manifest);wm=wm.rename(columns={'original_split':'split'})
    recordings,rm=load_recordings(args.data,wm)
    candidates={unit:[] for unit in ['window','recording']};source_hashes={}
    for study,module_name in STUDIES:
        base=Path(study);verification=json.loads((base/'verification.json').read_text())
        assert all(r['all_outer_predictions_and_metrics_match'] for r in verification['units'])
        module=importlib.import_module('eegis.'+module_name)
        original_plan=json.loads((base/'plan.json').read_text())
        assert original_plan['source_sha256']==module.fingerprint()
        source_hashes[study]={'plan':hashlib.sha256((base/'plan.json').read_bytes()).hexdigest(),
            'verification':hashlib.sha256((base/'verification.json').read_bytes()).hexdigest()}
        for unit in candidates:
            data=json.loads((base/unit/'final_search/selection.json').read_text())
            for value in data['families']:candidates[unit].append({'study':study,'module':module_name,**value})
    # No final predictions exist before this selection is frozen on disk.
    winners={unit:max(rows,key=lambda r:(r['inner_balanced_accuracy'],r['inner_macro_f1'])) for unit,rows in candidates.items()}
    plan={'selection':'original-training inner-CV balanced accuracy, macro-F1, declared study/family order',
        'winners':winners,'all_candidates':candidates,'source_studies':source_hashes,
        'new_refit_stage':'original train plus original validation groups; original source models preserved',
        'evaluation':'original test only, previously reported in earlier research; exploratory',
        'primary_model':'window, predeclared regardless final scores','subject_independent':False,
        'prototype_target':{'balanced_accuracy':.7,'macro_f1':.7,'approximate_lower_interval_above':.5},
        'no_duration_or_hyperparameter_selection_by_test':True,'raw_window_sha256':digest(windows),
        'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if (root/'selection_plan.json').exists() and json.loads((root/'selection_plan.json').read_text())!=plan:
        raise ValueError('Frozen final-refit plan differs; use a new output')
    if (root/'results.json').exists():
        raise ValueError('This fixed final evaluation already completed; do not repeatedly score it')
    write(root/'selection_plan.json',plan)
    results=[];receipts=[]
    for unit,raw,m in [('window',windows,wm),('recording',recordings,rm)]:
        chosen=winners[unit];source=Path(chosen['study'])/unit
        cache=joblib.load(source/'features.joblib');assert cache['source_sha']==digest(raw)
        module=importlib.import_module('eegis.'+chosen['module']);assert cache['code']==module.fingerprint()
        bank,template={n:(b,e) for n,b,e,_ in module.candidates()}[chosen['name']]
        assert bank==chosen['bank']
        fit=np.flatnonzero(m.split!='test');test=np.flatnonzero(m.split=='test');y=m.label.to_numpy()
        assert set(m.overlap_group.iloc[fit]).isdisjoint(set(m.overlap_group.iloc[test]))
        if unit=='window':
            audit=m.copy();audit['split']=np.where(m.split=='test','evaluation','fitting')
            audit_shared_samples(raw,audit)
        model=clone(template).set_params(**chosen['parameters']).fit(cache['banks'][bank][fit],y[fit])
        pipeline=exported(chosen['study'],model,bank,raw.shape[1],cache)
        pred=model.predict(cache['banks'][bank][test]);majority=int(pd.Series(y[fit]).mode().iloc[0])
        np.testing.assert_array_equal(pipeline.predict(raw[test[:2]]),pred[:2])
        out=root/unit;out.mkdir(exist_ok=True)
        joblib.dump(pipeline,out/'best_model.joblib')
        predictions=m.iloc[test][['overlap_group','label']].copy();predictions['prediction']=pred;predictions['majority']=majority
        predictions.to_csv(out/'test_predictions.csv',index=False)
        np.savetxt(out/'confusion.csv',confusion_matrix(y[test],pred,labels=[6,8]),fmt='%d',delimiter=',')
        uncertainty=grouped_uncertainty(predictions)
        value={'unit':unit,'duration_seconds':raw.shape[1]/128,'source_study':chosen['study'],'family':chosen['name'],
            'selection_cv_balanced_accuracy':chosen['inner_balanced_accuracy'],'metrics':score(y[test],pred),
            'majority':score(y[test],np.full(len(test),majority)),'uncertainty':uncertainty,
            'fit_windows_or_recordings':len(fit),'test_windows_or_recordings':len(test),
            'fit_overlap_components':m.overlap_group.iloc[fit].nunique(),'test_overlap_components':m.overlap_group.iloc[test].nunique(),
            'subject_independent':False,'exploratory':True,'test_previously_reported':True,
            'classification_report':classification_report(y[test],pred,labels=[6,8],target_names=['No','Sí'],output_dict=True,zero_division=0)}
        write(out/'results.json',value);results.append(value)
        # Independent second refit, separate from the saved estimator object.
        reference=clone(template).set_params(**chosen['parameters']).fit(cache['banks'][bank][fit],y[fit])
        np.testing.assert_array_equal(reference.predict(cache['banks'][bank][test]),pred)
        loaded=joblib.load(out/'best_model.joblib')
        np.testing.assert_array_equal(loaded.predict(raw[test[:1]]),pred[:1])
        receipts.append({'unit':unit,'selected_by_training_cv_only':True,'refit_has_no_test_groups':True,
            'independent_final_refit_matches_all_predictions':True,'saved_raw_inference_matches':True,
            'all_test_metrics_and_confusion_computed_from_saved_predictions':True})
        if unit=='window':joblib.dump(pipeline,root/'best_model.joblib')
        print('FINAL EXPLORATORY',unit,value['metrics'],flush=True)
    write(root/'results.json',{'models':results,'primary_model':'window','all_results_exploratory':True})
    write(root/'verification.json',{'units':receipts,'frozen_plan_before_any_final_prediction':True,
        'primary_predeclared_window_model_verified':True,'source_study_measurements_preserved':True})
    write(root/'best_model.json',{'file':'best_model.joblib','unit':'window','class_ids':[6,8],
        'input':'raw (N,128,14) EEG','selection':plan['selection'],'exploratory':True,'subject_independent':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='artifacts_accuracy_refit')
    p.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset');p.add_argument('--manifest',default='artifacts/manifest.csv')
    args=p.parse_args()
    with threadpool_limits(limits=4):run(args)


if __name__=='__main__':main()
