"""Second predeclared train-only comparison: connectivity, lags and tiny CNNs."""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .data import digest, audit_shared_samples
from .epochs import load_recordings
from .signal_models import BinaryPowerCNN, SignalModel, signal_banks
from .train_accuracy import choose_index, folds, score, write
from .train_paper_binary import binary_data


def candidates():
    specs=[]
    for bank in ['connectivity','lagged','high_frequency','joint']:
        specs.append((bank+'_ridge',bank,Pipeline([('select',SelectKBest(f_classif)),('scale',StandardScaler()),
            ('model',RidgeClassifier(class_weight='balanced',solver='lsqr',tol=1e-4))]),
            {'select__k':[30,100,300],'model__alpha':[1.,100.]}))
    specs.append(('connectivity_svm','connectivity',Pipeline([('select',SelectKBest(f_classif)),('scale',StandardScaler()),
        ('model',SVC(class_weight='balanced'))]),{'select__k':[30,100],'model__C':[.1,1.,10.]}))
    specs.append(('power_cnn','raw',BinaryPowerCNN(),{'high':[40.,60.]}))
    specs.append(('majority','high_frequency',DummyClassifier(strategy='most_frequent'),{}))
    return specs


def fingerprint():
    files=['train_signal_models.py','signal_models.py','accuracy_features.py','train_accuracy.py',
           'data.py','epochs.py','improve.py','train_paper_binary.py','paper_ga.py','paper_features.py']
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files}


def search(banks,y,rows,cv,out,context):
    out.mkdir(parents=True,exist_ok=True);summaries=[]
    for name,bank,model,grid in candidates():
        path=out/(name+'.json')
        if path.exists():
            summary=json.loads(path.read_text())
        else:
            print(context,'fitting',name,flush=True)
            selected=GridSearchCV(model,grid,scoring={'balanced_accuracy':'balanced_accuracy','macro_f1':'f1_macro'},
                refit=choose_index,cv=cv,n_jobs=1,error_score='raise').fit(banks[bank][rows],y[rows])
            index=selected.best_index_
            summary={'name':name,'bank':bank,'parameters':selected.best_params_,
                'inner_balanced_accuracy':float(selected.cv_results_['mean_test_balanced_accuracy'][index]),
                'inner_macro_f1':float(selected.cv_results_['mean_test_macro_f1'][index])}
            pd.DataFrame(selected.cv_results_).to_csv(out/(name+'_cv.csv'),index=False)
            write(path,summary)
            if name=='power_cnn':
                pd.DataFrame(selected.best_estimator_.history_).to_csv(out/'cnn_training_history.csv',index=False)
        summaries.append(summary);print(context,name,round(summary['inner_balanced_accuracy'],4),flush=True)
    winner=max(summaries,key=lambda r:(r['inner_balanced_accuracy'],r['inner_macro_f1']))
    write(out/'selection.json',{'winner':winner,'families':summaries})
    bank,model={n:(b,e) for n,b,e,_ in candidates()}[winner['name']]
    estimator=clone(model).set_params(**winner['parameters']).fit(banks[bank][rows],y[rows])
    return winner,estimator


def run(args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    x,wm=binary_data(args.data,args.manifest);wm=wm.rename(columns={'original_split':'split'})
    plan={'source_sha256':fingerprint(),'window_sha256':digest(x),
        'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        'python':platform.python_version(),'libraries':{n:version(n) for n in ['numpy','scipy','scikit-learn','torch','pandas','joblib']},
        'outer_folds':5,'inner_folds':3,'seed':42,'threads':4,'classes':[6,8],
        'units':['recording','window'],'prototype_target_balanced_accuracy':.7,
        'only_original_train_used_for_fitting_and_search':True,'original_holdouts_not_evaluated':True,
        'all_results_exploratory':True,'subject_independent':False,
        'later_method_batch_after_initial_train_only_comparison':True,
        'selection':'inner grouped balanced accuracy, then macro-F1, then declared order',
        'candidates':[{'name':n,'bank':b,'estimator':repr(e),'grid':g,'configurations':len(list(ParameterGrid(g)))} for n,b,e,g in candidates()],
        'cnn':{'epochs':50,'fixed_duration':True,'temporal_filters':8,'spatial_filters':16,'dropout':.5,
            'learning_rate':.001,'weight_decay':.01,'augmentation':'training only; shared circular shifts +-8 samples and Gaussian noise .02',
            'normalization':'within each example/channel','early_stopping':False},
        'high_frequency':'0.5..60Hz; 50Hz notch Q30; spectral features exclude 49..51Hz',
        'connectivity':'PLV, wPLI, imaginary coherence, envelope correlations; independently per example',
        'lagged_correlations':'lags1,2,4,8,16 samples; independently per example'}
    if (root/'plan.json').exists() and json.loads((root/'plan.json').read_text())!=plan:
        raise ValueError('Frozen experiment differs; use a new output')
    write(root/'plan.json',plan)
    write(root/'raw_audit.json',audit_shared_samples(x,wm))
    recordings,rm=load_recordings(args.data,wm)
    wm.to_csv(root/'window_manifest.csv',index=False);rm.to_csv(root/'recording_manifest.csv',index=False)
    for unit,raw,m in [('recording',recordings,rm),('window',x,wm)]:
        directory=root/unit;directory.mkdir(exist_ok=True);cache=directory/'features.joblib'
        if cache.exists():
            saved=joblib.load(cache)
            assert saved['source_sha']==digest(raw) and saved['code']==fingerprint()
            banks=saved['banks']
        else:
            print('Computing',unit,'signal banks',flush=True)
            banks=signal_banks(raw)
            assert all(np.isfinite(a).all() for a in banks.values())
            joblib.dump({'banks':banks,'source_sha':digest(raw),'code':fingerprint()},cache,compress=1)
        y=m.label.to_numpy();training=np.flatnonzero(m.split=='train')
        predictions=m.iloc[training][['overlap_group','label']].copy()
        predictions['outer_fold']=-1;predictions['prediction']=-1;predictions['majority']=-1
        history=[];members=[]
        for fold,(fit,val) in enumerate(folds(m,training,5,42)):
            a,b=training[fit],training[val];inner=folds(m,a,3,100+fold)
            members.append({'outer_fold':fold,'fit_groups':sorted(m.overlap_group.iloc[a].unique().tolist()),
                'validation_groups':sorted(m.overlap_group.iloc[b].unique().tolist()),
                'inner':[{'fit_groups':sorted(m.overlap_group.iloc[a[i]].unique().tolist()),
                          'validation_groups':sorted(m.overlap_group.iloc[a[j]].unique().tolist())} for i,j in inner]})
            write(directory/'cv_membership.json',members)
            winner,model=search(banks,y,a,inner,directory/('outer_'+str(fold)),unit+' fold'+str(fold))
            predicted=model.predict(banks[winner['bank']][b]);majority=int(pd.Series(y[a]).mode().iloc[0])
            predictions.iloc[val,predictions.columns.get_loc('prediction')]=predicted
            predictions.iloc[val,predictions.columns.get_loc('outer_fold')]=fold
            predictions.iloc[val,predictions.columns.get_loc('majority')]=majority
            history.append({'fold':fold,'winner':winner,**score(y[b],predicted)})
            print('NESTED',unit,fold,score(y[b],predicted),flush=True)
        assert (predictions.prediction!=-1).all()
        predictions.to_csv(directory/'nested_predictions.csv',index=False)
        write(directory/'nested_results.json',{'unit':unit,'duration_seconds':raw.shape[1]/128,
            'examples':len(training),'independent_overlap_groups':m.overlap_group.iloc[training].nunique(),
            'nested':score(predictions.label,predictions.prediction),'majority':score(predictions.label,predictions.majority),
            'folds':history,'exploratory':True,'subject_independent':False,'original_validation_test_evaluated':False})
        winner,model=search(banks,y,training,folds(m,training,3,142),directory/'final_search',unit+' final')
        exported=SignalModel(model,winner['bank'],raw.shape[1])
        np.testing.assert_array_equal(exported.predict(raw[training[:2]]),model.predict(banks[winner['bank']][training[:2]]))
        joblib.dump(exported,directory/'best_model.joblib')
        write(directory/'best_model.json',{'winner':winner,'samples':raw.shape[1],'class_ids':[6,8],
            'training_groups':sorted(m.overlap_group.iloc[training].unique().tolist()),'raw_inference_checked':True})
        print('COMPLETE',unit,score(predictions.label,predictions.prediction),flush=True)
    write(root/'search_completed.json',{'numerical_source_sha256':fingerprint(),'original_holdouts_not_evaluated':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset')
    parser.add_argument('--manifest',default='artifacts/manifest.csv')
    parser.add_argument('--out',default='artifacts_signal_models')
    args=parser.parse_args()
    with threadpool_limits(limits=4):run(args)


if __name__=='__main__':main()
