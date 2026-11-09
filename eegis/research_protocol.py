"""Reusable fixed nested-group protocol for additional binary research batches."""
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterGrid

from .data import audit_shared_samples,digest
from .epochs import load_recordings
from .train_accuracy import choose_index,folds,score,write
from .train_paper_binary import binary_data


def fingerprint(names):
    return {n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}


def search(spec,banks,y,rows,cv,out,context):
    out.mkdir(parents=True,exist_ok=True);summaries=[]
    for name,bank,model,grid in spec['candidates']():
        path=out/(name+'.json')
        if path.exists():value=json.loads(path.read_text())
        else:
            print(context,'fitting',name,flush=True)
            fit=GridSearchCV(model,grid,scoring={'balanced_accuracy':'balanced_accuracy','macro_f1':'f1_macro'},
                refit=choose_index,cv=cv,n_jobs=1,error_score='raise').fit(banks[bank][rows],y[rows])
            i=fit.best_index_
            value={'name':name,'bank':bank,'parameters':fit.best_params_,
                'inner_balanced_accuracy':float(fit.cv_results_['mean_test_balanced_accuracy'][i]),
                'inner_macro_f1':float(fit.cv_results_['mean_test_macro_f1'][i])}
            pd.DataFrame(fit.cv_results_).to_csv(out/(name+'_cv.csv'),index=False);write(path,value)
        summaries.append(value);print(context,name,round(value['inner_balanced_accuracy'],4),flush=True)
    selected=max(summaries,key=lambda r:(r['inner_balanced_accuracy'],r['inner_macro_f1']))
    write(out/'selection.json',{'winner':selected,'families':summaries})
    bank,model={n:(b,e) for n,b,e,_ in spec['candidates']()}[selected['name']]
    return selected,clone(model).set_params(**selected['parameters']).fit(banks[bank][rows],y[rows])


def run(spec,args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    windows,wm=binary_data(args.data,args.manifest);wm=wm.rename(columns={'original_split':'split'})
    plan={'study_kind':spec['name'],'source_sha256':fingerprint(spec['dependencies']),
        'window_sha256':digest(windows),'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        'python':platform.python_version(),'libraries':{n:version(n) for n in ['numpy','scipy','scikit-learn','pandas','joblib']},
        'outer_folds':5,'inner_folds':3,'seed':42,'threads':4,'classes':[6,8],
        'original_holdouts_not_evaluated':True,'only_original_train_used_for_fitting_and_search':True,
        'all_results_exploratory':True,'subject_independent':False,'prototype_target_balanced_accuracy':.7,
        'selection':'inner grouped balanced accuracy, then macro-F1, then declared order',
        'methods':spec['description'],'candidates':[{'name':n,'bank':b,'estimator':repr(e),'grid':g,
            'configurations':len(list(ParameterGrid(g)))} for n,b,e,g in spec['candidates']()]}
    if (root/'plan.json').exists() and json.loads((root/'plan.json').read_text())!=plan:raise ValueError('Frozen study differs')
    write(root/'plan.json',plan);write(root/'raw_audit.json',audit_shared_samples(windows,wm))
    recordings,rm=load_recordings(args.data,wm)
    wm.to_csv(root/'window_manifest.csv',index=False);rm.to_csv(root/'recording_manifest.csv',index=False)
    for unit,x,m in [('recording',recordings,rm),('window',windows,wm)]:
        out=root/unit;out.mkdir(exist_ok=True);path=out/'features.joblib'
        if path.exists():
            cache=joblib.load(path);assert cache['code']==plan['source_sha256'] and cache['source_sha']==digest(x)
            banks=cache['banks']
        else:
            print('Computing',unit,'features',flush=True);banks=spec['features'](x)
            assert all(np.isfinite(b).all() for b in banks.values())
            joblib.dump({'banks':banks,'code':plan['source_sha256'],'source_sha':digest(x)},path,compress=1)
        y=m.label.to_numpy();training=np.flatnonzero(m.split=='train')
        predictions=m.iloc[training][['overlap_group','label']].copy()
        predictions['outer_fold']=-1;predictions['prediction']=-1;predictions['majority']=-1
        members=[];history=[]
        for fold,(fit,val) in enumerate(folds(m,training,5,42)):
            a,b=training[fit],training[val];inner=folds(m,a,3,100+fold)
            members.append({'outer_fold':fold,'fit_groups':sorted(m.overlap_group.iloc[a].unique().tolist()),
                'validation_groups':sorted(m.overlap_group.iloc[b].unique().tolist()),
                'inner':[{'fit_groups':sorted(m.overlap_group.iloc[a[i]].unique().tolist()),
                    'validation_groups':sorted(m.overlap_group.iloc[a[j]].unique().tolist())} for i,j in inner]})
            write(out/'cv_membership.json',members)
            chosen,model=search(spec,banks,y,a,inner,out/('outer_'+str(fold)),unit+' fold'+str(fold))
            p=model.predict(banks[chosen['bank']][b]);majority=int(pd.Series(y[a]).mode().iloc[0])
            predictions.iloc[val,predictions.columns.get_loc('prediction')]=p
            predictions.iloc[val,predictions.columns.get_loc('majority')]=majority
            predictions.iloc[val,predictions.columns.get_loc('outer_fold')]=fold
            history.append({'fold':fold,'winner':chosen,**score(y[b],p)});print('NESTED',unit,fold,score(y[b],p),flush=True)
        assert (predictions.prediction!=-1).all();predictions.to_csv(out/'nested_predictions.csv',index=False)
        write(out/'nested_results.json',{'unit':unit,'duration_seconds':x.shape[1]/128,'examples':len(training),
            'independent_overlap_groups':m.overlap_group.iloc[training].nunique(),'nested':score(predictions.label,predictions.prediction),
            'majority':score(predictions.label,predictions.majority),'folds':history,'exploratory':True,
            'subject_independent':False,'original_validation_test_evaluated':False})
        chosen,model=search(spec,banks,y,training,folds(m,training,3,142),out/'final_search',unit+' final')
        exported=spec['factory'](model,chosen['bank'],x.shape[1])
        np.testing.assert_array_equal(exported.predict(x[training[:2]]),model.predict(banks[chosen['bank']][training[:2]]))
        joblib.dump(exported,out/'best_model.joblib')
        write(out/'best_model.json',{'winner':chosen,'samples':x.shape[1],'class_ids':[6,8],
            'training_groups':sorted(m.overlap_group.iloc[training].unique().tolist()),'raw_inference_checked':True})
        print('COMPLETE',unit,score(predictions.label,predictions.prediction),flush=True)
    write(root/'search_completed.json',{'numerical_source_sha256':plan['source_sha256'],'original_holdouts_not_evaluated':True})
