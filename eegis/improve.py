"""Spatial covariance experiment; same fixed split, training-group CV selection."""
import argparse
import ast
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from threadpoolctl import threadpool_limits
from .data import CHANNELS,LABELS,preprocess,features,audit_shared_samples


def covariance_input(x):
    """Demean, filter independently, and regularize channel covariance matrices."""
    z=signal.detrend(x,axis=1,type='linear')
    cov=[]
    for band in [None,(4,8),(8,13),(13,30),(30,40)]:
        a=z if band is None else signal.sosfiltfilt(signal.butter(3,band,btype='bandpass',fs=128,output='sos'),z,axis=1)
        a=a-a.mean(axis=1,keepdims=True)
        c=np.einsum('ntc,ntd->ncd',a,a)/127
        scale=np.trace(c,axis1=1,axis2=2)/14
        c=.9*c+.1*scale[:,None,None]*np.eye(14)[None]
        cov.append(c)
    return np.stack(cov,axis=1)

class TangentFeatures(TransformerMixin,BaseEstimator):
    """Train-fitted covariance whitening followed by symmetric matrix logarithm."""
    def __init__(self,mode='tangent',normalize=False):
        self.mode=mode;self.normalize=normalize
    def _norm(self,x):
        return x/np.trace(x,axis1=-2,axis2=-1)[...,None,None].clip(1e-12) if self.normalize else x
    def fit(self,x,y=None):
        x=self._norm(x)
        mean=x.mean(axis=0)
        w,v=np.linalg.eigh(mean)
        self.whitener_=np.einsum('bik,bk,bjk->bij',v,1/np.sqrt(w.clip(1e-12)),v)
        return self
    def transform(self,x):
        x=self._norm(x)
        if self.mode=='tangent':
            x=np.einsum('bik,nbkl,blj->nbij',self.whitener_,x,self.whitener_)
        w,v=np.linalg.eigh(x)
        log=np.einsum('nbik,nbk,nbjk->nbij',v,np.log(w.clip(1e-12)),v)
        i,j=np.triu_indices(14)
        return (log[:,:,i,j]*np.where(i==j,1,np.sqrt(2))).reshape(len(x),-1)


def score(y,p):return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro',labels=range(9),zero_division=0))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset');ap.add_argument('--out',default='artifacts_improved');ap.add_argument('--resume',action='store_true',help='Reuse completed family searches only with the unchanged dataset and split');args=ap.parse_args()
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True);start=time.time()
    m=pd.read_csv('artifacts/manifest.csv')
    if args.resume and (out/'manifest.csv').exists():
        pd.testing.assert_frame_equal(m,pd.read_csv(out/'manifest.csv'))
    m.to_csv(out/'manifest.csv',index=False)
    data=Path(args.data)
    x=np.stack([pd.read_csv(data/p).iloc[:,1:].to_numpy(dtype=np.float64) for p in m.path])
    audit=audit_shared_samples(x,m);(out/'leakage_audit.json').write_text(json.dumps(audit,indent=2))
    ix={s:np.flatnonzero(m.split==s) for s in ['train','validation','test']};tr,va,te=[ix[s] for s in ix]
    y=m.label.to_numpy();g=m.overlap_group.to_numpy()
    cov=covariance_input(x);f=features(preprocess(x))
    cv=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(tr,y[tr],g[tr]))
    # Freeze the search space before examining new test predictions.
    specs=[
      ('TangentSVM',cov,Pipeline([('spatial',TangentFeatures()),('scale',StandardScaler()),('model',SVC())]),
       {'spatial__normalize':[False,True],'model__C':[.01,.1,1,10],'model__kernel':['rbf']}),
      ('TangentRidge',cov,Pipeline([('spatial',TangentFeatures()),('scale',StandardScaler()),('model',RidgeClassifier())]),
       {'spatial__normalize':[False,True],'model__alpha':[.1,1,10,100,1000]}),
      ('TangentLogistic',cov,Pipeline([('spatial',TangentFeatures()),('scale',StandardScaler()),('model',LogisticRegression(max_iter=2000))]),
       {'spatial__normalize':[False,True],'model__C':[.001,.01,.1,1,10]}),
      ('LogCovSVM',cov,Pipeline([('spatial',TangentFeatures(mode='log')),('scale',StandardScaler()),('model',SVC())]),
       {'spatial__normalize':[False,True],'model__C':[.01,.1,1,10],'model__kernel':['rbf']}),
      ('ExtraTrees',f,ExtraTreesClassifier(random_state=42,n_jobs=4,n_estimators=500),
       {'max_features':['sqrt',.5,1.0],'min_samples_leaf':[1,4,12]}),
      ('BoostedTrees',f,HistGradientBoostingClassifier(random_state=42,max_iter=150,early_stopping=False),
       {'max_leaf_nodes':[7,15],'l2_regularization':[1.,10.],'learning_rate':[.03,.1]})]
    results={};models={};inputs={};rows=[]
    with threadpool_limits(limits=4):
        for name,a,model,grid in specs:
            print('Searching',name,flush=True)
            if args.resume and (out/f'{name}.joblib').exists() and (out/f'{name}_cv.csv').exists():
                est=joblib.load(out/f'{name}.joblib')
                cached=pd.read_csv(out/f'{name}_cv.csv');best=cached.loc[cached.mean_test_accuracy.idxmax()]
                best_score=float(best.mean_test_accuracy);params=ast.literal_eval(best.params)
                print('Resuming completed family',name,flush=True)
            else:
                search=GridSearchCV(model,grid,scoring={'accuracy':'accuracy','macro_f1':'f1_macro'},refit='accuracy',cv=cv,n_jobs=1,error_score='raise',verbose=2)
                search.fit(a[tr],y[tr])
                pd.DataFrame(search.cv_results_).to_csv(out/f'{name}_cv.csv',index=False)
                est=search.best_estimator_;best_score=float(search.best_score_);params=search.best_params_
            p=est.predict(a[va]);sc=score(y[va],p)
            results[name]={'validation':sc,'training_cv_accuracy':best_score,'parameters':params}
            models[name]=est;inputs[name]=a
            joblib.dump(est,out/f'{name}.joblib')
            print(name,results[name],flush=True)
    # User requested accuracy improvement: select on validation accuracy, macro-F1 tie-break.
    winner=max(results,key=lambda n:(results[n]['validation']['accuracy'],results[n]['validation']['macro_f1']))
    selection={'winner':winner,'criterion':'validation accuracy, macro-F1 tie-break','candidates':results}
    (out/'selection.json').write_text(json.dumps(selection,indent=2))
    # Do not alter the winner using held-out scores.
    preds=m.loc[te,['path','label','overlap_group']].copy()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    for name,model in models.items():
        p=model.predict(inputs[name][te]);preds[name]=p
        results[name]['test']=score(y[te],p)
        results[name]['classification_report']=classification_report(y[te],p,labels=range(9),target_names=LABELS,output_dict=True,zero_division=0)
        cm=confusion_matrix(y[te],p,labels=range(9));np.savetxt(out/f'{name}_confusion.csv',cm,delimiter=',',fmt='%d')
        fig,ax=plt.subplots(figsize=(8,7));ConfusionMatrixDisplay(cm,display_labels=LABELS).plot(ax=ax,xticks_rotation=45,colorbar=False);ax.set_title(name);fig.tight_layout();fig.savefig(out/f'{name}_confusion.png',dpi=160);plt.close(fig)
        rows.append({'model':name,**results[name]['test'],'validation_accuracy':results[name]['validation']['accuracy']})
        print('TEST',name,results[name]['test'],flush=True)
    preds.to_csv(out/'test_predictions.csv',index=False)
    table=pd.DataFrame(rows);table.to_csv(out/'comparison.csv',index=False)
    ax=table.set_index('model')[['accuracy','macro_f1']].plot.bar(rot=25,ylim=(0,.5));ax.figure.tight_layout();ax.figure.savefig(out/'comparison.png',dpi=160);plt.close(ax.figure)
    # Store ordinary importable classes, not __main__ pickle references.
    joblib.dump(models[winner],out/'best_model.joblib')
    (out/'best_model.json').write_text(json.dumps({'model':winner,'file':'best_model.joblib','input_transform':'covariance_input' if inputs[winner] is cov else 'features','channels':CHANNELS,'labels':LABELS},indent=2,ensure_ascii=False))
    report={'models':results,'best_model':winner,'selection_criterion':selection['criterion'],'elapsed_seconds':time.time()-start,'split_source':'artifacts/manifest.csv','evaluation':'same overlap-component holdout; not subject-independent; test set previously reported'}
    (out/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('Selected',winner,'elapsed',report['elapsed_seconds'],flush=True)

if __name__=='__main__':
    # Ensure joblib can reload transformers from the stable module path.
    from eegis.improve import main
    main()
