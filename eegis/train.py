"""Reproducible, grouped EEGIS experiment. Run: python -m eegis.train --help."""
import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from .data import LABELS,CHANNELS,load_data,make_split,audit_shared_samples,preprocess,features
from .models import EEGNet


def scores(y,p):
    return {'accuracy':float(accuracy_score(y,p)), 'macro_f1':float(f1_score(y,p,labels=range(9),average='macro',zero_division=0))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset')
    ap.add_argument('--out',default='artifacts')
    ap.add_argument('--subject-map',help='Optional verified CSV: path,subject_id; paths relative to dataset root')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--epochs',type=int,default=80)
    ap.add_argument('--patience',type=int,default=15)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    start=time.time()
    print('Inspecting every CSV and loading raw EEG...',flush=True)
    x,m=load_data(args.data,out)
    m=make_split(m,args.subject_map,args.seed)
    audit=audit_shared_samples(x,m)
    m.to_csv(out/'manifest.csv',index=False)
    (out/'leakage_audit.json').write_text(json.dumps(audit,indent=2))
    ix={s:np.flatnonzero(m.split.values==s) for s in ['train','validation','test']}
    y=m.label.to_numpy()
    z=preprocess(x);f=features(z)
    print('Partitions:',{s:len(i) for s,i in ix.items()},'features:',f.shape,flush=True)
    tr,va,te=[ix[s] for s in ['train','validation','test']]
    candidates={
      'SVM': [dict(C=c,gamma=g) for c in [0.1,1,10,100] for g in ['scale',.001]],
      'RandomForest':[dict(n_estimators=300,max_depth=d,min_samples_leaf=l) for d in [None,15] for l in [1,3]]}
    selected={};validation={};search=[]
    for name,configs in candidates.items():
        best=-1
        for cfg in configs:
            if name=='SVM':model=make_pipeline(StandardScaler(),SVC(**cfg,class_weight='balanced',random_state=args.seed))
            else:model=RandomForestClassifier(**cfg,class_weight='balanced',random_state=args.seed,n_jobs=4)
            t=time.time();model.fit(f[tr],y[tr]);sc=scores(y[va],model.predict(f[va]))
            search.append({'model':name,'params':json.dumps(cfg),'seconds':time.time()-t,**sc})
            print(name,cfg,sc,flush=True)
            if sc['macro_f1']>best:
                best=sc['macro_f1'];selected[name]=model;validation[name]={**sc,'parameters':cfg}
        joblib.dump(selected[name],out/f'{name}.joblib')
    pd.DataFrame(search).to_csv(out/'baseline_search.csv',index=False)
    # Normalize CNN with training examples only; retain absolute amplitude variation.
    mean=z[tr].mean(axis=(0,1),keepdims=True);std=z[tr].std(axis=(0,1),keepdims=True).clip(1e-6)
    a=((z-mean)/std).transpose(0,2,1)[:,None].copy()
    xt=torch.from_numpy(a);yt=torch.from_numpy(y).long()
    loader=DataLoader(TensorDataset(xt[tr],yt[tr]),batch_size=64,shuffle=True,
        generator=torch.Generator().manual_seed(args.seed))
    net=EEGNet();opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001)
    weights=len(tr)/(9*np.bincount(y[tr],minlength=9))
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
    best=-1;state=None;history=[];stale=0
    for epoch in range(1,args.epochs+1):
        net.train();total=0;correct=0
        for xb,yb in loader:
            opt.zero_grad();logits=net(xb);loss=loss_fn(logits,yb);loss.backward();opt.step()
            total+=loss.item()*len(yb);correct+=(logits.argmax(1)==yb).sum().item()
        net.eval()
        with torch.no_grad():
            logits=net(xt[va]);vl=loss_fn(logits,yt[va]).item();vp=logits.argmax(1).numpy()
        sc=scores(y[va],vp)
        history.append({'epoch':epoch,'train_loss':total/len(tr),'train_accuracy':correct/len(tr),
                        'validation_loss':vl,'validation_accuracy':sc['accuracy'],'validation_macro_f1':sc['macro_f1']})
        if sc['macro_f1']>best:
            best=sc['macro_f1'];state=copy.deepcopy(net.state_dict());stale=0
            validation['EEGNet']={**sc,'parameters':{'epoch':epoch,'learning_rate':.001,'dropout':.35}}
        else:stale+=1
        print(f'EEGNet epoch {epoch}: {history[-1]}',flush=True)
        if stale>=args.patience:break
    net.load_state_dict(state);net.eval()
    torch.save({'state_dict':state,'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),
                'channels':CHANNELS,'labels':LABELS},out/'EEGNet.pt')
    h=pd.DataFrame(history);h.to_csv(out/'training_history.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for col in ['train_loss','validation_loss']:axes[0].plot(h.epoch,h[col],label=col)
    for col in ['train_accuracy','validation_accuracy','validation_macro_f1']:axes[1].plot(h.epoch,h[col],label=col)
    for ax in axes:ax.legend();ax.set_xlabel('Epoch')
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
    # Freeze selection on validation before evaluating any test predictions.
    winner=max(validation,key=lambda k:validation[k]['macro_f1'])
    (out/'selection.json').write_text(json.dumps({'criterion':'validation macro-F1','winner':winner,'validation':validation},indent=2))
    results={};predictions=m.loc[te,['path','label','overlap_group']].copy()
    for name in ['SVM','RandomForest','EEGNet']:
        if name=='EEGNet':
            with torch.no_grad():p=net(xt[te]).argmax(1).numpy()
        else:p=selected[name].predict(f[te])
        predictions[name]=p
        results[name]={'validation':validation[name], 'test':scores(y[te],p),
             'classification_report':classification_report(y[te],p,labels=range(9),target_names=LABELS,output_dict=True,zero_division=0)}
        cm=confusion_matrix(y[te],p,labels=range(9));np.savetxt(out/f'{name}_confusion.csv',cm,delimiter=',',fmt='%d')
        fig,ax=plt.subplots(figsize=(8,7));ConfusionMatrixDisplay(cm,display_labels=LABELS).plot(ax=ax,xticks_rotation=45,colorbar=False)
        ax.set_title(f'{name} — held-out test windows');fig.tight_layout();fig.savefig(out/f'{name}_confusion.png',dpi=160);plt.close(fig)
        print('TEST',name,results[name]['test'],flush=True)
    predictions.to_csv(out/'test_predictions.csv',index=False)
    # Non-trainable majority baseline provides context without choosing a test-driven model.
    majority=int(np.bincount(y[tr]).argmax())
    report={'seed':args.seed,'evaluation':'subject-independent' if args.subject_map else 'overlap-component holdout; participants unidentified',
      'split_sizes':{s:len(i) for s,i in ix.items()},'split_groups':m.groupby('split').overlap_group.nunique().to_dict(),
      'feature_count':f.shape[1], 'best_model':winner,'selection_criterion':'validation macro-F1',
      'majority_baseline':scores(y[te],np.full(len(te),majority)), 'models':results,
      'elapsed_seconds':time.time()-start,'versions':{'numpy':np.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__},
      'arguments':vars(args)}
    (out/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    ext='pt' if winner=='EEGNet' else 'joblib'
    shutil.copyfile(out/f'{winner}.{ext}',out/f'best_model.{ext}')
    (out/'best_model.json').write_text(json.dumps({'model':winner,'file':f'best_model.{ext}',
        'preprocessing':'eegis.data.preprocess','features':'eegis.data.features' if winner!='EEGNet' else None,
        'channels':CHANNELS,'labels':LABELS,'sample_rate_hz':128,'input_shape':[128,14]},indent=2,ensure_ascii=False))
    table=pd.DataFrame([{'model':name,'validation_macro_f1':r['validation']['macro_f1'],**r['test']} for name,r in results.items()])
    table.to_csv(out/'comparison.csv',index=False)
    ax=table.set_index('model')[['accuracy','macro_f1']].plot.bar(rot=0,ylim=(0,1),ylabel='Held-out test score')
    ax.figure.tight_layout();ax.figure.savefig(out/'comparison.png',dpi=160);plt.close(ax.figure)
    print('Selected by validation:',winner,'Elapsed:',report['elapsed_seconds'],flush=True)

if __name__=='__main__':main()
