"""Optional multi-window decisions; never presented as one-second accuracy."""
import argparse
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import CHANNELS,LABELS,features,preprocess,overlap_groups
from .improve import covariance_input,score

SPATIAL={'TangentSVM','TangentRidge','TangentLogistic','LogCovSVM'}

def model_scores(model,x,name):
    a=covariance_input(x) if name in SPATIAL else features(preprocess(x))
    if hasattr(model,'predict_proba'):
        return model.predict_proba(a)
    return model.decision_function(a)

def aggregate(scores,groups,y=None):
    ids=np.unique(groups);values=np.stack([scores[groups==g].mean(axis=0) for g in ids])
    labels=None
    if y is not None:
        labels=[]
        for g in ids:
            unique=np.unique(y[groups==g])
            if len(unique)!=1:raise ValueError('Inconsistent labels in overlap group')
            labels.append(unique[0])
        labels=np.array(labels)
    return ids,values.argmax(axis=1),labels


def predict_context(paths,artifacts='artifacts_improved'):
    out=Path(artifacts);meta=json.loads((out/'context_selection.json').read_text());name=meta['winner']
    arrays=[]
    for path in paths:
        a=pd.read_csv(path)
        if a.shape!=(128,15) or list(a.columns[1:])!=CHANNELS:raise ValueError('Invalid EEGIS CSV')
        arrays.append(a.iloc[:,1:].to_numpy(dtype=np.float64))
    x=np.stack(arrays)
    if len(x)!=12 or not np.isfinite(x).all() or len(np.unique(overlap_groups(x)))!=1:
        raise ValueError('Provide exactly 12 finite, connected EEGIS overlap windows')
    # Reject repeats, missing windows, or branched overlap graphs.
    from .data import digest
    if len({digest(a) for a in x})!=12:raise ValueError('Duplicate windows')
    prefix={digest(a[:48]) for a in x};suffix={digest(a[-48:]) for a in x}
    if len(prefix)!=12 or len(suffix)!=12 or len(prefix&suffix)!=11:
        raise ValueError('Windows must form one complete overlap chain')
    scores=model_scores(joblib.load(out/meta['file']),x,name)
    label=int(scores.mean(axis=0).argmax())
    return {'model':name,'class_id':label,'label':LABELS[label],'context_seconds':1008/128}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--artifacts',default='artifacts_improved');ap.add_argument('--predict',nargs='+',help='Exactly 12 CSV paths from one complete overlap chain');args=ap.parse_args()
    if args.predict:
        print(json.dumps(predict_context(args.predict,args.artifacts),ensure_ascii=False));return
    out=Path(args.artifacts);r=json.loads((out/'results.json').read_text());m=pd.read_csv(out/'manifest.csv');data=Path('EEGIS - Electroencephalogram Imagined Speech dataset')
    frames={s:m[m.split==s].copy() for s in ['validation','test']}
    arrays={s:np.stack([pd.read_csv(data/p).iloc[:,1:].to_numpy(dtype=np.float64) for p in frame.path]) for s,frame in frames.items()}
    results={};models={}
    for name in r['models']:
        model=joblib.load(out/f'{name}.joblib');models[name]=model;frame=frames['validation']
        ids,p,y=aggregate(model_scores(model,arrays['validation'],name),frame.overlap_group.values,frame.label.values)
        results[name]={'validation':score(y,p)}
    winner=max(results,key=lambda n:(results[n]['validation']['accuracy'],results[n]['validation']['macro_f1']))
    shutil.copyfile(out/f'{winner}.joblib',out/'best_context_model.joblib')
    (out/'context_selection.json').write_text(json.dumps({'winner':winner,'file':'best_context_model.joblib','criterion':'validation component accuracy, macro-F1 tie-break','context_seconds':7.875,'validation':results},indent=2))
    frame=frames['test'];predictions=None
    for name,model in models.items():
        ids,p,y=aggregate(model_scores(model,arrays['test'],name),frame.overlap_group.values,frame.label.values)
        results[name]['test']=score(y,p)
        if predictions is None:predictions=pd.DataFrame({'overlap_group':ids,'label':y})
        predictions[name]=p
    predictions.to_csv(out/'context_test_predictions.csv',index=False)
    testgroup=predictions.overlap_group.iloc[0]
    paths=[data/p for p in frame.loc[frame.overlap_group==testgroup,'path']]
    assert predict_context(paths,out)['class_id']==int(predictions.iloc[0][winner])
    report={'winner':winner,'models':results,'test_groups':len(predictions),'context_seconds':7.875,
        'task':'one decision per complete overlap component using 12 windows; distinct from one-second classification',
        'single_chain_inference_verified':True,'evaluation':'reused overlap-component holdout; not subject-independent'}
    (out/'context_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
