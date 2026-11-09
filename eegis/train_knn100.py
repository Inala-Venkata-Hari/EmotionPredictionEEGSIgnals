"""Full kNN GA searches: paper-style diagnostic and frozen grouped holdout.

Both searches finish and masks are frozen before either final evaluation.
Every completed generation is resumable with data/code/configuration guards.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from .data import audit_shared_samples, digest
from .diagnose_leakage import leakage_counts
from .knn_search import resumable_search, selected_data
from .paper_equations import COLUMNS, EquationConfig, specification
from .paper_ga import CLASS_NAMES, genome_columns
from .paper_protocol import EquationModel, score, stage_splits
from .train_paper_binary import array_sha, binary_data, dump, split_binary


def fingerprint():
    names=['knn_search.py','train_knn100.py','paper_protocol.py','paper_equations.py',
           'paper_ga.py','paper_features.py','data.py','epochs.py','train_paper_binary.py']
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}


def load_bank(feature_run,data,manifest):
    root=Path(feature_run)
    audit=json.loads((root/'feature_audit.json').read_text())
    spec=json.loads((root/'feature_spec.json').read_text())
    config=EquationConfig(**spec['config'])
    if specification(config)!=spec:
        raise ValueError('Feature configuration or numerical library versions differ')
    for name in ['paper_equations.py','paper_features.py','data.py','epochs.py']:
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=audit['code_sha256'][name]:
            raise ValueError('Feature source differs from frozen cache: '+name)
    x,source=binary_data(data,manifest)
    pd.testing.assert_frame_equal(source,pd.read_csv(root/'source_manifest.csv'))
    if digest(x)!=audit['source_sha256']:
        raise ValueError('Raw data differs from cached feature source')
    with np.load(audit['cache_path'],allow_pickle=False) as cache:
        bank=cache['features']
        if array_sha(bank)!=audit['array_sha256'] or str(cache['key'])!=audit['cache_key']:
            raise ValueError('Invalid feature cache')
    return x,source,bank,config,audit


def partitions(source,seed=42,validation_seed=43):
    labels=source.label.to_numpy()
    fit,val,train,test=stage_splits(labels,seed,validation_seed)
    _,outer,heldout,inner,fitness=split_binary(source,'grouped',seed)
    return {'paper':{'fit':fit,'fitness':val,'final_fit':train,'final_evaluation':test,
                     'search_rows':np.arange(len(source))},
            'grouped':{'fit':inner,'fitness':fitness,'final_fit':outer,'final_evaluation':heldout,
                       'search_rows':outer}}


def run(args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    x,source,bank,config,audit=load_bank(args.feature_run,args.data,args.manifest)
    split=partitions(source,args.seed,args.validation_seed)
    plan={'generations':args.generations,'population':args.population,'seed':args.seed,
        'validation_seed':args.validation_seed,'modes':['paper','grouped'],
        'feature_run':str(args.feature_run),'feature_array_sha256':audit['array_sha256'],
        'raw_source_sha256':audit['source_sha256'],'feature_config':config.__dict__,
        'source_code_sha256':fingerprint(),'normalization':'within-window','undefined_policy':'median',
        'knn_backend':'independent default sklearn prediction for each of seven k values',
        'paper_final_k':9,'grouped_final_k':'selected by inner GA fitness',
        'selection_policy':'freeze both GA outputs before any final evaluation; no seed/normalization/threshold search',
        'primary_model':'grouped','subject_independent':False,'grouped_test_previously_reported':True,
        'interpretation':'paper score is diagnostic; grouped holdout excludes exact EEG overlap but is exploratory',
        'threadpool_limit':4,
        'partitions':{mode:{key:list(map(int,value)) for key,value in item.items()} for mode,item in split.items()}}
    plan_path=root/'study_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('Frozen study configuration/data/code changed; use a new output directory')
    dump(plan_path,plan)
    study_sha=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    labels,groups=source.label.to_numpy(),source.overlap_group.to_numpy()
    selected={}
    for mode,item in split.items():
        out=root/mode;out.mkdir(exist_ok=True)
        manifest=source.copy();manifest['search_role']='excluded'
        manifest.loc[item['fit'],'search_role']='fit'
        manifest.loc[item['fitness'],'search_role']='fitness'
        manifest['final_role']='excluded'
        manifest.loc[item['final_fit'],'final_role']='fit'
        manifest.loc[item['final_evaluation'],'final_role']='evaluation'
        manifest.to_csv(out/'manifest.csv',index=False)
        overlap={'search':leakage_counts(x,groups,item['fit'],item['fitness']),
                 'final':leakage_counts(x,groups,item['final_fit'],item['final_evaluation'])}
        if mode=='grouped':
            if any(value for counts in overlap.values() for value in counts.values()):
                raise ValueError('Grouped study contains EEG overlap')
            m=manifest.copy();m['split']=m.final_role
            audit_shared_samples(x,m)
            inner=manifest.iloc[item['search_rows']].copy();inner['split']=inner.search_role
            audit_shared_samples(x[item['search_rows']],inner)
        dump(out/'overlap_audit.json',overlap)
        rows=item['search_rows']
        fit_local=np.flatnonzero(np.isin(rows,item['fit']))
        val_local=np.flatnonzero(np.isin(rows,item['fitness']))
        best,history,states=resumable_search(bank[rows],labels[rows],fit_local,val_local,out,
            generations=args.generations,population_size=args.population,seed=args.seed,
            guard={'study_sha256':study_sha,'mode':mode},
            progress=lambda line:print(mode+': '+line,flush=True))
        evaluations=best.pop('evaluations')
        pd.DataFrame([{'genome':''.join(map(str,row['genome'])),'accuracy':row['accuracy'],
            'macro_f1':row['macro_f1'],'k':row['k']} for row in evaluations]).to_csv(out/'evaluations.csv',index=False)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        np.savez_compressed(out/'populations.npz',genomes=np.array([row['genomes'] for row in states]),
                            accuracies=np.array([row['accuracies'] for row in states]))
        best['flattened_values']=len(genome_columns(best['genome']))
        best['feature_columns']=[name for name,bit in zip(COLUMNS,best['genome'][14:]) if bit]
        dump(out/'selection.json',best)
        selected[mode]=best
    # This barrier is written only after both searches are complete.
    dump(root/'frozen_selections.json',{'study_sha256':study_sha,'frozen_before_final_predictions':True,
                                      'selections':selected})
    records=[]
    for mode,item in split.items():
        out=root/mode;best=selected[mode];genome=np.asarray(best['genome'],dtype=np.uint8)
        transform,z=selected_data(bank,genome,item['final_fit'])
        k=9 if mode=='paper' else best['k']
        estimator=KNeighborsClassifier(n_neighbors=k).fit(z[item['final_fit']],labels[item['final_fit']])
        model=EquationModel(genome,transform,estimator,config)
        predicted=model.predict_features(bank[item['final_evaluation']])
        majority=int(pd.Series(labels[item['final_fit']]).mode().iloc[0])
        scores=score(labels[item['final_evaluation']],predicted)
        baseline=score(labels[item['final_evaluation']],np.full(len(predicted),majority))
        result={'mode':mode,'fitness_accuracy':best['accuracy'],'final':scores,'majority':baseline,
            'k':k,'selected_search_k':best['k'],'channels':int(genome[:14].sum()),
            'feature_types':int(genome[14:].sum()),'flattened_values':best['flattened_values'],
            'diagnostic_only':mode=='paper','exploratory':True,'subject_independent':False,
            'final_fit_windows':len(item['final_fit']),'evaluation_windows':len(predicted),
            'test_previously_reported':True,'all_parameters_frozen_before_evaluation':True,
            'report':classification_report(labels[item['final_evaluation']],predicted,labels=[6,8],
                target_names=CLASS_NAMES,output_dict=True,zero_division=0)}
        dump(out/'results.json',result)
        joblib.dump(model,out/'model.joblib')
        if mode=='grouped':
            joblib.dump(model,root/'best_model.joblib')
        p=source.iloc[item['final_evaluation']][['path','label','overlap_group']].copy()
        p['prediction'],p['majority']=predicted,majority
        p.to_csv(out/'predictions.csv',index=False)
        np.savetxt(out/'confusion.csv',confusion_matrix(labels[item['final_evaluation']],predicted,labels=[6,8]),fmt='%d',delimiter=',')
        records.append({'mode':mode,'fitness_accuracy':best['accuracy'],
            'accuracy':scores['accuracy'],'macro_f1':scores['macro_f1'],
            'majority_accuracy':baseline['accuracy'],'k':k,'diagnostic_only':mode=='paper',
            'evaluation_windows':len(predicted)})
        print(mode+' FINAL MEASURED '+json.dumps(records[-1]),flush=True)
    pd.DataFrame(records).to_csv(root/'comparison.csv',index=False)
    dump(root/'best_model.json',{'mode':'grouped','file':'best_model.joblib',
        'selection':'predeclared grouped model; inner fitness selected genome/k; final scores never select a model',
        'feature_config':config.__dict__,'input':'raw EEG windows (N,128,14)',
        'class_ids':[6,8],'exploratory':True,'subject_independent':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset')
    parser.add_argument('--manifest',default='artifacts/manifest.csv')
    parser.add_argument('--feature-run',default='artifacts_paper_method/smoke')
    parser.add_argument('--out',default='artifacts_paper_method/knn100')
    parser.add_argument('--generations',type=int,default=100)
    parser.add_argument('--population',type=int,default=100)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--validation-seed',type=int,default=43)
    args=parser.parse_args()
    if args.generations<1 or args.population<2 or args.population%2:
        parser.error('Require positive generations and an even population>=2')
    with threadpool_limits(limits=4):
        run(args)


if __name__=='__main__':
    main()
