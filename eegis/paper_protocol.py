"""Paper's binary GA experiment and separate 20/80 classifier experiment.

This is an explicit reconstruction of the published method, with an assumptions
ledger. Scores are diagnostic: overlapping windows and selection-label reuse
prevent an independent generalization claim. Previous experiments are untouched.
"""
import argparse
import hashlib
import json
import time
import warnings
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .data import CHANNELS, digest
from .diagnose_leakage import leakage_counts
from .paper_equations import (COLUMNS, EquationConfig, equation_features, filter_layers,
                              literal_higuchi, specification)
from .paper_ga import CLASSIFIERS, CLASS_IDS, CLASS_NAMES, K_VALUES, genetic_search, genome_columns
from .train_paper_binary import array_sha, binary_data, dump


FACTORIES = {'knn':KNeighborsClassifier,'logistic':LogisticRegression,
    'random_forest':RandomForestClassifier,'decision_tree':DecisionTreeClassifier,
    'gradient_boosting':GradientBoostingClassifier,'adaboost':AdaBoostClassifier,
    'naive_bayes':GaussianNB,'lda':LinearDiscriminantAnalysis,'mlp':MLPClassifier,'svm':SVC}


class EquationTransform:
    """Standardize selected feature columns; missing-value policy is explicit."""
    def __init__(self,channels,features,normalization='within-window',undefined_policy='median'):
        if normalization not in ['within-window','train'] or undefined_policy not in ['error','median']:
            raise ValueError('Invalid normalization or undefined-value policy')
        self.channels,self.features = int(channels),int(features)
        self.normalization,self.undefined_policy = normalization,undefined_policy

    def _check(self,x):
        x = np.asarray(x,dtype=float)
        if x.ndim!=2 or x.shape[1]!=self.channels*self.features or np.isinf(x).any():
            raise ValueError('Invalid selected feature matrix')
        if self.undefined_policy=='error' and np.isnan(x).any():
            raise ValueError('Printed feature expression is undefined; see undefined_features.csv')
        return x

    def fit(self,x,y=None):
        x = self._check(x)
        self.training_missing_values_ = int(np.isnan(x).sum())
        if self.normalization=='train':
            self.imputer_ = SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
            self.scaler_ = StandardScaler().fit(self.imputer_.transform(x))
        return self

    def transform(self,x):
        x = self._check(x)
        if self.normalization=='train':
            return self.scaler_.transform(self.imputer_.transform(x))
        a = x.reshape(-1,self.channels,self.features)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='All-NaN slice encountered')
            medians = np.nanmedian(a,axis=1,keepdims=True)
        # No other example enters these medians. An entirely undefined column
        # receives zero; this is a disclosed numerical extension, not a formula.
        medians = np.where(np.isnan(medians),0,medians)
        a = np.where(np.isnan(a),medians,a)
        columns = a.transpose(1,0,2).reshape(self.channels,-1)
        z = StandardScaler().fit_transform(columns)
        return z.reshape(self.channels,len(a),self.features).transpose(1,0,2).reshape(len(a),-1)


class EquationModel:
    def __init__(self,genome,transform,estimator,config):
        self.genome,self.transform,self.estimator,self.config = np.asarray(genome),transform,estimator,config
        self.classes_ = np.asarray(CLASS_IDS)

    def predict_features(self,features):
        x = features.reshape(len(features),-1)[:,genome_columns(self.genome)]
        return self.estimator.predict(self.transform.transform(x))

    def predict(self,windows):
        return self.predict_features(equation_features(windows,self.config))


def estimator(name,k=7):
    if name not in FACTORIES:
        raise ValueError('Unknown classifier')
    return FACTORIES[name](n_neighbors=k) if name=='knn' else FACTORIES[name]()


def seeded_fit(model,x,y,seed):
    # Keep actual sklearn defaults (random_state=None), while making each fit
    # reproducible through a declared global RNG seed. Do not alter the GA RNG.
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            model.fit(x,y)
    finally:
        np.random.set_state(state)
    model.paper_fit_warnings_ = sorted({str(record.message) for record in records})
    return model


def score(labels,predicted):
    return {'accuracy':float(np.mean(np.asarray(labels)==predicted)),
            'macro_f1':float(f1_score(labels,predicted,labels=CLASS_IDS,average='macro',zero_division=0))}


def phenotype(features,genome,fit,normalization,undefined_policy):
    flat = features.reshape(len(features),-1)[:,genome_columns(genome)]
    transform = EquationTransform(np.sum(genome[:14]),np.sum(genome[14:]),normalization,undefined_policy).fit(flat[fit])
    return transform,transform.transform(flat)


def fitness(features,labels,train,validation,genome,classifier,seed,device,
            normalization='within-window',undefined_policy='median'):
    if device!='cpu':
        raise ValueError('Paper-default sklearn classifiers use CPU; CUDA approximation is not used here')
    _,z = phenotype(features,genome,train,normalization,undefined_policy)
    records = []
    all_warnings = set()
    for k in K_VALUES if classifier=='knn' else [None]:
        try:
            model = seeded_fit(estimator(classifier,k or 7),z[train],labels[train],seed)
            predicted = model.predict(z[validation])
            all_warnings.update(model.paper_fit_warnings_)
            records.append({'k':k,**score(labels[validation],predicted),'fitness_is_penalty':False})
        except (ValueError,IndexError,np.linalg.LinAlgError,FloatingPointError) as exc:
            records.append({'k':k,'accuracy':0.,'macro_f1':0.,'fitness_is_penalty':True,
                            'fit_failure':type(exc).__name__+': '+str(exc)})
    best = max(records,key=lambda row:row['accuracy'])
    return {**best,'all_k_scores':records,'fit_warnings':sorted(all_warnings)}


def stage_splits(labels,seed=42,validation_seed=43):
    indices = np.arange(len(labels))
    train,evaluation = train_test_split(indices,test_size=.2,random_state=seed,stratify=labels)
    later_train,later_evaluation = train_test_split(indices,test_size=.8,random_state=validation_seed,stratify=labels)
    return tuple(np.sort(rows) for rows in [train,evaluation,later_train,later_evaluation])


def code_fingerprint():
    names = ['paper_equations.py','paper_protocol.py','paper_features.py','paper_ga.py',
             'data.py','epochs.py','train_paper_binary.py']
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}


def prepare(args,out,config):
    x,source = binary_data(args.data,args.manifest)
    if len(source)!=924 or source.label.value_counts().to_dict()!={6:480,8:444}:
        raise ValueError('Expected the paper\'s 480 No and 444 Sí windows')
    source.to_csv(out/'source_manifest.csv',index=False)
    cases = []
    for index in [0,479,480,923]:
        for layer,a in enumerate(filter_layers(x[index:index+1])):
            value = literal_higuchi(a[0,:,0],config.kmax)
            cases.append({'source_row':index,'layer':layer,'channel':'AF3',
                          'value':float(value) if np.isfinite(value) else None})
    literal_audit = {'cases':cases,'undefined_cases':sum(row['value'] is None for row in cases),
        'selected_mode':config.higuchi,'literal_is_not_silently_repaired':True}
    dump(out/'literal_higuchi_audit.json',literal_audit)
    if args.strict_literal and literal_audit['undefined_cases']:
        dump(out/'strict_literal_status.json',{'training_executed':False,
            'reason':'A22-A24 require log of nonpositive signed curve lengths on actual EEG',
            'undefined_cases_in_24_declared_checks':literal_audit['undefined_cases']})
        raise ValueError('Literal A22-A24 is undefined on real examples; see literal_higuchi_audit.json. No replacement formula or training was used.')
    spec = specification(config)
    fingerprint = code_fingerprint()
    feature_code = {name:value for name,value in fingerprint.items()
                    if name in ['paper_equations.py','paper_features.py','data.py','epochs.py']}
    key = hashlib.sha256(json.dumps({'source':digest(x),'spec':spec,'code':feature_code},sort_keys=True).encode()).hexdigest()
    cache_dir = Path(args.cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    cache_path = cache_dir/(key+'.npz')
    if cache_path.exists():
        with np.load(cache_path,allow_pickle=False) as cache:
            features = cache['features']
            if str(cache['key'])!=key or str(cache['array_sha'])!=array_sha(features):
                raise ValueError('Feature-cache fingerprint mismatch')
    else:
        features = equation_features(x,config,batch_size=args.batch_size,progress=lambda line:print(line,flush=True))
        np.savez_compressed(cache_path,features=features,key=key,array_sha=array_sha(features))
    if features.shape!=(924,14,214) or np.isinf(features).any():
        raise ValueError('Invalid equation feature bank')
    missing = np.isnan(features).sum(axis=(0,1))
    pd.DataFrame({'column':COLUMNS,'undefined_values':missing}).to_csv(out/'undefined_features.csv',index=False)
    dump(out/'feature_spec.json',spec)
    dump(out/'feature_audit.json',{'shape':list(features.shape),'source_sha256':digest(x),
        'array_sha256':array_sha(features),'cache_key':key,'cache_path':str(cache_path),
        'undefined_values':int(np.isnan(features).sum()),'code_sha256':fingerprint})
    if args.undefined_policy=='error' and np.isnan(features).any():
        raise ValueError('Literal equations contain undefined values; inspect undefined_features.csv. No training metrics were produced.')
    return x,source,features


def run(args):
    out = Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists() and not args.resume:
        raise ValueError('Output already contains an experiment. Use a new --out or --resume.')
    config = EquationConfig(higuchi='literal' if args.strict_literal else args.higuchi,
                           power_low=args.power_low,power_high=args.power_high)
    if args.strict_literal:
        args.undefined_policy = 'error'
    plan = {'classifiers':args.classifiers,'generations':args.generations,'population':args.population,
        'seed':args.seed,'validation_seed':args.validation_seed,'normalization':args.normalization,
        'undefined_policy':args.undefined_policy,'feature_config':asdict(config),
        'stage1':'80% fit / 20% GA fitness, without an additional inner split',
        'stage2':'freeze every GA output, then fit on20% and evaluate on80%; kNN k=9 per Section3',
        'final_knn_k':9,'knn_search':K_VALUES,'diagnostic_only':True,
        'selection':'highest stage1 fitness only; classifier order resolves ties; freeze before stage2',
        'ga_rates':{'mutation':1/228,'uniform_crossover_bit_probability':.5},
        'device':'cpu','threadpool_limit':4,'code_sha256':code_fingerprint(),
        'assumptions':[
            'Original split indices, seed, GA rates and exact library versions are not supplied.',
            'Stratified row splits approximate the stated ratios and permit overlapping EEG.',
            'Within-window feature-column standardization is the default interpretation of MxN maps before flattening.',
            'Missing mathematical values require an explicit error/median policy; this is absent from the paper.',
            'Sklearn estimators retain defaults; a declared global RNG seed makes random_state=None repeatable.',
            'A degenerate genome whose classifier cannot fit receives a declared zero fitness penalty, not a measured accuracy.',
            'Accuracy is standard accuracy. Printed precision TP/(TP+TN) is reported separately from standard metrics.',
        ]}
    if (out/'plan.json').exists() and args.resume:
        if json.loads((out/'plan.json').read_text())!=plan:
            raise ValueError('Resume configuration or source code differs from the saved plan')
    dump(out/'plan.json',plan)
    x,source,features = prepare(args,out,config)
    if args.prepare_only:
        return
    labels,groups = source.label.to_numpy(),source.overlap_group.to_numpy()
    fit,search,later_fit,later_test = stage_splits(labels,args.seed,args.validation_seed)
    manifest = source.copy()
    manifest['stage1'] = 'fit';manifest.loc[search,'stage1'] = 'fitness'
    manifest['stage2'] = 'fit';manifest.loc[later_test,'stage2'] = 'evaluation'
    manifest.to_csv(out/'manifest.csv',index=False)
    dump(out/'split_audit.json',{'counts':{'stage1_fit':len(fit),'stage1_fitness':len(search),
        'stage2_fit':len(later_fit),'stage2_evaluation':len(later_test)},
        'stage1_overlap':leakage_counts(x,groups,fit,search),
        'stage2_overlap':leakage_counts(x,groups,later_fit,later_test),
        'later_evaluation_previously_used_for_fitness_labels':len(np.intersect1d(later_test,search)),
        'diagnostic_only':True,'subject_independent':False})
    selected,histories = {},{}
    started = time.time()
    for name in args.classifiers:
        directory = out/name;directory.mkdir(exist_ok=True)
        if args.resume and (directory/'selection.json').exists():
            selected[name] = json.loads((directory/'selection.json').read_text())
            histories[name] = pd.read_csv(directory/'history.csv').to_dict('records')
            continue
        callback = partial(fitness,normalization=args.normalization,undefined_policy=args.undefined_policy)
        best,history,populations = genetic_search(features,labels,fit,search,classifier=name,
            generations=args.generations,population_size=args.population,seed=args.seed,device='cpu',
            evaluator=callback,history_path=directory/'history.csv',progress=lambda line:print(line,flush=True))
        evaluations = best.pop('evaluations')
        if best.get('fitness_is_penalty'):
            raise ValueError(name+' has no fitted winning model; all best fitness values are failure penalties')
        pd.DataFrame([{'genome':''.join(map(str,row['genome'])),'accuracy':row['accuracy'],
                       'macro_f1':row['macro_f1'],'k':row['k'],
                       'fitness_is_penalty':row.get('fitness_is_penalty',False),
                       'fit_failure':row.get('fit_failure','')} for row in evaluations]).to_csv(directory/'evaluations.csv',index=False)
        np.savez_compressed(directory/'populations.npz',
            genomes=np.array([row['genomes'] for row in populations],dtype=np.uint8),
            accuracies=np.array([row['accuracies'] for row in populations]))
        best['channels'] = [c for c,bit in zip(CHANNELS,best['genome'][:14]) if bit]
        best['feature_columns'] = [c for c,bit in zip(COLUMNS,best['genome'][14:]) if bit]
        best['flattened_values'] = len(genome_columns(best['genome']))
        dump(directory/'selection.json',best)
        selected[name],histories[name] = best,history
    winner = max(args.classifiers,key=lambda name:selected[name]['accuracy'])
    dump(out/'selection.json',{'winner':winner,'frozen_before_stage2':True,
        'criterion':plan['selection'],'stage1_fitness':{name:row['accuracy'] for name,row in selected.items()}})
    results,rows,predictions = {},[],[]
    for name in args.classifiers:
        directory = out/name
        genome = np.asarray(selected[name]['genome'],dtype=np.uint8)
        transform,z = phenotype(features,genome,later_fit,args.normalization,args.undefined_policy)
        k = 9 if name=='knn' else 7
        model = seeded_fit(estimator(name,k),z[later_fit],labels[later_fit],args.seed)
        predicted = model.predict(z[later_test])
        checkpoint = EquationModel(genome,transform,model,config)
        joblib.dump(checkpoint,directory/'model.joblib')
        if name==winner:
            joblib.dump(checkpoint,out/'best_model.joblib')
        values = score(labels[later_test],predicted)
        cm = confusion_matrix(labels[later_test],predicted,labels=CLASS_IDS)
        np.savetxt(directory/'confusion.csv',cm,fmt='%d',delimiter=',')
        tn,fp,fn,tp = cm.ravel()
        printed_precision = float(tp/(tp+tn)) if tp+tn else 0.
        recall = float(tp/(tp+fn)) if tp+fn else 0.
        values.update({'paper_printed_precision_for_Si':printed_precision,
            'paper_printed_f1_for_Si':2*printed_precision*recall/(printed_precision+recall) if printed_precision+recall else 0.,
            'report':classification_report(labels[later_test],predicted,labels=CLASS_IDS,target_names=CLASS_NAMES,output_dict=True,zero_division=0)})
        results[name] = {'stage1_fitness':selected[name]['accuracy'],'stage1_k':selected[name]['k'],
                         'stage2_k':9 if name=='knn' else None,'stage2':values,'fit_warnings':model.paper_fit_warnings_}
        rows.append({'classifier':name,'stage1_fitness':selected[name]['accuracy'],
            'stage1_k':selected[name]['k'],'stage2_k':9 if name=='knn' else None,
            'stage2_accuracy':values['accuracy'],'stage2_macro_f1':values['macro_f1'],'diagnostic_only':True})
        part = source.iloc[later_test][['path','label','overlap_group']].copy()
        part['classifier'],part['prediction'] = name,predicted
        predictions.append(part)
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    majority = int(pd.Series(labels[later_fit]).mode().iloc[0])
    dump(out/'results.json',{'models':results,'selected_model':winner,'selection_uses_stage2':False,
        'diagnostic_only':True,'majority':score(labels[later_test],np.full(len(later_test),majority)),
        'elapsed_seconds':time.time()-started,'strict_literal':args.strict_literal,
        'full_100_generation_study':args.generations==100 and args.population==100 and args.classifiers==CLASSIFIERS,
        'exact_unpublished_author_execution_reproduced':False})
    plot_report(out,histories,rows)


def plot_report(out,histories,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig,ax = plt.subplots(figsize=(10,5))
    for name,history in histories.items():
        ax.plot([row['generation'] for row in history],[row['best_validation_accuracy'] for row in history],label=name)
    ax.set(xlabel='GA generation (0=initialization)',ylabel='Adaptive stage1 fitness',ylim=(0,1))
    ax.legend(ncol=2);fig.tight_layout();fig.savefig(out/'training_graph.png',dpi=140);plt.close(fig)
    for row in rows:
        name = row['classifier'];directory = out/name
        fig,ax = plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay(np.loadtxt(directory/'confusion.csv',delimiter=',',dtype=int),display_labels=CLASS_NAMES).plot(ax=ax,colorbar=False)
        ax.set_title(name+': stage2 diagnostic');fig.tight_layout();fig.savefig(directory/'confusion.png',dpi=140);plt.close(fig)
    fig,ax = plt.subplots(figsize=(11,5))
    ax.bar([row['classifier'] for row in rows],[row['stage2_accuracy'] for row in rows])
    ax.set(ylabel='Stage2 diagnostic accuracy',ylim=(0,1));ax.tick_params(axis='x',rotation=30)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset')
    parser.add_argument('--manifest',default='artifacts/manifest.csv')
    parser.add_argument('--out',default='artifacts_paper_method/full')
    parser.add_argument('--cache-dir',default='artifacts_paper_method/cache')
    parser.add_argument('--classifiers',nargs='+',choices=CLASSIFIERS,default=CLASSIFIERS.copy())
    parser.add_argument('--generations',type=int,default=100)
    parser.add_argument('--population',type=int,default=100)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--validation-seed',type=int,default=43)
    parser.add_argument('--normalization',choices=['within-window','train'],default='within-window')
    parser.add_argument('--higuchi',choices=['conventional','literal'],default='conventional')
    parser.add_argument('--undefined-policy',choices=['median','error'],default='median')
    parser.add_argument('--power-low',type=float,default=0.)
    parser.add_argument('--power-high',type=float,default=64.)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--strict-literal',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if args.generations<1 or args.population<2 or args.population%2 or args.batch_size<1:
        parser.error('Require positive generations, even population >=2, positive batch size')
    if len(args.classifiers)!=len(set(args.classifiers)):
        parser.error('Duplicate classifiers')
    with threadpool_limits(limits=4):
        run(args)


if __name__=='__main__':
    # Keep checkpoint classes importable when the trainer is executed with -m.
    import sys
    sys.modules['eegis.paper_protocol']=sys.modules[__name__]
    EquationTransform.__module__='eegis.paper_protocol'
    EquationModel.__module__='eegis.paper_protocol'
    main()
