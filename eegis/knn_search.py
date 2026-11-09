"""Reference kNN fitness and resumable GA with unchanged numerical semantics."""
import copy
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .paper_ga import CLASS_IDS, K_VALUES, genome_columns
from .paper_protocol import EquationTransform


class SparseMissingTransform(EquationTransform):
    """Compute only medians actually needed; retain sklearn column scaling."""
    def transform(self,x):
        if self.normalization!='within-window':
            return super().transform(x)
        x=self._check(x)
        a=x.reshape(-1,self.channels,self.features)
        missing=np.isnan(a)
        if missing.any():
            a=a.copy()
            rows,columns=np.where(missing.any(axis=1))
            selected=a[rows,:,columns]
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='All-NaN slice encountered')
                median=np.nanmedian(selected,axis=1)
            median=np.where(np.isnan(median),0.,median)
            a[rows,:,columns]=np.where(np.isnan(selected),median[:,None],selected)
        columns=a.transpose(1,0,2).reshape(self.channels,-1)
        z=StandardScaler().fit_transform(columns)
        return z.reshape(self.channels,len(a),self.features).transpose(1,0,2).reshape(len(a),-1)


def selected_data(features,genome,fit):
    flat=features.reshape(len(features),-1)[:,genome_columns(genome)]
    transform=SparseMissingTransform(np.sum(genome[:14]),np.sum(genome[14:]),'within-window','median').fit(flat[fit])
    return transform,transform.transform(flat)


def reference_fitness(features,labels,fit,validation,genome):
    _,z=selected_data(features,genome,fit)
    a,b=z[fit],z[validation]
    predictions=np.stack([KNeighborsClassifier(n_neighbors=k).fit(a,labels[fit]).predict(b) for k in K_VALUES])
    target=labels[validation][None]
    accuracy=np.mean(predictions==target,axis=1)
    f1=[]
    for label in CLASS_IDS:
        tp=np.sum((predictions==label)&(target==label),axis=1)
        denominator=np.sum(predictions==label,axis=1)+np.sum(target==label,axis=1)
        f1.append(2*tp/np.maximum(denominator,1))
    macro=np.mean(f1,axis=0)
    records=[{'k':k,'accuracy':float(acc),'macro_f1':float(value)}
             for k,acc,value in zip(K_VALUES,accuracy,macro)]
    return {**records[int(np.argmax(accuracy))],'all_k_scores':records}


def atomic_dump(value,path):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    joblib.dump(value,temporary,compress=1)
    os.replace(temporary,path)


def resumable_search(features,labels,fit,validation,out,generations=100,population_size=100,
                     seed=42,guard=None,evaluator=None,stop_after=None,progress=None):
    """Same population/RNG transitions as paper_ga.genetic_search.

    stop_after is a test hook that simulates interruption after an atomic
    checkpoint; normal training always runs all requested generations.
    """
    if generations<1 or population_size<2 or population_size%2:
        raise ValueError('Positive generations and even population >=2 required')
    if set(fit)&set(validation) or min(len(fit),len(validation))==0:
        raise ValueError('Fitness partitions must be nonempty and disjoint')
    if set(np.unique(labels[fit]))!=set(CLASS_IDS) or len(fit)<max(K_VALUES):
        raise ValueError('Both classes and at least37 fitting windows required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'search_state.joblib'
    rng=np.random.default_rng(seed)
    config={'generations':generations,'population':population_size,'seed':seed,'guard':guard}
    if checkpoint.exists():
        state=joblib.load(checkpoint)
        if state['config']!=config:
            raise ValueError('Checkpoint data/code/configuration differs; use a new output')
        population,cache,history,states=state['population'],state['cache'],state['history'],state['states']
        rng.bit_generator.state=state['rng_state']
        first_generation=state['generation']+1
        prior_elapsed=history[-1]['elapsed_seconds']
    else:
        population=rng.integers(0,2,size=(population_size,228),dtype=np.uint8)
        cache,history,states={},[],[]
        first_generation,prior_elapsed=0,0.
    def repair(genome):
        if not genome[:14].any():
            genome[rng.integers(14)]=1
        if not genome[14:].any():
            genome[14+rng.integers(214)]=1
        return genome
    if first_generation==0:
        population=np.stack([repair(row) for row in population])
    def evaluate(genome):
        key=genome.tobytes()
        if key not in cache:
            cache[key]=(reference_fitness(features,labels,fit,validation,genome) if evaluator is None
                        else evaluator(features,labels,fit,validation,genome))
        return cache[key]
    started=time.monotonic()
    for generation in range(first_generation,generations+1):
        if generation:
            weights=np.arange(population_size,0,-1,dtype=float)
            weights/=weights.sum()
            children=[]
            for _ in range(population_size//2):
                parents=rng.choice(population_size,2,replace=False,p=weights)
                first,second=population[parents]
                crossover=rng.random(228)<.5
                for child in [np.where(crossover,first,second),np.where(crossover,second,first)]:
                    child=child.astype(np.uint8)
                    child^=(rng.random(228)<1/228).astype(np.uint8)
                    children.append(repair(child))
            candidates=np.concatenate([population,np.stack(children)])
        else:
            candidates=population
        scores=[evaluate(row) for row in candidates]
        order=sorted(range(len(candidates)),key=lambda index:-scores[index]['accuracy'])[:population_size]
        population=candidates[order].copy()
        scores=[scores[index] for index in order]
        best=scores[0]
        row={'generation':generation,'best_validation_accuracy':best['accuracy'],
            'best_validation_macro_f1':best['macro_f1'],
            'mean_survivor_accuracy':float(np.mean([value['accuracy'] for value in scores])),
            'best_k':best['k'],'selected_channels':int(population[0,:14].sum()),
            'selected_feature_types':int(population[0,14:].sum()),
            'unique_evaluations':len(cache),'elapsed_seconds':prior_elapsed+time.monotonic()-started}
        history.append(row)
        states.append({'generation':generation,'genomes':population.copy(),
                       'accuracies':np.array([value['accuracy'] for value in scores])})
        atomic_dump({'config':config,'generation':generation,'population':population,'cache':cache,
                     'history':history,'states':states,'rng_state':rng.bit_generator.state},checkpoint)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        if progress:
            progress(f'generation {generation:03d}/{generations}: fitness={best["accuracy"]:.4f}; '
                     f'k={best["k"]}; unique={len(cache)}; elapsed={row["elapsed_seconds"]:.1f}s')
        if stop_after is not None and generation>=stop_after and generation<generations:
            return None,history,states
    result=copy.deepcopy(evaluate(population[0]))
    result.update({'genome':population[0].tolist(),'generations':generations,'population_size':population_size,
        'unique_evaluations':len(cache),'mutation_rate':1/228,
        'evaluations':[{'genome':list(key),**value} for key,value in cache.items()]})
    return result,history,states
