"""Predeclared alternative interpretation of 'standardize columns before flattening'."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from .paper_ga import CLASS_IDS, K_VALUES, TorchKNN, genome_columns, resolve_device


class WithinWindowScaler:
    """Per-example feature-column z-score across selected electrodes; no population fit."""
    per_example = True

    def __init__(self, channels, features):
        self.channels, self.features = channels, features

    def fit(self, x, y=None):
        if np.asarray(x).shape[1] != self.channels*self.features:
            raise ValueError('Unexpected selected matrix size')
        return self

    def transform(self, x):
        x=np.asarray(x,dtype=np.float64).reshape(-1,self.channels,self.features)
        # Each example occupies its own feature columns. Statistics reduce only
        # electrode rows, exactly as separate sklearn scalers, without Python loops.
        independent_columns=x.transpose(1,0,2).reshape(self.channels,-1)
        z=StandardScaler().fit_transform(independent_columns)
        return z.reshape(self.channels,len(x),self.features).transpose(1,0,2).reshape(len(x),-1)


def window_fitness(features, labels, train, validation, genome, classifier, seed, device):
    if classifier!='knn':
        raise ValueError('kNN only')
    columns=genome_columns(genome)
    selected=features.reshape(len(features),-1)[:,columns]
    scaler=WithinWindowScaler(int(genome[:14].sum()),int(genome[14:].sum())).fit(selected[train])
    if device=='cuda':
        raise ValueError('Exact sklearn tie handling in this normalization variant requires CPU')
    a,b=scaler.transform(selected[train]),scaler.transform(selected[validation])
    # Query each k independently: slicing a k=37 neighbor list is not equivalent
    # to sklearn's per-k selection when many distances are equal.
    predictions=[KNeighborsClassifier(n_neighbors=k).fit(a,labels[train]).predict(b) for k in K_VALUES]
    # Calculate the objective in NumPy; independently checked against sklearn in tests/verifier.
    records=[]
    for k,predicted in zip(K_VALUES,predictions):
        target=labels[validation]
        f1=[]
        for label in CLASS_IDS:
            tp=np.sum((predicted==label)&(target==label))
            denominator=np.sum(predicted==label)+np.sum(target==label)
            f1.append(2*tp/max(denominator,1))
        records.append({'k':k,'accuracy':float(np.mean(predicted==target)),'macro_f1':float(np.mean(f1))})
    best=max(range(len(records)),key=lambda i:records[i]['accuracy'])
    return {**records[best],'all_k_scores':records}


def main():
    from .author_reproduction import prepare,run_ga
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset')
    p.add_argument('--manifest',default='artifacts/manifest.csv')
    p.add_argument('--out',default='artifacts_author_reproduction')
    p.add_argument('--generations',type=int,default=100)
    p.add_argument('--population',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=p.parse_args();args.normalization='within-window'
    if args.generations<1 or args.population<2 or args.population%2:
        p.error('Positive generations and even population >=2 required')
    resolve_device(args.device);torch.set_num_threads(4)
    out=Path(args.out)
    plan=json.loads((out/'experiment_plan.json').read_text())
    if 'additional_predeclared_variant' not in plan:
        raise ValueError('Declare this normalization interpretation before running the additional experiment')
    with threadpool_limits(limits=4):
        x,source,layers,features=prepare(args.data,args.manifest,out)
        run_ga(args,x,source,features,'grouped',out/'within_window_grouped')


if __name__=='__main__':
    main()
