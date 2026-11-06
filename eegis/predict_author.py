"""Predict No/Sí from one aligned six-layer EEGIS example in the pinned environment."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .epochs import read_window


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset')
    p.add_argument('--window',default='class_8/0.csv',help='Path below each signal layer, such as class_8/0.csv')
    p.add_argument('--artifacts',default='artifacts_author_reproduction')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=p.parse_args()
    layers=np.stack([read_window(Path(args.data)/band/args.window)
                     for band in ['raw','delta','theta','alpha','beta','gamma']])[None]
    model=joblib.load(Path(args.artifacts)/'best_model.joblib')
    from sklearn.neighbors import KNeighborsClassifier
    if isinstance(model.estimator,KNeighborsClassifier) and args.device=='cuda':
        raise ValueError('The exact sklearn tie-handling checkpoint uses CPU; CUDA is supported by the Torch-backed variant only')
    model.estimator.device=args.device
    with threadpool_limits(limits=4):
        label=int(model.predict_layers(layers)[0])
    print(json.dumps({'class_id':label,'label':{6:'No',8:'Sí'}[label],
                      'model':'AuthorSourceGA_kNN','input_layers':6},ensure_ascii=False))


if __name__=='__main__':
    main()
