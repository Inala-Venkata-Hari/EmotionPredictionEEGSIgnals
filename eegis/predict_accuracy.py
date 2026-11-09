"""Classify one raw window or one reconstructed No/Sí component."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .epochs import read_window,reconstruct


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('windows',nargs='+',type=Path,help='Original raw EEGIS CSV files')
    p.add_argument('--artifacts',type=Path,default=Path('artifacts_accuracy_refit'))
    p.add_argument('--unit',choices=['window','recording'],default='window')
    args=p.parse_args();required=1 if args.unit=='window' else 12
    if len(args.windows)!=required:p.error(f'{args.unit} inference requires {required} raw CSV file(s)')
    values=np.stack([read_window(path) for path in args.windows])
    raw=values if args.unit=='window' else reconstruct(values)[0][None]
    path=args.artifacts/args.unit/'best_model.joblib';model=joblib.load(path)
    with threadpool_limits(limits=4):label=int(model.predict(raw)[0])
    print(json.dumps({'class_id':label,'label':{6:'No',8:'Sí'}[label],'unit':args.unit,
        'input_seconds':raw.shape[1]/128,'model_file':str(path),'exploratory':True,
        'subject_independent':False},ensure_ascii=False))


if __name__=='__main__':main()
