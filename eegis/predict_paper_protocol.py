"""Predict No or Sí with a saved printed-paper feature pipeline."""
import argparse
import json
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from .epochs import read_window


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('window',type=Path,help='One original raw EEGIS CSV')
    parser.add_argument('--artifacts',type=Path,default=Path('artifacts_paper_method/smoke'))
    args=parser.parse_args()
    model=joblib.load(args.artifacts/'best_model.joblib')
    with threadpool_limits(limits=4):
        label=int(model.predict(read_window(args.window)[None])[0])
    print(json.dumps({'class_id':label,'label':{6:'No',8:'Sí'}[label],
        'model_file':str(args.artifacts/'best_model.joblib'),
        'checkpoint_from_diagnostic_experiment':True},ensure_ascii=False))


if __name__=='__main__':
    main()
