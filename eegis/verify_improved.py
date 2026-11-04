"""Verify improved checkpoints against their saved window-level predictions."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import preprocess,features
from .improve import covariance_input,score
from .predict import predict

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--artifacts',default='artifacts_improved');args=ap.parse_args()
    out=Path(args.artifacts);report=json.loads((out/'results.json').read_text())
    m=pd.read_csv(out/'manifest.csv');original=pd.read_csv('artifacts/manifest.csv')
    pd.testing.assert_frame_equal(m,original)
    p=pd.read_csv(out/'test_predictions.csv')
    data=Path('EEGIS - Electroencephalogram Imagined Speech dataset')
    x=np.stack([pd.read_csv(data/path).iloc[:,1:].to_numpy(dtype=np.float64) for path in p.path])
    cov=covariance_input(x);f=features(preprocess(x))
    for name,r in report['models'].items():
        model=joblib.load(out/f'{name}.joblib')
        pred=model.predict(cov if name in ['TangentSVM','TangentRidge','TangentLogistic','LogCovSVM'] else f)
        np.testing.assert_array_equal(pred,p[name].values)
        for metric,value in score(p.label,pred).items():assert abs(value-r['test'][metric])<1e-12
        assert np.loadtxt(out/f'{name}_confusion.csv',delimiter=',').sum()==len(p)
        print(name,'checkpoint and test metrics verified',flush=True)
    expected=max(report['models'],key=lambda n:(report['models'][n]['validation']['accuracy'],report['models'][n]['validation']['macro_f1']))
    assert expected==report['best_model']
    assert predict(data/p.iloc[0].path,out)['class_id']==int(p.iloc[0][expected])
    (out/'verification.json').write_text(json.dumps({'same_split_as_original':True,'all_checkpoints_match_predictions':True,'metrics_recomputed':True,'selection_valid':True,'inference_verified':True,'test_windows':len(p)},indent=2))
    print('Fixed split, selection, and inference verified')

if __name__=='__main__':main()
