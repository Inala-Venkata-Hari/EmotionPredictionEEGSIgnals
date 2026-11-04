"""Independently reload checkpoints and verify saved test predictions/metrics."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from .data import preprocess,features
from .models import EEGNet
from .train import scores

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--artifacts',default='artifacts');args=ap.parse_args()
    out=Path(args.artifacts);r=json.loads((out/'results.json').read_text())
    data=Path(r['arguments']['data']);m=pd.read_csv(out/'manifest.csv');p=pd.read_csv(out/'test_predictions.csv')
    assert m.groupby('overlap_group').split.nunique().max()==1
    assert len(p)==r['split_sizes']['test']
    x=np.stack([pd.read_csv(data/path).iloc[:,1:].to_numpy(dtype=np.float64) for path in p.path])
    z=preprocess(x);f=features(z)
    for name in ['SVM','RandomForest','EEGNet']:
        if name=='EEGNet':
            saved=torch.load(out/'EEGNet.pt',map_location='cpu',weights_only=True)
            net=EEGNet();net.load_state_dict(saved['state_dict']);net.eval();torch.set_num_threads(4)
            a=((z-saved['mean'].numpy())/saved['std'].numpy()).transpose(0,2,1)[:,None].copy()
            with torch.no_grad():pred=net(torch.from_numpy(a)).argmax(1).numpy()
        else:pred=joblib.load(out/f'{name}.joblib').predict(f)
        np.testing.assert_array_equal(pred,p[name].values)
        for metric,value in scores(p.label,pred).items():
            assert abs(value-r['models'][name]['test'][metric])<1e-12
        cm=np.loadtxt(out/f'{name}_confusion.csv',delimiter=',')
        assert cm.sum()==len(p)
        print(name,'checkpoint predictions and reported metrics verified')
    expected=max(r['models'],key=lambda k:r['models'][k]['validation']['macro_f1'])
    assert r['best_model']==expected
    from .predict import predict
    one=p.iloc[0]
    assert predict(data/one.path,out)['class_id']==int(one[expected])
    (out/'verification.json').write_text(json.dumps({'checkpoints_match_predictions':True,
        'metrics_recomputed':True,'confusion_counts_valid':True,'best_selection_valid':True,
        'inference_matches_checkpoint':True,'test_windows':len(p)},indent=2))
    print('Best-model selection and single-window inference verified')

if __name__=='__main__':main()
