"""Inference on an exported EEGIS raw CSV, using the validation-selected model."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from .data import CHANNELS,LABELS,preprocess,features
from .models import EEGNet

def predict(csv, artifacts='artifacts'):
    root=Path(artifacts)
    meta=json.loads((root/'best_model.json').read_text())
    frame=pd.read_csv(csv)
    if frame.shape!=(128,15) or list(frame.columns[1:])!=CHANNELS:
        raise ValueError('Expected 128 rows, an index column, and the documented 14 EEG channels in order')
    x=frame.iloc[:,1:].to_numpy(dtype=np.float64)[None]
    if not np.isfinite(x).all():raise ValueError('Nonfinite signal')
    z=preprocess(x)
    if meta['model']=='EEGNet':
        state=torch.load(root/meta['file'],map_location='cpu',weights_only=True)
        net=EEGNet();net.load_state_dict(state['state_dict']);net.eval()
        a=((z-state['mean'].numpy())/state['std'].numpy()).transpose(0,2,1)[:,None].copy()
        with torch.no_grad():label=int(net(torch.from_numpy(a)).argmax(1).item())
    else:
        model=joblib.load(root/meta['file'])
        if meta.get('input_transform') == 'raw_window_pipeline':
            model_input=x
        elif meta.get('input_transform') == 'covariance_input':
            from .improve import covariance_input
            model_input=covariance_input(x)
        else:
            model_input=features(z)
        label=int(model.predict(model_input)[0])
    return {'class_id':label,'label':LABELS[label],'model':meta['model']}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('csv');ap.add_argument('--artifacts',default='artifacts')
    args=ap.parse_args();print(json.dumps(predict(args.csv,args.artifacts),ensure_ascii=False))
