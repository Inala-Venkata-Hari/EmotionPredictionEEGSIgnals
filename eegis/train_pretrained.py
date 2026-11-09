"""Train-only nested classification with frozen external EEG representations."""
import argparse
from importlib.metadata import version

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .pretrained_features import MODES,PretrainedModel,checkpoint_sha,pretrained_banks
from .research_protocol import fingerprint as source_fingerprint,run


DEPENDENCIES=['train_pretrained.py','pretrained_features.py','labram_backbone.py','labram_channels.py','labram_support.py',
    'research_protocol.py','signal_models.py','accuracy_features.py','train_accuracy.py','data.py','epochs.py',
    'improve.py','train_paper_binary.py','paper_ga.py','paper_features.py']


def candidates():
    specs=[]
    for mode in MODES:
        for pooling in ['cls','mean']:
            bank=mode+'__'+pooling
            specs.append((bank+'_ridge',bank,Pipeline([('scale',StandardScaler()),
                ('model',RidgeClassifier(class_weight='balanced',solver='lsqr',tol=1e-4))]),{'model__alpha':[1.,100.,1000.]}))
            specs.append((bank+'_svm',bank,Pipeline([('scale',StandardScaler()),
                ('model',SVC(class_weight='balanced'))]),{'model__C':[.1,1.,10.]}))
    specs.append(('majority','physical__mean',DummyClassifier(strategy='most_frequent'),{}))
    return specs


def fingerprint():return source_fingerprint(DEPENDENCIES)


def specification():
    return {'name':'pretrained','dependencies':DEPENDENCIES,'candidates':candidates,'features':pretrained_banks,'factory':PretrainedModel,
        'description':'Frozen official LaBraM base encoder; all encoder weights loaded, only pretraining mask/class heads unused. No EEGIS encoder fitting. Four label-free input normalizations and CLS/mean pooling. Native 128Hz data lowpass60/notch50 then resampled200Hz; 1s/7.875s inputs, latter zero-padded .125s internally. Native amplitude calibration unknown; physical mode assumes microvolts/100. Encoder checkpoint SHA256='+checkpoint_sha()+'; Torch='+version('torch')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset')
    p.add_argument('--manifest',default='artifacts/manifest.csv');p.add_argument('--out',default='artifacts_pretrained')
    args=p.parse_args()
    with threadpool_limits(limits=4):run(specification(),args)


if __name__=='__main__':main()
