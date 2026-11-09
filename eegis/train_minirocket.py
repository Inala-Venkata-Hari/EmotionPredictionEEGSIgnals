"""Train-only nested MiniRocket using the reference numerical implementation."""
import argparse
from dataclasses import dataclass
from importlib.metadata import version

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .aeon_minirocket import MiniRocket
from .accuracy_features import validate
from .research_protocol import fingerprint as source_fingerprint,run


DEPENDENCIES=['train_minirocket.py','aeon_minirocket.py','research_protocol.py','accuracy_features.py',
    'train_accuracy.py','data.py','epochs.py','improve.py','train_paper_binary.py','paper_ga.py','paper_features.py']


def candidates():
    model=Pipeline([('rocket',MiniRocket()),('scale',StandardScaler(with_mean=False)),
        ('model',RidgeClassifier(class_weight='balanced',solver='lsqr',tol=1e-4))])
    return [('minirocket','raw',model,{'rocket__mode':['raw','filtered','derivative'],'model__alpha':[1.,100.,1000.]}),
            ('majority','raw',DummyClassifier(strategy='most_frequent'),{})]


@dataclass
class MiniRocketModel:
    estimator:object
    bank:str
    samples:int

    def features(self,x):
        x=validate(x)
        if x.shape[1]!=self.samples:raise ValueError('Incorrect input duration')
        return x

    def predict(self,x):return self.estimator.predict(self.features(x))


def banks(x):return {'raw':validate(x)}


def fingerprint():return source_fingerprint(DEPENDENCIES)


def specification():
    return {'name':'minirocket','dependencies':DEPENDENCIES,'candidates':candidates,'features':banks,'factory':MiniRocketModel,
        'description':'Aeon BSD-licensed unchanged MiniRocket numerical helpers; EEG/sklearn adapter. All learned thresholds fit inside each current training fold. 10,000 requested features, 84 fixed kernel patterns, three input representations and fixed Ridge alpha grid. Numba='+version('numba')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset')
    p.add_argument('--manifest',default='artifacts/manifest.csv');p.add_argument('--out',default='artifacts_minirocket')
    args=p.parse_args()
    with threadpool_limits(limits=4):run(specification(),args)


if __name__=='__main__':
    from eegis.train_minirocket import main
    main()
