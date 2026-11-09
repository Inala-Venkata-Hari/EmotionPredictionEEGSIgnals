"""Third train-only batch: low-dimensional contrasts and latent alignment."""
import argparse

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .domain_features import DomainModel,LatentAlignment,topographic_banks
from .research_protocol import fingerprint as source_fingerprint,run


DEPENDENCIES=['train_domain_models.py','domain_features.py','research_protocol.py','accuracy_features.py',
    'train_accuracy.py','data.py','epochs.py','improve.py','train_paper_binary.py','paper_ga.py','paper_features.py']


def candidates():
    specs=[]
    for bank in ['lateral','topography','relative_spectrum','spectral']:
        specs.append((bank+'_ridge',bank,Pipeline([('select',SelectKBest(f_classif)),('scale',StandardScaler()),
            ('model',RidgeClassifier(class_weight='balanced',solver='lsqr',tol=1e-4))]),
            {'select__k':[3,10,20],'model__alpha':[1.,100.]}))
        specs.append((bank+'_svm',bank,Pipeline([('select',SelectKBest(f_classif)),('scale',StandardScaler()),
            ('model',SVC(class_weight='balanced'))]),{'select__k':[3,10,20],'model__C':[.1,1.]}))
    specs.append(('lateral_lda','lateral',Pipeline([('scale',StandardScaler()),
        ('model',LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[.5,.5]))]),{}))
    specs.append(('latent_alignment','domain',Pipeline([('align',LatentAlignment()),
        ('model',RidgeClassifier(class_weight='balanced',solver='lsqr',tol=1e-4))]),
        {'align__clusters':[2,5,10],'align__mode':['center','interact'],'model__alpha':[1.,100.]}))
    specs.append(('majority','lateral',DummyClassifier(strategy='most_frequent'),{}))
    return specs


def fingerprint():return source_fingerprint(DEPENDENCIES)


def specification():
    return {'name':'domain','dependencies':DEPENDENCIES,'candidates':candidates,'features':topographic_banks,'factory':DomainModel,
        'description':'Fixed homologous-electrode contrasts, within-example spectral topography, inner-fold low-dimensional feature selection, and training-only latent nuisance alignment. Clusters are NOT verified participant IDs.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset')
    p.add_argument('--manifest',default='artifacts/manifest.csv');p.add_argument('--out',default='artifacts_domain_models')
    args=p.parse_args()
    with threadpool_limits(limits=4):run(specification(),args)


if __name__=='__main__':main()
