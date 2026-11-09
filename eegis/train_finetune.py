"""Fixed five-epoch last-block adaptation, evaluated within nested group CV."""
import argparse
from importlib.metadata import version

from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from .finetune_features import FineTuneModel,LastBlockClassifier,prefix_tokens
from .pretrained_features import checkpoint_sha
from .research_protocol import fingerprint as source_fingerprint,run


DEPENDENCIES=['train_finetune.py','finetune_features.py','pretrained_features.py','labram_backbone.py','labram_channels.py',
    'labram_support.py','research_protocol.py','signal_models.py','accuracy_features.py','train_accuracy.py','data.py',
    'epochs.py','improve.py','train_paper_binary.py','paper_ga.py','paper_features.py']


def candidates():
    return [('last_block','tokens',LastBlockClassifier(),{}),('majority','tokens',DummyClassifier(strategy='most_frequent'),{})]


def banks(x):return {'tokens':prefix_tokens(x)}


def fingerprint():return source_fingerprint(DEPENDENCIES)


def specification():
    return {'name':'finetune','dependencies':DEPENDENCIES,'candidates':candidates,'features':banks,'factory':FineTuneModel,
        'description':'Fixed five-epoch last-block adaptation pilot: first11 pretrainedblocks frozen; lastblock+norm+linearhead fitted ONLYinsidecurrenttrainingfold. AdamW lastblock lr1e-4/head1e-3,wd.01,dropout.25,gradientclip1,seed42. Normalized noCAR inputs; no epoch/lr search or early stopping. CheckpointSHA256='+checkpoint_sha()+';Torch='+version('torch')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset')
    p.add_argument('--manifest',default='artifacts/manifest.csv');p.add_argument('--out',default='artifacts_finetune')
    args=p.parse_args()
    with threadpool_limits(limits=4):run(specification(),args)


if __name__=='__main__':main()
