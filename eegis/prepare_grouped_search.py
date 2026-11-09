"""Prepare the frozen grouped search in another CPU process, without evaluation.

The primary driver adopts a completed checkpoint if this worker finishes first.
Atomic directory promotion prevents writes to a primary-owned output directory.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .knn_search import resumable_search
from .train_knn100 import fingerprint, load_bank
from .train_paper_binary import dump


class PrimaryOwnsOutput(Exception):
    pass


def run(root,data,manifest):
    plan=json.loads((root/'study_plan.json').read_text())
    if plan['source_code_sha256']!=fingerprint():
        raise ValueError('Training source changed')
    destination=root/'grouped'
    if destination.exists():
        print('Primary already owns grouped output; no parallel work started',flush=True)
        return
    x,source,bank,config,audit=load_bank(plan['feature_run'],data,manifest)
    if audit['array_sha256']!=plan['feature_array_sha256']:
        raise ValueError('Feature bank differs')
    study_sha=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    item=plan['partitions']['grouped']
    rows=np.asarray(item['search_rows'])
    fit=np.flatnonzero(np.isin(rows,item['fit']))
    validation=np.flatnonzero(np.isin(rows,item['fitness']))
    out=root/'_grouped_search_preparation'
    dump(root/'parallel_execution.json',{'numerical_settings_unchanged':True,
        'study_sha256':study_sha,'worker_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'final_predictions_generated_by_worker':False,'threads_per_search':4,
        'completed_checkpoint_promotion':'atomic; never overwrite a primary-owned directory'})
    def progress(line):
        print('grouped worker: '+line,flush=True)
        if destination.exists():
            raise PrimaryOwnsOutput()
    try:
        best,history,states=resumable_search(bank[rows],source.label.to_numpy()[rows],fit,validation,out,
            generations=plan['generations'],population_size=plan['population'],seed=plan['seed'],
            guard={'study_sha256':study_sha,'mode':'grouped'},progress=progress)
    except PrimaryOwnsOutput:
        print('Primary began grouped search; duplicate worker stopped after a checkpoint',flush=True)
        return
    try:
        out.rename(destination)
    except (FileExistsError,OSError):
        if destination.exists():
            print('Primary owns grouped output; prepared duplicate retained separately',flush=True)
            return
        raise
    dump(root/'grouped_search_prepared.json',{'generation':plan['generations'],
        'study_sha256':study_sha,'checkpoint_promoted':True,'final_evaluation_executed':False})
    print('Grouped search complete; checkpoint ready for primary driver. No final scores inspected.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=Path('artifacts_paper_method/knn100'))
    parser.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset')
    parser.add_argument('--manifest',default='artifacts/manifest.csv')
    args=parser.parse_args()
    with (args.artifacts/'grouped_worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with threadpool_limits(limits=4):
            run(args.artifacts,args.data,args.manifest)


if __name__=='__main__':
    main()
