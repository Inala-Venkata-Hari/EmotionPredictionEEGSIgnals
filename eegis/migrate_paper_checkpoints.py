"""Repair our local pre-fix __main__ checkpoint references without retraining.

Require an exact source snapshot and identical executable code outside the CLI
guard. Archive originals and verify every saved prediction before/after repair.
"""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .paper_protocol import EquationModel, EquationTransform, code_fingerprint
from .train_paper_binary import dump


def without_cli_guard(text):
    module=ast.parse(text)
    module.body=[node for node in module.body if not (
        isinstance(node,ast.If) and isinstance(node.test,ast.Compare)
        and isinstance(node.test.left,ast.Name) and node.test.left.id=='__name__')]
    return ast.dump(module,include_attributes=False)


def check_source(root,plan,current):
    for name,expected in plan['code_sha256'].items():
        snapshot=root/'source_snapshot'/name
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==expected,name
        if name!='paper_protocol.py':
            assert current[name]==expected,name
    before=(root/'source_snapshot'/'paper_protocol.py').read_text()
    after=Path(__file__).with_name('paper_protocol.py').read_text()
    assert without_cli_guard(before)==without_cli_guard(after),'Numerical code changed; this repair is not applicable'


def migrate(root):
    plan=json.loads((root/'plan.json').read_text())
    current=code_fingerprint()
    check_source(root,plan,current)
    audit=json.loads((root/'feature_audit.json').read_text())
    with np.load(audit['cache_path'],allow_pickle=False) as cache:
        bank=cache['features']
    manifest=pd.read_csv(root/'manifest.csv')
    test=np.flatnonzero(manifest.stage2=='evaluation')
    predictions=pd.read_csv(root/'predictions.csv')
    winner=json.loads((root/'selection.json').read_text())['winner']
    # This resolves only our trusted local __main__ references. Saved objects
    # become canonical paper_protocol classes when they are serialized again.
    setattr(sys.modules['__main__'],'EquationModel',EquationModel)
    setattr(sys.modules['__main__'],'EquationTransform',EquationTransform)
    archive=root/'original_checkpoints';archive.mkdir(exist_ok=True)
    records=[]
    cases=[(root/name/'model.joblib',name) for name in plan['classifiers']]
    cases.append((root/'best_model.joblib',winner))
    for path,name in cases:
        original=path.read_bytes()
        backup=archive/(name+'_model.joblib' if path.name!='best_model.joblib' else 'best_model.joblib')
        if backup.exists():
            raise ValueError('Original archive exists; do not repeat migration')
        backup.write_bytes(original)
        model=joblib.load(path)
        expected=predictions[predictions.classifier==name].prediction.to_numpy()
        np.testing.assert_array_equal(model.predict_features(bank[test]),expected)
        assert type(model).__module__=='eegis.paper_protocol'
        assert type(model.transform).__module__=='eegis.paper_protocol'
        joblib.dump(model,path)
        reloaded=joblib.load(path)
        np.testing.assert_array_equal(reloaded.predict_features(bank[test]),expected)
        records.append({'file':str(path.relative_to(root)),
            'original_sha256':hashlib.sha256(original).hexdigest(),
            'repaired_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'all_predictions_unchanged':True})
    dump(root/'checkpoint_migration.json',{'reason':'CLI __main__ pickle module references',
        'original_training_code_sha256':plan['code_sha256'],'current_code_sha256':current,
        'all_code_outside_cli_guard_ast_identical':True,'new_training_executed':False,
        'models':records,'original_plan_preserved':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=Path('artifacts_paper_method/smoke'))
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        migrate(args.artifacts)


if __name__=='__main__':
    main()
