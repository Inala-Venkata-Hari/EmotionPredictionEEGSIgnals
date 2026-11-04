"""Paired overlap-group bootstrap for the preselected models from both runs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    old=Path('artifacts');new=Path('artifacts_improved')
    a=json.loads((old/'results.json').read_text());b=json.loads((new/'results.json').read_text())
    pa=pd.read_csv(old/'test_predictions.csv');pb=pd.read_csv(new/'test_predictions.csv')
    assert pa.path.tolist()==pb.path.tolist()
    assert pa.label.tolist()==pb.label.tolist()
    oldwin=a['best_model'];newwin=b['best_model']
    oldcorrect=(pa[oldwin].values==pa.label.values).astype(float)
    newcorrect=(pb[newwin].values==pb.label.values).astype(float)
    frame=pd.DataFrame({'group':pa.overlap_group,'difference':newcorrect-oldcorrect})
    pergroup=frame.groupby('group').difference.mean().to_numpy()
    rng=np.random.default_rng(42)
    draws=rng.choice(pergroup,(5000,len(pergroup)),replace=True).mean(axis=1)
    result={'original_selected_model':oldwin,'new_selected_model':newwin,
        'original_test_accuracy':float(oldcorrect.mean()),'new_test_accuracy':float(newcorrect.mean()),
        'accuracy_difference_percentage_points':float((newcorrect.mean()-oldcorrect.mean())*100),
        'paired_group_bootstrap_95_percent_interval_pp':(np.quantile(draws,[.025,.975])*100).tolist(),
        'bootstrap_replicates':5000,'groups':len(pergroup),
        'limitation':'Exploratory follow-up on a previously reported test split; participants unidentified. Does not quantify participant-independent uncertainty.'}
    (new/'paired_comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
