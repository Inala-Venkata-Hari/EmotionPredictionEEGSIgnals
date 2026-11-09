"""Exploratory, training-only kernel class-separation diagnostic."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def main():
    root=Path('artifacts_signal_audit');root.mkdir(exist_ok=True)
    m=pd.read_csv('artifacts_accuracy/recording_manifest.csv')
    training=np.flatnonzero(m.split=='train');labels=m.label.to_numpy()[training]
    first=joblib.load('artifacts_accuracy/recording/features.joblib')['banks']
    domain=joblib.load('artifacts_domain_models/recording/features.joblib')['banks']
    phase=joblib.load('artifacts_signal_models/recording/features.joblib')['banks']
    banks={'spectral':first['spectral'],'spectral_shape':first['shape'],
        'spatial_topography':domain['topography'],'phase_connectivity':phase['connectivity'],
        'raw_offset_scale_artifact_control':domain['domain'][:,:28]}
    rng=np.random.default_rng(42);weights=np.where(labels==6,1/np.sum(labels==6),-1/np.sum(labels==8))
    permutations=np.stack([rng.permutation(weights) for _ in range(999)])
    rows=[];nulls={}
    for name,x in banks.items():
        z=StandardScaler().fit_transform(x[training])
        distance=pairwise_distances(z,metric='sqeuclidean')
        width=np.median(distance[np.triu_indices(len(z),1)]).clip(1e-12)
        kernel=np.exp(-distance/(2*width))
        observed=float(weights@kernel@weights)
        null=np.einsum('ni,ij,nj->n',permutations,kernel,permutations)
        nulls[name]=null
        rows.append({'representation':name,'mmd_squared_biased':observed,
                     'permutation_p':float((1+np.sum(null>=observed))/(len(null)+1))})
    order=np.argsort([r['permutation_p'] for r in rows]);previous=0.
    for rank,index in enumerate(order):
        value=min(1.,rows[index]['permutation_p']*(len(rows)-rank));previous=max(previous,value)
        rows[index]['holm_adjusted_p']=previous
    output={'purpose':'exploratory group-level class-separation diagnostic, not a classifier accuracy',
        'only_original_training_groups_used':True,'overlap_components':len(training),'class_counts':{str(c):int(np.sum(labels==c)) for c in [6,8]},
        'permutations':999,'seed':42,'method':'biased RBF-kernel empirical mean discrepancy; median-distance bandwidth; Holm correction across five representations',
        'results':rows,'interpretation':'A nonsignificant result does not prove absence of information or incorrect labels. The limited sample and adaptive research prevent a definitive signal/noise or upper-accuracy conclusion.',
        'components_not_verified_trials_or_participants':True,'subject_independent':False}
    (root/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    pd.DataFrame(rows).to_csv(root/'class_separation.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(10,9))
    for ax,row in zip(axes.flat,rows):
        ax.hist(nulls[row['representation']],bins=30,color='steelblue')
        ax.axvline(row['mmd_squared_biased'],color='red',label='Observed')
        ax.set_title(row['representation']);ax.legend()
    axes.flat[-1].axis('off');fig.suptitle('Original training components only: permuted class-separation statistics')
    fig.tight_layout();fig.savefig(root/'class_separation.png',dpi=140);plt.close(fig)
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
