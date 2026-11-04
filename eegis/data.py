"""Strict raw loader, overlap reconstruction, and train-only signal transforms."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis

CHANNELS = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
LABELS = ['rest','Ayuda','Baño','Dolor','Gracias','Hambre','No','Sed','Sí']
BANDS = [(0.5,4),(4,8),(8,13),(13,30),(30,40)]

def digest(a):
    a = np.asarray(a, dtype='<f8').copy()
    a[a == 0] = 0  # canonicalize signed zero
    return hashlib.sha256(a.tobytes()).hexdigest()

def overlap_groups(x):
    """Join all exact 48-frame prefix/suffix matches globally, independent of filenames."""
    parent = list(range(len(x)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(i,j):
        parent[root(j)] = root(i)
    seen = {}
    for i, a in enumerate(x):
        for block in (a[:48], a[-48:], a):
            key = (len(block), digest(block))
            if key in seen:
                join(i, seen[key])
            else:
                seen[key] = i
    return np.array([root(i) for i in range(len(x))])

def load_data(data, out):
    data, out = Path(data), Path(out)
    records, arrays, inventory = [], [], {}
    for band in ['raw','delta','theta','alpha','beta','gamma']:
        inventory[band] = {}
        for c in range(9):
            files = sorted((data/band/f'class_{c}').glob('*.csv'), key=lambda p:int(p.stem))
            inventory[band][str(c)] = len(files)
            if not files:
                raise ValueError(f'Missing data: {band}/class_{c}')
            for f in files:
                # Inspect every CSV, including unused band variants.
                a = pd.read_csv(f)
                if a.shape != (128,15) or list(a.columns[1:]) != CHANNELS:
                    raise ValueError(f'Unexpected CSV layout: {f}: {a.shape}')
                if not np.array_equal(a.iloc[:,0].values, np.arange(128)):
                    raise ValueError(f'Invalid exported row index: {f}')
                v = a.iloc[:,1:].to_numpy(dtype=np.float64)
                if not np.isfinite(v).all():
                    raise ValueError(f'Nonfinite EEG: {f}')
                if band == 'raw':
                    records.append({'path':str(f.relative_to(data)), 'label':c, 'file_id':int(f.stem)})
                    arrays.append(v)
    x = np.stack(arrays)
    g = overlap_groups(x)
    manifest = pd.DataFrame(records)
    manifest['overlap_group'] = g
    for _, rows in manifest.groupby('overlap_group'):
        if rows.label.nunique() != 1:
            raise ValueError('Exact overlapping data cross class labels')
    sizes = manifest.groupby('overlap_group').size()
    report = {'source':'https://data.mendeley.com/datasets/73g4fw884c/1',
              'inventory':inventory, 'channels':CHANNELS, 'labels':dict(enumerate(LABELS)),
              'sample_rate_hz':128, 'raw_shape':list(x.shape),
              'participant_ids_present':False, 'subject_independent':False,
              'overlap_samples':48, 'overlap_groups':len(sizes),
              'overlap_group_sizes':{str(k):int(v) for k,v in sizes.value_counts().items()},
              'raw_min':float(x.min()),'raw_max':float(x.max())}
    (out/'inspection.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return x, manifest

def make_split(manifest, subject_map=None, seed=42):
    from sklearn.model_selection import GroupShuffleSplit
    m = manifest.copy()
    if subject_map:
        meta = pd.read_csv(subject_map, dtype=str)
        if not {'path','subject_id'} <= set(meta) or meta.path.duplicated().any():
            raise ValueError('Subject map requires unique path and subject_id columns')
        m = m.merge(meta[['path','subject_id']],on='path',how='left',validate='one_to_one')
        if m.subject_id.isna().any() or (m.subject_id == '').any():
            raise ValueError('Subject map must cover every raw window')
        if m.groupby('overlap_group').subject_id.nunique().max() != 1:
            raise ValueError('Overlapping windows assigned to different subjects')
        ids = m.subject_id.values
        a,b = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(m,groups=ids))
        aa,bb = next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1).split(m.iloc[a],groups=ids[a]))
        splits = {'train':a[aa],'validation':a[bb],'test':b}
    else:
        rng = np.random.default_rng(seed)
        splits = {s:[] for s in ['train','validation','test']}
        for c in range(9):
            groups = m.loc[m.label==c,'overlap_group'].unique()
            rng.shuffle(groups)
            n = len(groups); v = max(1, round(n*.2))
            for name, selected in zip(splits,[groups[2*v:],groups[:v],groups[v:2*v]]):
                splits[name].extend(m.index[(m.label==c)&m.overlap_group.isin(selected)])
    m['split']=''
    for name, indices in splits.items():
        m.loc[indices,'split']=name
        if set(m.loc[indices,'label']) != set(range(9)):
            raise ValueError(f'{name} lacks classes; use a different split seed or richer subject metadata')
    if (m.split=='').any() or m.groupby('overlap_group').split.nunique().max()!=1:
        raise AssertionError('Split leakage')
    if subject_map and m.groupby('subject_id').split.nunique().max()!=1:
        raise AssertionError('Subject leakage')
    return m

def audit_shared_samples(x, manifest):
    """Stricter secondary check: no identical multichannel time sample crosses splits."""
    seen = {}; shared = 0
    for a, split in zip(x,manifest.split):
        for row in a:
            k=digest(row)
            if k in seen and seen[k]!=split:
                shared += 1
            seen[k]=split
    if shared:
        raise ValueError(f'{shared} exact shared multichannel samples cross partitions')
    return {'cross_split_shared_samples':0, 'cross_split_overlap_groups':0}

def preprocess(x):
    # Within-window demeaning and filtering use no other example or split statistics.
    z = x - x.mean(axis=1,keepdims=True)
    sos = signal.butter(4,[.5,40],btype='bandpass',fs=128,output='sos')
    return signal.sosfiltfilt(sos,z,axis=1).astype(np.float32)

def features(x):
    f, p = signal.welch(x,fs=128,nperseg=128,axis=1)
    bp=np.stack([np.trapezoid(p[:,(f>=lo)&(f<=hi)],f[(f>=lo)&(f<=hi)],axis=1) for lo,hi in BANDS],axis=1)
    diff=np.diff(x,axis=1); dd=np.diff(diff,axis=1)
    var=x.var(axis=1)+1e-12
    mob=np.sqrt(diff.var(axis=1)/var)
    comp=np.sqrt(dd.var(axis=1)/(diff.var(axis=1)+1e-12))/(mob+1e-12)
    stats=[np.log(var),np.log(np.mean(diff**2,axis=1)+1e-12),mob,comp,
           skew(x,axis=1),kurtosis(x,axis=1),np.ptp(x,axis=1)]
    return np.nan_to_num(np.concatenate([np.log(bp+1e-12).reshape(len(x),-1),
        (bp/(bp.sum(axis=1,keepdims=True)+1e-12)).reshape(len(x),-1),*stats],axis=1)).astype(np.float32)
