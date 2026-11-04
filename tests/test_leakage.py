import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from eegis.data import overlap_groups,make_split,audit_shared_samples,preprocess,features

class LeakageTests(unittest.TestCase):
    def test_shuffled_overlap_chain(self):
        rng=np.random.default_rng(0)
        signal=rng.normal(size=(288,14))
        a=np.stack([signal[160:288],signal[:128],signal[80:208],rng.normal(size=(128,14))])
        g=overlap_groups(a)
        self.assertEqual(len(set(g[:3])),1)
        self.assertNotEqual(g[0],g[3])
        m=pd.DataFrame({'split':['train','test','train','test']})
        with self.assertRaises(ValueError):audit_shared_samples(a,m)

    def test_group_split(self):
        rows=[{'path':f'raw/class_{c}/{i}.csv','label':c,'overlap_group':c*100+i//12} for c in range(9) for i in range(120)]
        m=make_split(pd.DataFrame(rows))
        self.assertEqual(m.groupby('overlap_group').split.nunique().max(),1)
        for split in m.split.unique():self.assertEqual(m[m.split==split].label.nunique(),9)

    def test_features(self):
        x=np.random.default_rng(0).normal(size=(3,128,14))
        a=preprocess(x);f=features(a)
        self.assertEqual(f.shape,(3,238))
        self.assertTrue(np.isfinite(f).all())
        # A held-out example cannot alter another example's preprocessing.
        np.testing.assert_allclose(preprocess(x[:1]),a[:1])

    def test_subject_map_keeps_people_disjoint(self):
        rows=[{'path':f'raw/class_{c}/{s*12+i}.csv','label':c,
               'overlap_group':s*9+c,'subject_id':str(s)}
              for s in range(10) for c in range(9) for i in range(12)]
        frame=pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'subjects.csv'
            frame[['path','subject_id']].to_csv(path,index=False)
            split=make_split(frame.drop(columns='subject_id'),path)
            self.assertEqual(split.groupby('subject_id').split.nunique().max(),1)
            # Contradictory subject assignments inside an overlap group must fail.
            frame.loc[0,'subject_id']='another_person'
            frame[['path','subject_id']].to_csv(path,index=False)
            with self.assertRaises(ValueError):
                make_split(frame.drop(columns='subject_id'),path)

if __name__=='__main__':unittest.main()
