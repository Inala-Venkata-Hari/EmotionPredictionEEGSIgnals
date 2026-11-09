import tempfile
import unittest

import numpy as np

from eegis.knn_search import SparseMissingTransform, reference_fitness, resumable_search
from eegis.paper_ga import genetic_search
from eegis.paper_protocol import EquationTransform, fitness


def objective(features,labels,fit,validation,genome):
    value=float(np.dot(genome,np.arange(1,229))%997)/997
    return {'k':5,'accuracy':value,'macro_f1':value,'all_k_scores':[]}


def legacy_objective(features,labels,fit,validation,genome,classifier,seed,device):
    return objective(features,labels,fit,validation,genome)


class KNNFullSearchTests(unittest.TestCase):
    def test_sparse_imputation_is_bitwise_equal_to_existing_scaler(self):
        rng=np.random.default_rng(31)
        for channels in [1,2,7,14]:
            for missing in [False,True]:
                x=rng.normal(size=(27,channels,17));x[:,:,0]=7
                if missing:
                    x[0,:,1]=np.nan;x[1,0,2]=np.nan;x[3,0,4]=np.nan
                flat=x.reshape(len(x),-1)
                old=EquationTransform(channels,17).fit(flat[:20])
                new=SparseMissingTransform(channels,17).fit(flat[:20])
                np.testing.assert_array_equal(old.transform(flat),new.transform(flat))

    def test_reference_fitness_matches_all_seven_sklearn_scores(self):
        rng=np.random.default_rng(4)
        bank=rng.normal(size=(80,14,214));bank[2,0,33]=np.nan
        y=np.array([6,8]*40)
        fit,val=np.arange(60),np.arange(60,80)
        for channels in [1,2,8]:
            genome=np.ones(228,dtype=np.uint8);genome[:14]=0;genome[:channels]=1
            a=fitness(bank,y,fit,val,genome,'knn',42,'cpu')
            b=reference_fitness(bank,y,fit,val,genome)
            for left,right in zip(a['all_k_scores'],b['all_k_scores']):
                self.assertEqual(left['k'],right['k'])
                self.assertEqual(left['accuracy'],right['accuracy'])
                self.assertAlmostEqual(left['macro_f1'],right['macro_f1'],places=14)

    def test_checkpoint_resume_matches_original_ga_population_and_rng(self):
        features=np.zeros((80,14,214));labels=np.array([6,8]*40)
        fit,val=np.arange(60),np.arange(60,80)
        original,h,states=genetic_search(features,labels,fit,val,generations=6,population_size=6,
            seed=17,device='cpu',evaluator=legacy_objective)
        with tempfile.TemporaryDirectory() as out:
            partial,_,_=resumable_search(features,labels,fit,val,out,generations=6,population_size=6,
                seed=17,guard='fixed',evaluator=objective,stop_after=2)
            self.assertIsNone(partial)
            actual,history,populations=resumable_search(features,labels,fit,val,out,generations=6,
                population_size=6,seed=17,guard='fixed',evaluator=objective)
            self.assertEqual(actual['genome'],original['genome'])
            self.assertEqual(actual['accuracy'],original['accuracy'])
            self.assertEqual(actual['unique_evaluations'],original['unique_evaluations'])
            for left,right in zip(states,populations):
                np.testing.assert_array_equal(left['genomes'],right['genomes'])
                np.testing.assert_array_equal(left['accuracies'],right['accuracies'])
            for left,right in zip(h,history):
                for key in left:
                    if key!='elapsed_seconds':
                        self.assertEqual(left[key],right[key])
            with self.assertRaises(ValueError):
                resumable_search(features,labels,fit,val,out,generations=6,population_size=6,
                    seed=17,guard='changed',evaluator=objective)


if __name__=='__main__':
    unittest.main()
