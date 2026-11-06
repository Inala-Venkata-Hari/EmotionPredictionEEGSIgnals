import unittest
import importlib.util

import numpy as np
import pandas as pd

from eegis.author_features import (author_features, author_series_features, published_genome,
                                   reference_library)
from eegis.author_reproduction import fast_fitness, fixed_cases, fit_model
from eegis.paper_features import COLUMNS
from eegis.paper_ga import fitness_predictions, genome_columns
from eegis.author_window_scaling import WithinWindowScaler,window_fitness
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score


class AuthorReproductionTests(unittest.TestCase):
    def test_within_example_scaling_matches_individual_standardscalers(self):
        x=np.random.default_rng(6).normal(size=(5,8,20));x[:,:,0]=7
        scaler=WithinWindowScaler(8,20).fit(x[:2].reshape(2,-1))
        expected=np.stack([StandardScaler().fit_transform(a).reshape(-1) for a in x])
        np.testing.assert_array_equal(scaler.transform(x.reshape(5,-1)),expected)
        x[1:]+=1000
        np.testing.assert_allclose(scaler.transform(x[:1].reshape(1,-1)),expected[:1],rtol=1e-12,atol=1e-12)

    def test_two_electrode_ties_match_sklearn_per_k(self):
        x=np.random.default_rng(31).normal(size=(80,14,214));y=np.array([6,8]*40)
        genome=published_genome();genome[:14]=0;genome[[4,9]]=1
        train,val=np.arange(60),np.arange(60,80)
        actual=window_fitness(x,y,train,val,genome,'knn',42,'cpu')
        a=x.reshape(len(x),-1)[:,genome_columns(genome)].reshape(len(x),2,103)
        z=np.stack([StandardScaler().fit_transform(row).ravel() for row in a])
        for row in actual['all_k_scores']:
            predicted=KNeighborsClassifier(n_neighbors=row['k']).fit(z[train],y[train]).predict(z[val])
            self.assertEqual(row['accuracy'],accuracy_score(y[val],predicted))

    def test_within_example_fitness_matches_independent_sklearn_pipeline(self):
        x=np.random.default_rng(3).normal(size=(80,14,214));y=np.array([6,8]*40)
        genome=published_genome();train=np.arange(60);val=np.arange(60,80)
        actual=window_fitness(x,y,train,val,genome,'knn',42,'cpu')
        selected=x.reshape(len(x),-1)[:,genome_columns(genome)].reshape(len(x),8,103)
        z=np.stack([StandardScaler().fit_transform(a).ravel() for a in selected])
        for row in actual['all_k_scores']:
            model=KNeighborsClassifier(n_neighbors=row['k'],algorithm='brute').fit(z[train],y[train])
            p=model.predict(z[val])
            self.assertEqual(row['accuracy'],accuracy_score(y[val],p))
            self.assertAlmostEqual(row['macro_f1'],f1_score(y[val],p,labels=[6,8],average='macro'),places=14)

    def test_published_mask_matches_paper_channel_and_feature_counts(self):
        genome = published_genome()
        self.assertEqual(len(genome), 228)
        np.testing.assert_array_equal(np.flatnonzero(genome[:14]), [0, 4, 6, 7, 8, 9, 11, 12])
        self.assertEqual(int(genome[14:].sum()), 103)
        self.assertEqual(len(genome_columns(genome)), 824)
        counts = [sum(int(b) for name, b in zip(COLUMNS, genome[14:]) if name.startswith(layer+'/'))
                  for layer in ['raw','delta','theta','alpha','beta','gamma']]
        self.assertEqual(counts, [15, 23, 16, 14, 20, 15])

    @unittest.skipUnless(importlib.util.find_spec('signalytica'), 'Author reference library is installed in the isolated .venv-paper environment')
    def test_vectorized_features_match_author_library_including_nonstandard_formulas(self):
        rng = np.random.default_rng(3)
        t = np.arange(128)/128
        examples = np.stack([rng.normal(size=128), 4000+10*np.sin(2*np.pi*3*t)+rng.normal(size=128),
                             np.repeat(np.arange(32), 4).astype(float)+.001*rng.normal(size=128)])
        result = author_series_features(examples, band=(8,13))
        source = reference_library()
        functions = ['mean','std','coeff_var','median','mode','max','min','q1','q3','IQR','kurt','skewness',
                     'activity_hjorth_param','mobility_hjorth_param','complexity_hjorth_param',
                     'total_power_spectral_density','centroid_power_spectral_density','determinism',
                     'trapping_time','diagonal_line_entropy','average_diagonal_line_length',
                     'compute_recurrence_rate','sef_25','sef_50','sef_75','hurst_exponent',
                     'relative_alpha_power','alpha_amplitude']
        indices = [*range(12),13,14,15,20,21,22,23,24,25,26,27,28,29,30,34,35]
        for row, x in enumerate(examples):
            for function, index in zip(functions, indices):
                with self.subTest(function=function,row=row):
                    np.testing.assert_allclose(result[row,index], getattr(source,function)(x), rtol=1e-9, atol=1e-9)

    def test_fast_metrics_do_not_change_ga_objective_or_neighbor_selection(self):
        rng = np.random.default_rng(17)
        x, y = rng.normal(size=(90,14,214)), np.array([6,8]*45)
        g = published_genome(); fit, val = np.arange(60),np.arange(60,80)
        fast = fast_fitness(x,y,fit,val,g,'knn',42,'cpu')
        original = fitness_predictions(x,y,fit,val,g,'knn',42,'cpu')
        self.assertEqual(fast['k'],original['k'])
        self.assertEqual(fast['accuracy'],original['accuracy'])
        for a,b in zip(fast['all_k_scores'],original['all_k_scores']):
            self.assertEqual(a['k'],b['k'])
            self.assertEqual(a['accuracy'],b['accuracy'])
            self.assertAlmostEqual(a['macro_f1'],b['macro_f1'],places=14)
        # Outer rows 80..89 are absent from every fitness operation.
        x[80:] *= 10000; y[80:] = y[80:][::-1]
        changed = fast_fitness(x,y,fit,val,g,'knn',42,'cpu')
        self.assertEqual(fast,changed)

    def test_source_feature_bank_is_per_example_and_rejects_constant_inputs(self):
        x=np.random.default_rng(42).normal(size=(2,6,128,14))
        bank=author_features(x)
        self.assertEqual(bank.shape,(2,14,214))
        np.testing.assert_allclose(bank[:1],author_features(x[:1]),rtol=1e-10,atol=1e-10)
        with self.assertRaises(ValueError):
            author_series_features(np.zeros((1,128)))

    def test_fitting_never_uses_outer_test_scaler_statistics(self):
        rng=np.random.default_rng(9)
        bank=rng.normal(size=(80,14,214)); y=np.array([6,8]*40)
        g=published_genome(); train=np.arange(60)
        a=fit_model(bank,y,train,g,7,device='cpu')
        bank[60:]+=1000
        b=fit_model(bank,y,train,g,7,device='cpu')
        np.testing.assert_array_equal(a.scaler.mean_,b.scaler.mean_)
        np.testing.assert_array_equal(a.estimator.x_,b.estimator.x_)


if __name__=='__main__':
    unittest.main()
