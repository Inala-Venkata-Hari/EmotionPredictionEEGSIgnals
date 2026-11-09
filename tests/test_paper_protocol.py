import unittest
import ast
import io
import sys
import types
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from scipy import signal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from eegis.paper_equations import (EquationConfig, TRAPZ, band_amplitude,
    equation_features, filter_layers, literal_higuchi, series_features)
from eegis.paper_ga import CLASSIFIERS, K_VALUES
from eegis.paper_protocol import EquationModel, EquationTransform, estimator, fitness, seeded_fit, stage_splits


class PaperProtocolTests(unittest.TestCase):
    def test_cli_checkpoints_load_outside_the_training_main_module(self):
        import eegis.paper_protocol as protocol
        source=ast.parse(Path(protocol.__file__).read_text())
        guard=source.body[-1]
        self.assertIsInstance(guard,ast.If)
        fake_main=types.ModuleType('__main__')
        fake_main.EquationModel=EquationModel
        fake_main.EquationTransform=EquationTransform
        namespace={'__name__':'__main__','EquationModel':EquationModel,
                   'EquationTransform':EquationTransform,'main':lambda:None}
        previous=[EquationModel.__module__,EquationTransform.__module__]
        stream=io.BytesIO()
        try:
            EquationModel.__module__=EquationTransform.__module__='__main__'
            with patch.dict(sys.modules,{'__main__':fake_main}):
                exec(compile(ast.Module(body=[guard],type_ignores=[]),protocol.__file__,'exec'),namespace)
                genome=np.zeros(228,dtype=np.uint8);genome[[0,14]]=1
                transform=EquationTransform(1,1).fit(np.arange(5.)[:,None])
                model=EquationModel(genome,transform,KNeighborsClassifier(n_neighbors=1).fit(
                    np.zeros((5,1)),np.array([6,8,6,8,6])),EquationConfig())
                joblib.dump(model,stream)
        finally:
            EquationModel.__module__,EquationTransform.__module__=previous
        stream.seek(0)
        loaded=joblib.load(stream)
        self.assertIs(type(loaded),EquationModel)
        np.testing.assert_array_equal(loaded.predict_features(np.zeros((2,14,214))),[6,6])

    def test_printed_statistics_and_shape_definitions(self):
        x = np.arange(1,129,dtype=float)[None]
        values = series_features(x)
        self.assertEqual(values.shape,(1,34))
        self.assertAlmostEqual(values[0,2],x.std()/x.mean())
        self.assertAlmostEqual(values[0,7],np.median(x[0,:64]))
        self.assertAlmostEqual(values[0,8],np.median(x[0,64:]))
        self.assertAlmostEqual(values[0,13],np.mean(x*x))
        self.assertNotEqual(values[0,13],x.var())

    def test_printed_fractal_formulas_and_undefined_values(self):
        x = np.array([2.,3.,1.,4.]*32)
        values = series_features(x[None])
        first = np.diff(x)
        changes = np.sum(np.sign(first[1:])!=np.sign(first[:-1]))
        self.assertAlmostEqual(values[0,32],1+np.log2(changes/(128-changes)))
        self.assertAlmostEqual(values[0,33],np.log10(np.abs(first).sum()/abs(x[-1]-x[0]))/np.log10(128))
        y=x.copy();y[-1]=y[0]
        self.assertTrue(np.isnan(series_features(y[None])[0,33]))

    def test_literal_higuchi_is_not_silently_changed_to_conventional(self):
        ramp=np.arange(128,dtype=float)
        self.assertAlmostEqual(literal_higuchi(ramp),0,places=12)
        self.assertTrue(np.isnan(literal_higuchi(-ramp)))
        literal=series_features(ramp[None],EquationConfig(higuchi='literal'))[0,19]
        conventional=series_features(ramp[None],EquationConfig(higuchi='conventional'))[0,19]
        self.assertAlmostEqual(literal,0,places=12)
        self.assertGreater(conventional,.9)

    def test_amplitude_uses_filtered_input_once_and_printed_time_derivative(self):
        x=(3*np.sin(2*np.pi*6*np.arange(128)/128))[None]
        envelope=np.sqrt(x*x+np.gradient(x,1/128,axis=-1)**2)
        expected=np.sum((envelope[:,1:]+envelope[:,:-1])*.5,axis=1)/127
        np.testing.assert_allclose(band_amplitude(x),expected,rtol=1e-13)

    def test_relative_power_has_raw_spectral_context_and_correct_tensor_positions(self):
        time=np.arange(128)/128
        raw=4+3*np.sin(2*np.pi*6*time)+np.sin(2*np.pi*20*time)
        x=np.repeat(raw[None,:,None],14,axis=2)
        bank=equation_features(x)
        frequency,psd=signal.periodogram(raw,fs=128,window='boxcar',detrend=False)
        selected=(frequency>=13)&(frequency<=30)
        expected=TRAPZ(psd[selected],frequency[selected])/TRAPZ(psd,frequency)
        self.assertEqual(bank.shape,(1,14,214))
        np.testing.assert_allclose(bank[:,:,142+34],expected,rtol=1e-12)
        layers=filter_layers(x)
        np.testing.assert_allclose(bank[0,:,34+35],band_amplitude(layers[1][0].T),rtol=1e-12)
        np.testing.assert_allclose(bank[0,:,0],raw.mean())

    def test_population_imputation_and_scaling_never_fit_evaluation_rows(self):
        x=np.array([[1.,np.nan,3.],[2.,4.,np.nan],[1000.,9000.,7.]])
        a=EquationTransform(1,3,'train').fit(x[:2])
        expected=a.transform(x[:2])
        x[2]*=1000
        b=EquationTransform(1,3,'train').fit(x[:2])
        np.testing.assert_array_equal(a.imputer_.statistics_,b.imputer_.statistics_)
        np.testing.assert_array_equal(a.scaler_.mean_,b.scaler_.mean_)
        np.testing.assert_array_equal(expected,b.transform(x[:2]))
        with self.assertRaises(ValueError):
            EquationTransform(1,3,'train','error').fit(x[:2])

    def test_within_window_missing_values_and_scaling_are_example_independent(self):
        x=np.array([[[1.,np.nan],[3.,np.nan]],[[4.,1.],[8.,3.]]])
        transform=EquationTransform(2,2).fit(x.reshape(2,-1))
        expected=StandardScaler().fit_transform(np.array([[1.,0.],[3.,0.]])).ravel()
        np.testing.assert_array_equal(transform.transform(x.reshape(2,-1))[0],expected)
        x[1]*=10000
        np.testing.assert_array_equal(transform.transform(x[:1].reshape(1,-1))[0],expected)

    def test_two_stages_have_published_ratios_without_an_inner_split(self):
        y=np.array([6]*480+[8]*444)
        fit,evaluation,later_fit,later_test=stage_splits(y)
        self.assertEqual([len(a) for a in [fit,evaluation,later_fit,later_test]],[739,185,184,740])
        self.assertFalse(set(fit)&set(evaluation))
        self.assertEqual(set(np.r_[fit,evaluation]),set(range(924)))
        self.assertFalse(set(later_fit)&set(later_test))

    def test_reference_knn_fitness_matches_sklearn_for_every_candidate_k(self):
        rng=np.random.default_rng(3)
        bank=rng.normal(size=(80,14,214));bank[:,0,28]=rng.integers(0,4,size=80)
        labels=rng.choice([6,8],size=80)
        genome=np.zeros(228,dtype=np.uint8);genome[[0,14+28]]=1
        fit,evaluation=np.arange(60),np.arange(60,80)
        actual=fitness(bank,labels,fit,evaluation,genome,'knn',42,'cpu',normalization='train')
        x=StandardScaler().fit(bank[fit,0,28,None]).transform(bank[:,0,28,None])
        self.assertEqual([row['k'] for row in actual['all_k_scores']],K_VALUES)
        for row in actual['all_k_scores']:
            predicted=KNeighborsClassifier(n_neighbors=row['k']).fit(x[fit],labels[fit]).predict(x[evaluation])
            self.assertEqual(row['accuracy'],np.mean(predicted==labels[evaluation]))

    def test_unfittable_degenerate_genome_is_a_penalty_not_a_measured_score(self):
        bank=np.ones((80,14,214))
        labels=np.array([6,8]*40)
        genome=np.zeros(228,dtype=np.uint8);genome[[0,14,15]]=1
        result=fitness(bank,labels,np.arange(60),np.arange(60,80),genome,'lda',42,'cpu')
        self.assertTrue(result['fitness_is_penalty'])
        self.assertIn('fit_failure',result)
        self.assertEqual(result['accuracy'],0.)

    def test_all_ten_factories_retain_default_parameters_and_global_rng_is_restored(self):
        self.assertEqual(set(CLASSIFIERS),{name for name in CLASSIFIERS if estimator(name) is not None})
        self.assertIsNone(estimator('random_forest').random_state)
        self.assertIsNone(estimator('random_forest').n_jobs)
        self.assertEqual(estimator('mlp').max_iter,200)
        rng=np.random.default_rng(2);x=rng.normal(size=(40,4));y=np.array([6,8]*20)
        np.random.seed(77);before=np.random.get_state()
        a=seeded_fit(estimator('decision_tree'),x,y,42)
        after=np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1])
        b=seeded_fit(estimator('decision_tree'),x,y,42)
        np.testing.assert_array_equal(a.predict(x),b.predict(x))


if __name__=='__main__':
    unittest.main()
