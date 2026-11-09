"""Replay adversarial paper-reproduction probes without rerunning or replacing a GA.

Historical Torch scoring is checked directly, even after the CPU scorer fix.
All stage-two scores are declared diagnostics, never model-selection criteria.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import antropy as ant
import numpy as np
import pandas as pd
import torch
from scipy import signal
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .author_features import (author_features, load_layers, published_genome,
                              reference_library, source_spec)
from .author_window_scaling import WithinWindowScaler
from .data import BANDS
from .feature_provenance import feature_fingerprint, fingerprint_sha
from .paper_features import COLUMNS
from .paper_ga import K_VALUES, TorchKNN, genome_columns
from .train_paper_binary import array_sha, dump


def feature_checks(source, bank, data, out):
    indices, channels = [0, 479, 480, 923], [0, 6, 13]
    layers = load_layers(data, source.iloc[indices])
    upstream = reference_library()
    functions = ['mean','std','coeff_var','median','mode','max','min','q1','q3','IQR',
        'kurt','skewness',None,'activity_hjorth_param','mobility_hjorth_param',
        'complexity_hjorth_param',None,None,None,None,'total_power_spectral_density',
        'centroid_power_spectral_density','determinism','trapping_time','diagonal_line_entropy',
        'average_diagonal_line_length','compute_recurrence_rate','sef_25','sef_50','sef_75',
        'hurst_exponent',None,None,None]
    restored = {12:ant.detrended_fluctuation,16:lambda x:ant.perm_entropy(x,normalize=True),
        17:ant.app_entropy,18:lambda x:ant.spectral_entropy(x,sf=128,method='fft',normalize=True),
        19:ant.higuchi_fd,31:lambda x:ant.svd_entropy(x,normalize=True),
        32:ant.petrosian_fd,33:ant.katz_fd}
    comparisons, upstream_count, restored_count, maximum = 0, 0, 0, 0.
    mismatches = []
    for local, index in enumerate(indices):
        for channel in channels:
            offset = 0
            for layer, band in enumerate([None, *BANDS]):
                x = layers[local,layer,:,channel]
                expected = [getattr(upstream,name)(x) if name else restored[i](x)
                            for i,name in enumerate(functions)]
                upstream_count += 26
                restored_count += 8
                if band is not None:
                    name = ['delta','theta','alpha','beta','gamma'][layer-1]
                    expected += [getattr(upstream,'relative_'+name+'_power')(x),
                                 getattr(upstream,name+'_amplitude')(x)]
                    upstream_count += 2
                expected = np.asarray(expected)
                got = bank[index,channel,offset:offset+len(expected)]
                for column in np.flatnonzero(~np.isclose(got,expected,rtol=1e-9,atol=1e-9)):
                    mismatches.append({'source_row':index,'channel':channel,
                        'column':COLUMNS[offset+column], 'actual':float(got[column]),
                        'reference':float(expected[column])})
                maximum = max(maximum,float(np.max(np.abs(got-expected))))
                comparisons += len(expected)
                offset += len(expected)
        print(f'Feature assembly: source row {index} checked',flush=True)
    filters = []
    for layer, band in enumerate(BANDS,1):
        b,a = signal.butter(4,band,fs=128,btype='bandpass')
        expected = signal.filtfilt(b,a,layers[:,0],axis=1)
        actual = layers[:,layer]
        filters.append({'band':list(band),'max_absolute_error':float(np.max(np.abs(expected-actual))),
                        'rms_error':float(np.sqrt(np.mean((expected-actual)**2)))})
    x = layers[0,0,:,0]
    ordered = np.sort(x)
    paper_equations = {
        'raw_coefficient_variation': {'appendix_A3':float(x.std()/x.mean()),'implemented':float(bank[0,0,2])},
        'raw_activity': {'appendix_A14':float(np.mean(x*x)),'implemented':float(bank[0,0,13])},
        'raw_q1': {'appendix_A8_median_lower_half':float(np.median(ordered[:64])),
                   'implemented_percentile':float(bank[0,0,7])},
        'raw_q3': {'appendix_A9_median_upper_half':float(np.median(ordered[64:])),
                   'implemented_percentile':float(bank[0,0,8])},
        'beta_relative_power_input': {'public_function_on_raw':float(upstream.relative_beta_power(x)),
                                     'public_function_on_beta_layer':float(bank[0,0,34+3*36+34])},
        'delta_amplitude_input': {'public_function_on_raw_once_filtered':float(upstream.delta_amplitude(x)),
                                  'public_function_on_delta_layer_twice_filtered':float(bank[0,0,34+35])},
        'interpretation': 'Library/input choices differ from printed equations; complete original assembly is unknown.',
    }
    result = {'source_rows':indices,'channels':channels,'comparisons':comparisons,
        'available_public_function_comparisons':upstream_count,
        'restored_antropy_call_comparisons_not_independent_author_reference':restored_count,
        'maximum_absolute_error':maximum,'mismatches':mismatches,
        'per_window_butterworth_vs_supplied_layers':filters,
        'paper_equation_and_input_flow_examples':paper_equations}
    dump(out/'feature_checks.json',result)
    assert not mismatches
    return layers, comparisons


def neighbor_checks(root, bank, out):
    historical, sparse = [], []
    for child in ['', 'paper_window_diagnostic']:
        directory = root/child
        manifest = pd.read_csv(directory/'manifest.csv')
        y = manifest.label.to_numpy()
        fit = np.flatnonzero(manifest.search_role=='fit')
        val = np.flatnonzero(manifest.search_role=='fitness_validation')
        evaluations = pd.read_csv(directory/'knn_search_evaluations.csv',dtype={'genome':str})
        genomes = np.array([list(row) for row in evaluations.genome],dtype=np.uint8)
        indices = np.union1d(np.linspace(0,len(genomes)-1,64,dtype=int),
                            np.flatnonzero(genomes[:,:14].sum(axis=1)<=3))
        disagreements = []
        for index in indices:
            selected = bank.reshape(len(bank),-1)[:,genome_columns(genomes[index])]
            scaler = StandardScaler().fit(selected[fit])
            a,b = scaler.transform(selected[fit]),scaler.transform(selected[val])
            candidates = TorchKNN(device='cpu').fit(a,y[fit]).predict_many(b,K_VALUES)
            for k,predicted in zip(K_VALUES,candidates):
                expected = KNeighborsClassifier(n_neighbors=k).fit(a,y[fit]).predict(b)
                if np.any(predicted!=expected):
                    disagreements.append({'candidate_index':int(index),'k':k,
                                          'predictions_different':int(np.sum(predicted!=expected))})
        historical.append({'run':child or 'grouped','candidate_indices':indices.tolist(),
            'genomes_checked':len(indices),'prediction_sets_checked':len(indices)*7,
            'disagreements':disagreements})
        print(f'Historical neighbor checks: {child or "grouped"}, {len(indices)} genomes',flush=True)
        if child:
            continue
        # Five legal but deliberately sparse masks, outside the saved searches.
        for channel,column in [(0,27),(0,28),(0,29),(6,27),(0,61)]:
            selected = bank[:,channel,column,None]
            scaler = StandardScaler().fit(selected[fit])
            a,b = scaler.transform(selected[fit]),scaler.transform(selected[val])
            candidates = TorchKNN(device='cpu').fit(a,y[fit]).predict_many(b,K_VALUES)
            for k,predicted in zip(K_VALUES,candidates):
                reference = KNeighborsClassifier(n_neighbors=k).fit(a,y[fit])
                expected = reference.predict(b)
                sparse.append({'channel':channel,'feature':COLUMNS[column],'k':k,
                    'prediction_disagreements':int(np.sum(predicted!=expected)),
                    'torch_fitness':float(np.mean(predicted==y[val])),
                    'sklearn_fitness':float(np.mean(expected==y[val])),
                    'sklearn_method':reference._fit_method})
    dump(out/'neighbor_checks.json',{'historical_samples':historical,'sparse_mask_cases':sparse,
        'sparse_cases_not_present_in_historical_runs':True,
        'scope': 'Sampled historical genomes, not exhaustive parity of every searched genome'})
    return sum(row['genomes_checked'] for row in historical),sum(row['prediction_disagreements']>0 for row in sparse)


def cache_check(layers, out):
    baseline_spec, baseline_fingerprint = source_spec(),feature_fingerprint()
    baseline = author_features(layers[:1])
    # The legacy key omitted band limits even though they change feature values.
    with patch('eegis.author_features.BANDS',[(.5,3.5),*BANDS[1:]]):
        changed_spec, changed_fingerprint = source_spec(),feature_fingerprint()
        changed = author_features(layers[:1])
    result = {'source_data_unchanged':True,'author_features_source_file_unchanged':True,
        'legacy_spec_key_unchanged':baseline_spec==changed_spec,
        'feature_values_changed':bool(np.any(baseline!=changed)),
        'new_dependency_key_changes':fingerprint_sha(baseline_fingerprint)!=fingerprint_sha(changed_fingerprint),
        'changed_values':int(np.sum(baseline!=changed))}
    dump(out/'cache_check.json',result)
    assert result['legacy_spec_key_unchanged'] and result['feature_values_changed'] and result['new_dependency_key_changes']


def stage_two(root, source, bank, out):
    y = source.label.to_numpy()
    train,test = train_test_split(np.arange(len(source)),test_size=.8,random_state=42,stratify=y)
    train,test = np.sort(train),np.sort(test)
    results,predictions = [],[]
    for child in ['', 'within_window_grouped', 'paper_window_diagnostic']:
        directory = root/child
        selected = json.loads((directory/'knn_selection.json').read_text())
        normalization = json.loads((directory/'results.json').read_text()).get('normalization','train')
        genome = np.array(selected['genome'],dtype=np.uint8)
        x = bank.reshape(len(bank),-1)[:,genome_columns(genome)]
        if normalization=='within-window':
            scaler = WithinWindowScaler(int(genome[:14].sum()),int(genome[14:].sum()))
        else:
            scaler = StandardScaler()
        scaler.fit(x[train])
        a,b = scaler.transform(x[train]),scaler.transform(x[test])
        for k in [selected['k'],9]:
            predicted = KNeighborsClassifier(n_neighbors=k).fit(a,y[train]).predict(b)
            row = {'source_run':child or 'grouped','normalization':normalization,'k':k,
                'train_windows':len(train),'test_windows':len(test),'diagnostic_only':True,
                'correct':int(np.sum(predicted==y[test])),
                'accuracy':float(accuracy_score(y[test],predicted)),
                'macro_f1':float(f1_score(y[test],predicted,labels=[6,8],average='macro'))}
            results.append(row)
            part = source.iloc[test][['path','label','overlap_group']].copy()
            part['source_run'],part['k'],part['prediction'] = row['source_run'],k,predicted
            predictions.append(part)
    pd.DataFrame(results).to_csv(out/'stage_two_diagnostics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'stage_two_predictions.csv',index=False)
    dump(out/'stage_two_audit.json',{'train_rows':train.tolist(),'test_rows':test.tolist(),
        'shared_overlap_groups':len(set(source.overlap_group.iloc[train])&set(source.overlap_group.iloc[test])),
        'diagnostic_only':True,'mask_selection_labels_reused':True,
        'author_original_split_indices_unknown':True,'winner_selected':False,
        'reason': 'Frozen GA masks can already encode labels in this later split; EEG overlap also crosses it.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=Path('artifacts_author_reproduction'))
    parser.add_argument('--out',type=Path,default=Path('artifacts_adversarial_review'))
    parser.add_argument('--data',default='EEGIS - Electroencephalogram Imagined Speech dataset')
    parser.add_argument('--paper',type=Path,default=Path('Feature_Generation_with_Genetic_Algorithms_for_Ima.pdf'))
    args = parser.parse_args()
    out,root = args.out,args.artifacts
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'plan.json',{'historical_genomes':'64 evenly spaced candidates plus all with <=3 electrodes in each Torch-backed run',
        'feature_checks':'rows 0,479,480,923; channels AF3,O1,AF4; all 214 values',
        'sparse_counterexamples':[[0,27],[0,28],[0,29],[6,27],[0,61]],
        'stage_two':'all three frozen GA masks; 20/80 stratified seed42; selected k and k9; both reported',
        'stage_two_diagnostic_only':True,'winner_selected':False,'ga_rerun':False,
        'threadpool_limit':4,'paper_sha256':hashlib.sha256(args.paper.read_bytes()).hexdigest(),
        'implementation_dependencies':feature_fingerprint(),
        'review_runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'training_code_sha256':hashlib.sha256(Path(__file__).with_name('author_reproduction.py').read_bytes()).hexdigest()})
    source = pd.read_csv(root/'source_manifest.csv')
    with np.load(root/'feature_cache.npz',allow_pickle=False) as cache:
        bank = cache['features']
    assert bank.shape==(924,14,214)
    assert array_sha(bank)==json.loads((root/'feature_audit.json').read_text())['feature_array_sha256']
    text = subprocess.run(['pdftotext','-raw',str(args.paper),'-'],check=True,capture_output=True,text=True).stdout
    matches = re.findall(r'best\s+genome\s*=\s*\[([01,\s]+)\]',text)
    assert matches
    for match in matches:
        np.testing.assert_array_equal(np.array(re.findall('[01]',match),dtype=np.uint8),published_genome())
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        layers,comparisons = feature_checks(source,bank,args.data,out)
        genomes,sparse_failures = neighbor_checks(root,bank,out)
        cache_check(layers,out)
        stage_two(root,source,bank,out)
    dump(out/'verification.json',{'published_genome_matches_pdf_bitwise':True,
        'historical_feature_array_hash_matches':True,'feature_comparisons':comparisons,
        'historical_candidate_genomes_sampled':genomes,'sparse_prediction_sets_with_counterexamples':sparse_failures,
        'cache_band_setting_counterexample_reproduced_and_new_guard_detects_change':True,
        'stage_two_diagnostic_cases':6,'ga_rerun':False,'winner_selected':False})


if __name__=='__main__':
    main()
