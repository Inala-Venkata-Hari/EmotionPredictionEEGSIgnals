"""Binary paper-style channel/feature genetic search using installed ML libraries.

The paper uses GA generations, not neural epochs. kNN can run on CPU or CUDA;
the other nine scikit-learn estimators and SciPy preprocessing run on CPU.
"""
import copy
import time

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from .paper_features import paper_features

K_VALUES = [5, 7, 13, 15, 17, 21, 37]
CLASSIFIERS = ['knn', 'logistic', 'random_forest', 'decision_tree', 'gradient_boosting',
               'adaboost', 'naive_bayes', 'lda', 'mlp', 'svm']
CLASS_IDS = [6, 8]
CLASS_NAMES = ['No', 'Sí']


def resolve_device(requested='auto'):
    if requested not in ['auto', 'cpu', 'cuda']:
        raise ValueError('Device must be auto, cpu, or cuda')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested, but this PyTorch build/driver exposes no CUDA device')
    return 'cuda' if requested != 'cpu' and torch.cuda.is_available() else 'cpu'


def genome_columns(genome):
    genome = np.asarray(genome)
    if genome.shape != (228,) or not np.isin(genome, [0, 1]).all():
        raise ValueError('Expected a 228-bit genome')
    if not genome[:14].any() or not genome[14:].any():
        raise ValueError('Genome must select at least one channel and one feature')
    return np.flatnonzero((genome[:14, None]*genome[None, 14:]).reshape(-1))


class TorchKNN(ClassifierMixin, BaseEstimator):
    """Uniform Euclidean kNN, with stable training-row tie order and CPU checkpoints."""
    def __init__(self, n_neighbors=7, device='auto', batch_size=256):
        self.n_neighbors = n_neighbors
        self.device = device
        self.batch_size = batch_size

    def fit(self, x, y):
        a, labels = np.asarray(x, dtype=np.float64), np.asarray(y)
        if a.ndim != 2 or len(a) != len(labels) or not np.isfinite(a).all():
            raise ValueError('Invalid kNN training arrays')
        if not 1 <= self.n_neighbors <= len(a):
            raise ValueError('Invalid neighbor count')
        self.classes_, encoded = np.unique(labels, return_inverse=True)
        self.x_ = a.copy()
        self.y_ = encoded
        self.n_features_in_ = a.shape[1]
        self.fit_device_ = resolve_device(self.device)
        return self

    def neighbors(self, x, max_neighbors=None):
        check_is_fitted(self, 'x_')
        query = np.asarray(x, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != self.n_features_in_ or not np.isfinite(query).all():
            raise ValueError('Invalid kNN query arrays')
        device = resolve_device(self.device)
        k = self.n_neighbors if max_neighbors is None else max_neighbors
        if not 1 <= k <= len(self.x_):
            raise ValueError('Invalid neighbor count')
        result = []
        with torch.inference_mode():
            train = torch.as_tensor(self.x_, device=device, dtype=torch.float64)
            norm = train.square().sum(dim=1)
            for start in range(0, len(query), self.batch_size):
                q = torch.as_tensor(query[start:start+self.batch_size], device=device, dtype=torch.float64)
                distances = (q.square().sum(dim=1)[:, None]+norm[None]-2*q@train.T).clamp_min(0)
                indices = torch.argsort(distances, dim=1, stable=True)[:, :k]
                result.append(indices.cpu().numpy())
        return np.concatenate(result)

    def predict_many(self, x, ks):
        indices = self.neighbors(x, max(ks))
        encoded = self.y_[indices]
        predictions = []
        for k in ks:
            counts = np.stack([(encoded[:, :k] == c).sum(axis=1) for c in range(len(self.classes_))], axis=1)
            predictions.append(self.classes_[counts.argmax(axis=1)])
        return predictions

    def predict(self, x):
        return self.predict_many(x, [self.n_neighbors])[0]

    def predict_proba(self, x):
        encoded = self.y_[self.neighbors(x)]
        return np.stack([(encoded == c).mean(axis=1) for c in range(len(self.classes_))], axis=1)


def make_estimator(name, seed=42, device='auto', k=7):
    if name == 'knn':
        return TorchKNN(n_neighbors=k, device=device)
    factories = {
        'logistic': lambda: LogisticRegression(random_state=seed),
        'random_forest': lambda: RandomForestClassifier(random_state=seed, n_jobs=4),
        'decision_tree': lambda: DecisionTreeClassifier(random_state=seed),
        'gradient_boosting': lambda: GradientBoostingClassifier(random_state=seed),
        'adaboost': lambda: AdaBoostClassifier(random_state=seed),
        'naive_bayes': GaussianNB,
        'lda': LinearDiscriminantAnalysis,
        'mlp': lambda: MLPClassifier(random_state=seed),
        'svm': SVC,
    }
    if name not in factories:
        raise ValueError(f'Unknown classifier: {name}')
    return factories[name]()


class PaperBinaryModel:
    """Portable full inference pipeline: raw EEG -> features -> mask -> scaler -> model."""
    def __init__(self, genome, scaler, estimator):
        self.genome = np.asarray(genome, dtype=np.uint8)
        self.scaler = scaler
        self.estimator = estimator
        self.classes_ = np.asarray(CLASS_IDS)

    def predict_features(self, features):
        selected = np.asarray(features).reshape(len(features), -1)[:, genome_columns(self.genome)]
        return self.estimator.predict(self.scaler.transform(selected))

    def predict(self, windows):
        return self.predict_features(paper_features(windows))


def fitness_predictions(features, labels, train, validation, genome, classifier,
                        seed=42, device='auto'):
    columns = genome_columns(genome)
    flat = np.asarray(features).reshape(len(features), -1)[:, columns]
    scaler = StandardScaler().fit(flat[train])
    a, b = scaler.transform(flat[train]), scaler.transform(flat[validation])
    model = make_estimator(classifier, seed, device)
    model.fit(a, labels[train])
    if classifier == 'knn':
        choices = [(k, p) for k, p in zip(K_VALUES, model.predict_many(b, K_VALUES))]
    else:
        choices = [(None, model.predict(b))]
    records = []
    for k, predicted in choices:
        records.append({'k': k, 'accuracy': float(accuracy_score(labels[validation], predicted)),
                        'macro_f1': float(f1_score(labels[validation], predicted, labels=CLASS_IDS,
                                                  average='macro', zero_division=0))})
    # Accuracy is the paper's sole objective; earliest k wins exact ties.
    best = max(range(len(records)), key=lambda i: records[i]['accuracy'])
    return {**records[best], 'all_k_scores': records}


def genetic_search(features, labels, train, validation, classifier='knn', generations=50,
                   population_size=100, seed=42, device='auto', mutation_rate=None,
                   progress=None, history_path=None, evaluator=None):
    if generations < 1 or population_size < 2 or population_size % 2:
        raise ValueError('Need positive generations and an even population size >=2')
    if len(set(train) & set(validation)):
        raise ValueError('Fitness fit and validation rows overlap')
    if min(np.bincount(np.searchsorted(CLASS_IDS, labels[train]), minlength=2)) == 0:
        raise ValueError('Both classes must appear in fitness training')
    if classifier == 'knn' and len(train) < max(K_VALUES):
        raise ValueError('At least 37 fitness training windows are required')
    mutation_rate = 1/228 if mutation_rate is None else mutation_rate
    if not 0 <= mutation_rate <= 1:
        raise ValueError('Mutation rate must be between zero and one')
    rng = np.random.default_rng(seed)
    population = rng.integers(0, 2, size=(population_size, 228), dtype=np.uint8)
    cache, history, population_audit = {}, [], []
    def repair(genome):
        if not genome[:14].any():
            genome[rng.integers(14)] = 1
        if not genome[14:].any():
            genome[14+rng.integers(214)] = 1
        return genome
    population = np.stack([repair(g) for g in population])
    def evaluate(genome):
        key = genome.tobytes()
        if key not in cache:
            cache[key] = (fitness_predictions(features, labels, train, validation, genome,
                                              classifier, seed, device) if evaluator is None
                          else evaluator(features, labels, train, validation, genome, classifier, seed, device))
        return cache[key]
    started = time.time()
    for generation in range(generations+1):
        if generation:
            # Rank-weighted mating; parent survival is deterministic top-100 elitism.
            weights = np.arange(population_size, 0, -1, dtype=np.float64)
            weights /= weights.sum()
            children = []
            for _ in range(population_size//2):
                parents = rng.choice(population_size, 2, replace=False, p=weights)
                first, second = population[parents]
                crossover = rng.random(228) < .5
                for child in [np.where(crossover, first, second), np.where(crossover, second, first)]:
                    child = child.astype(np.uint8)
                    child ^= (rng.random(228) < mutation_rate).astype(np.uint8)
                    children.append(repair(child))
            candidates = np.concatenate([population, np.stack(children)])
        else:
            candidates = population
        scores = [evaluate(g) for g in candidates]
        order = sorted(range(len(candidates)), key=lambda i: -scores[i]['accuracy'])
        survivors = order[:population_size]
        population = candidates[survivors].copy()
        scores = [scores[i] for i in survivors]
        best = scores[0]
        row = {'generation': generation, 'best_validation_accuracy': best['accuracy'],
               'best_validation_macro_f1': best['macro_f1'], 'mean_survivor_accuracy': float(np.mean([s['accuracy'] for s in scores])),
               'best_k': best['k'], 'selected_channels': int(population[0, :14].sum()),
               'selected_feature_types': int(population[0, 14:].sum()),
               'unique_evaluations': len(cache), 'elapsed_seconds': time.time()-started}
        history.append(row)
        population_audit.append({'generation': generation, 'genomes': [g.tolist() for g in population],
                                 'accuracies': [s['accuracy'] for s in scores]})
        if history_path:
            import pandas as pd
            pd.DataFrame(history).to_csv(history_path, index=False)
        if progress:
            progress(f'{classifier} generation {generation:02d}/{generations}: '
                     f'fitness={best["accuracy"]:.4f}; mean={row["mean_survivor_accuracy"]:.4f}; '
                     f'k={best["k"]}; unique={len(cache)}; elapsed={row["elapsed_seconds"]:.1f}s')
    result = copy.deepcopy(evaluate(population[0]))
    result.update({'genome': population[0].tolist(), 'generations': generations,
                   'population_size': population_size, 'unique_evaluations': len(cache),
                   'mutation_rate': mutation_rate,
                   'evaluations': [{'genome': list(key), **value} for key, value in cache.items()]})
    return result, history, population_audit


def fit_final(features, labels, train, genome, classifier, seed=42, device='auto', k=7):
    columns = genome_columns(genome)
    selected = features.reshape(len(features), -1)[train][:, columns]
    scaler = StandardScaler().fit(selected)
    estimator = make_estimator(classifier, seed, device, k=k)
    estimator.fit(scaler.transform(selected), labels[train])
    return PaperBinaryModel(genome, scaler, estimator)
