"""Different one-second EEG techniques: supervised filter-bank CSP, Haar features,
and a compact square/log power CNN. Learned steps are fitted inside group CV.
"""
import numpy as np
from scipy import signal, linalg
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .data import BANDS


def validate_windows(x):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (128, 14) or not np.isfinite(a).all():
        raise ValueError('Expected finite EEG windows with shape (N, 128, 14)')
    return a


def centered(x, reference):
    z = signal.detrend(validate_windows(x), axis=1, type='linear')
    if reference == 'car':
        z -= z.mean(axis=2, keepdims=True)
    elif reference != 'none':
        raise ValueError('Unknown channel reference')
    return z


def filterbank_covariances(x, reference='car'):
    z = centered(x, reference)
    result = []
    for band in BANDS:
        sos = signal.butter(4, band, btype='bandpass', fs=128, output='sos')
        a = signal.sosfiltfilt(sos, z, axis=1)
        a -= a.mean(axis=1, keepdims=True)
        c = np.einsum('nti,ntj->nij', a, a)/127
        result.append(c/np.trace(c, axis1=-2, axis2=-1)[:, None, None].clip(1e-12))
    return np.stack(result, axis=1)


class FilterBankCSP(TransformerMixin, BaseEstimator):
    """One-vs-rest CSP fitted exclusively from the current training fold labels."""
    def __init__(self, pairs=1, reference='car', shrinkage=.1):
        self.pairs = pairs
        self.reference = reference
        self.shrinkage = shrinkage

    def fit(self, x, y):
        if self.pairs not in [1, 2] or not 0 < self.shrinkage < 1:
            raise ValueError('Unsupported CSP regularization or number of pairs')
        cov = filterbank_covariances(x, self.reference)
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        if len(labels) != len(cov) or len(self.classes_) < 2:
            raise ValueError('CSP needs matching labels from at least two classes')
        class_means = np.stack([cov[labels == c].mean(axis=0) for c in self.classes_])
        self.filters_ = np.empty((5, len(self.classes_), 14, 2*self.pairs))
        eye = np.eye(14)
        for b in range(5):
            for k in range(len(self.classes_)):
                positive = class_means[k, b]
                negative = np.delete(class_means[:, b], k, axis=0).mean(axis=0)
                positive = (1-self.shrinkage)*positive + self.shrinkage*eye/14
                negative = (1-self.shrinkage)*negative + self.shrinkage*eye/14
                _, vectors = linalg.eigh(positive, positive+negative)
                indices = [*range(self.pairs), *range(14-self.pairs, 14)]
                self.filters_[b, k] = vectors[:, indices]
        return self

    def transform(self, x):
        check_is_fitted(self, 'filters_')
        cov = filterbank_covariances(x, self.reference)
        variance = np.einsum('bkif,nbij,bkjf->nbkf', self.filters_, cov, self.filters_, optimize=True).clip(1e-12)
        relative = variance/variance.sum(axis=-1, keepdims=True)
        return np.log(relative).reshape(len(cov), -1)


def haar_coefficients(x):
    """Orthonormal six-level Haar decomposition; time dimension must be 128."""
    current = np.asarray(x, dtype=np.float64)
    if current.ndim != 3 or current.shape[1] != 128:
        raise ValueError('Haar input must have 128 time samples')
    parts = []
    for _ in range(6):
        even, odd = current[:, ::2], current[:, 1::2]
        parts.append((even-odd)/np.sqrt(2))
        current = (even+odd)/np.sqrt(2)
    return [*parts, current]


class HaarFeatures(TransformerMixin, BaseEstimator):
    """Cycle-spun Haar log energy, relative energy, and coefficient entropy."""
    def __init__(self, reference='none'):
        self.reference = reference

    def fit(self, x, y=None):
        validate_windows(x)
        return self

    def transform(self, x):
        z = centered(x, self.reference)
        sos = signal.butter(4, [.5, 40], btype='bandpass', fs=128, output='sos')
        z = signal.sosfiltfilt(sos, z, axis=1)
        vectors = []
        for shift in range(4):
            parts = haar_coefficients(np.roll(z, shift, axis=1))
            energies = np.stack([(part**2).sum(axis=1) for part in parts], axis=1)
            relative = energies/(energies.sum(axis=1, keepdims=True)+1e-12)
            entropy = []
            for part in parts:
                p = part**2
                p /= p.sum(axis=1, keepdims=True)+1e-12
                entropy.append(-(p*np.log(p+1e-12)).sum(axis=1)/np.log(part.shape[1]))
            vectors.append(np.concatenate([np.log(energies+1e-12).reshape(len(z), -1),
                                           relative.reshape(len(z), -1),
                                           np.stack(entropy, axis=1).reshape(len(z), -1)], axis=1))
        return np.mean(vectors, axis=0)


def cnn_inputs(x):
    z = centered(x, 'car')
    sos = signal.butter(4, [.5, 40], btype='bandpass', fs=128, output='sos')
    z = signal.sosfiltfilt(sos, z, axis=1)
    z /= z.std(axis=1, keepdims=True).clip(1e-6)
    return np.clip(z, -8, 8).transpose(0, 2, 1)[:, None].astype(np.float32).copy()


def make_power_network():
    import torch
    from torch import nn

    class Square(nn.Module):
        def forward(self, x):
            return x.square()

    class SafeLog(nn.Module):
        def forward(self, x):
            return torch.log(x.clamp_min(1e-6))

    return nn.Sequential(nn.Conv2d(1, 16, (1, 25), padding=(0, 12), bias=False),
                         nn.Conv2d(16, 16, (14, 1), bias=False), nn.BatchNorm2d(16),
                         Square(), nn.AvgPool2d((1, 128)), SafeLog(),
                         nn.Dropout(.5), nn.Flatten(), nn.Linear(16, 9))


class ShallowPowerClassifier(ClassifierMixin, BaseEstimator):
    """Fixed-duration neural training; fold scores do not choose checkpoints."""
    def __init__(self, epochs=50, seed=42, batch_size=64, learning_rate=.0008):
        self.epochs = epochs
        self.seed = seed
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self, x, y):
        import torch
        from torch import nn
        from torch.utils.data import TensorDataset, DataLoader
        torch.set_num_threads(4)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        labels = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(labels)
        if not np.array_equal(self.classes_, np.arange(9)):
            raise ValueError('All nine classes must be represented in the training fold')
        a = torch.from_numpy(cnn_inputs(x))
        target = torch.from_numpy(labels)
        net = make_power_network()
        weights = len(labels)/(9*np.bincount(labels, minlength=9))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32), label_smoothing=.05)
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.learning_rate, weight_decay=.01)
        loader = DataLoader(TensorDataset(a, target), batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed))
        self.history_ = []
        for epoch in range(1, self.epochs+1):
            net.train(); total = correct = 0
            for xb, yb in loader:
                # Fixed training-only augmentation; neither held-out signals nor labels enter it.
                xb = torch.roll(xb, int(torch.randint(-8, 9, ()).item()), dims=-1)
                xb = xb*(.8+.4*torch.rand(len(xb), 1, 14, 1)) + .03*torch.randn_like(xb)
                optimizer.zero_grad()
                logits = net(xb); loss = loss_fn(logits, yb)
                loss.backward(); optimizer.step()
                total += loss.item()*len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
            self.history_.append({'epoch': epoch, 'training_loss': total/len(labels), 'training_accuracy': correct/len(labels)})
            if epoch % 10 == 0:
                print('ShallowPowerCNN', self.history_[-1], flush=True)
        # Store state tensors only: locally defined activation classes are never pickled.
        self.state_ = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        return self

    def predict_proba(self, x):
        import torch
        check_is_fitted(self, 'state_')
        torch.set_num_threads(4)
        net = make_power_network(); net.load_state_dict(self.state_); net.eval()
        a = torch.from_numpy(cnn_inputs(x)); output = []
        with torch.no_grad():
            for start in range(0, len(a), 256):
                output.append(net(a[start:start+256]).softmax(1).numpy())
        return np.concatenate(output)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]
