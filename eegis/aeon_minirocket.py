"""MiniRocket numeric helpers adapted from the BSD-licensed Aeon implementation.
Source commit: 98c13323251a011722de2d6f68a772554af9b9ea
Source: https://github.com/aeon-toolkit/aeon/blob/98c13323251a011722de2d6f68a772554af9b9ea/aeon/transformations/collection/convolution_based/_minirocket.py
Aeon notes original MiniRocket adaptation with owner permission.
Numerical helper bodies are unchanged; the sklearn/EEG adapter is local.

BSD 3-Clause License

Copyright (c) The aeon developers.
Copyright (c) 2022 The sktime developers.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
from itertools import combinations
import numpy as np
from numba import njit,prange,vectorize,get_num_threads,set_num_threads
from sklearn.base import BaseEstimator,TransformerMixin
from .accuracy_features import prepared,validate


class MiniRocket(TransformerMixin,BaseEstimator):
    """Sklearn EEG adapter around Aeon's unchanged multivariate numeric helpers.

    Learned convolution thresholds use only the data supplied to fit().
    Inputs are (examples, time samples, 14 channels), as in this project.
    """
    _indices=np.array([*combinations(np.arange(9),3)],dtype=np.int32)

    def __init__(self,n_kernels=10000,mode='filtered',seed=42):
        self.n_kernels=n_kernels;self.mode=mode;self.seed=seed

    def _input(self,x):
        x=validate(x)
        z=x.copy() if self.mode=='raw' else prepared(x)
        if self.mode=='derivative':z=np.diff(z,axis=1)
        elif self.mode not in ['raw','filtered']:raise ValueError('Unknown input representation')
        z-=z.mean(axis=1,keepdims=True)
        z/=z.std(axis=1,keepdims=True).clip(1e-9)
        return np.ascontiguousarray(z.transpose(0,2,1),dtype=np.float32)

    def fit(self,x,y=None):
        x=validate(x);self.samples_=x.shape[1]
        state=np.random.get_state()
        try:self.parameters_=_static_fit(self._input(x),max(84,self.n_kernels),32,self.seed)
        finally:np.random.set_state(state)
        return self

    def transform(self,x):
        x=validate(x)
        if x.shape[1]!=self.samples_:raise ValueError('Input duration differs from fitting data')
        prior=get_num_threads()
        try:
            set_num_threads(4)
            return _static_transform_multi(self._input(x),self.parameters_,self._indices)
        finally:set_num_threads(prior)

def _fit_dilations(n_timepoints, n_features, max_dilations_per_kernel):
    n_kernels = 84
    n_features_per_kernel = n_features // n_kernels
    true_max_dilations_per_kernel = min(n_features_per_kernel, max_dilations_per_kernel)
    multiplier = n_features_per_kernel / true_max_dilations_per_kernel
    max_exponent = np.log2((n_timepoints - 1) / (9 - 1))
    dilations, n_features_per_dilation = np.unique(
        np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
            np.int32
        ),
        return_counts=True,
    )
    n_features_per_dilation = (n_features_per_dilation * multiplier).astype(
        np.int32
    )  # this is a vector
    remainder = n_features_per_kernel - np.sum(n_features_per_dilation)
    i = 0
    while remainder > 0:
        n_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(n_features_per_dilation)
    return dilations, n_features_per_dilation


def _quantiles(n):
    return np.array(
        [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
    )


def _static_fit(X, n_features=10_000, max_dilations_per_kernel=32, seed=None):
    if seed is not None:
        np.random.seed(seed)
    _, n_channels, n_timepoints = X.shape
    n_kernels = 84
    dilations, n_features_per_dilation = _fit_dilations(
        n_timepoints, n_features, max_dilations_per_kernel
    )
    n_features_per_kernel = np.sum(n_features_per_dilation)
    quantiles = _quantiles(n_kernels * n_features_per_kernel)
    n_dilations = len(dilations)
    n_combinations = n_kernels * n_dilations
    max_n_channels = min(n_channels, 9)
    max_exponent = np.log2(max_n_channels + 1)
    n_channels_per_combination = (
        2 ** np.random.uniform(0, max_exponent, n_combinations)
    ).astype(np.int32)
    channel_indices = np.zeros(n_channels_per_combination.sum(), dtype=np.int32)
    n_channels_start = 0
    for combination_index in range(n_combinations):
        n_channels_this_combination = n_channels_per_combination[combination_index]
        n_channels_end = n_channels_start + n_channels_this_combination
        channel_indices[n_channels_start:n_channels_end] = np.random.choice(
            n_channels, n_channels_this_combination, replace=False
        )
        n_channels_start = n_channels_end
    biases = _fit_biases(
        X,
        n_channels_per_combination,
        channel_indices,
        dilations,
        n_features_per_dilation,
        quantiles,
        MiniRocket._indices,
        seed,
    )
    return (
        n_channels_per_combination,
        channel_indices,
        dilations,
        n_features_per_dilation,
        biases,
    )


@vectorize(nopython=True, cache=True)
def _PPV(a, b):
    if a > b:
        return 1
    return 0


@njit(
    fastmath=True,
    parallel=True,
    cache=True,
)
def _static_transform_multi(X, parameters, indices):
    n_cases, n_channels, n_timepoints = X.shape
    (
        n_channels_per_combination,
        channel_indices,
        dilations,
        n_features_per_dilation,
        biases,
    ) = parameters
    n_kernels = len(indices)
    n_dilations = len(dilations)
    n_features = n_kernels * np.sum(n_features_per_dilation)
    features = np.zeros((n_cases, n_features), dtype=np.float32)
    for i in prange(n_cases):
        _X = X[i]
        A = -_X
        G = 3 * _X
        f_start = 0
        comb = 0
        n_channels_start = 0
        for j in range(n_dilations):
            _padding0 = j % 2
            dilation = dilations[j]
            padding = (8 * dilation) // 2
            n_features_this_dilation = n_features_per_dilation[j]
            C_alpha = np.zeros((n_channels, n_timepoints), dtype=np.float32)
            C_alpha[:] = A
            C_gamma = np.zeros((9, n_channels, n_timepoints), dtype=np.float32)
            C_gamma[4] = G
            start = dilation
            end = n_timepoints - padding
            for gamma_index in range(4):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]
                end += dilation

            for gamma_index in range(5, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]
                start += dilation

            for kernel_index in range(n_kernels):
                f_end = f_start + n_features_this_dilation
                n_channels_this_combo = n_channels_per_combination[comb]
                n_channels_end = n_channels_start + n_channels_this_combo
                channels_this_combo = channel_indices[n_channels_start:n_channels_end]
                _padding1 = (_padding0 + kernel_index) % 2
                index_0, index_1, index_2 = indices[kernel_index]
                C = (
                    C_alpha[channels_this_combo]
                    + C_gamma[index_0][channels_this_combo]
                    + C_gamma[index_1][channels_this_combo]
                    + C_gamma[index_2][channels_this_combo]
                )
                C = np.sum(C, axis=0)
                if _padding1 == 0:
                    for feature_count in range(n_features_this_dilation):
                        features[i, f_start + feature_count] = _PPV(
                            C, biases[f_start + feature_count]
                        ).mean()
                else:
                    for feature_count in range(n_features_this_dilation):
                        features[i, f_start + feature_count] = _PPV(
                            C[padding:-padding],
                            biases[f_start + feature_count],
                        ).mean()
                f_start = f_end
                comb += 1
                n_channels_start = n_channels_end
    return features


@njit(
    fastmath=True,
    parallel=False,
    cache=True,
)
def _fit_biases(
    X,
    n_channels_per_combination,
    channel_indices,
    dilations,
    n_features_per_dilation,
    quantiles,
    indices,
    seed,
):
    if seed is not None:
        np.random.seed(seed)
    n_cases, n_columns, n_timepoints = X.shape
    n_kernels = len(indices)
    n_dilations = len(dilations)
    n_features = n_kernels * np.sum(n_features_per_dilation)
    biases = np.zeros(n_features, dtype=np.float32)
    feature_index_start = 0
    combination_index = 0
    n_channels_start = 0
    for dilation_index in range(n_dilations):
        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2
        n_features_this_dilation = n_features_per_dilation[dilation_index]
        for kernel_index in range(n_kernels):
            feature_index_end = feature_index_start + n_features_this_dilation
            n_channels_this_combination = n_channels_per_combination[combination_index]
            n_channels_end = n_channels_start + n_channels_this_combination
            channels_this_combination = channel_indices[n_channels_start:n_channels_end]
            _X = X[np.random.randint(n_cases)][channels_this_combination]
            A = -_X  # A = alpha * X = -X
            G = _X + _X + _X  # G = gamma * X = 3X
            C_alpha = np.zeros(
                (n_channels_this_combination, n_timepoints), dtype=np.float32
            )
            C_alpha[:] = A
            C_gamma = np.zeros(
                (9, n_channels_this_combination, n_timepoints), dtype=np.float32
            )
            C_gamma[9 // 2] = G
            start = dilation
            end = n_timepoints - padding
            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]
                end += dilation
            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]
                start += dilation
            index_0, index_1, index_2 = indices[kernel_index]
            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
            C = np.sum(C, axis=0)
            biases[feature_index_start:feature_index_end] = np.quantile(
                C, quantiles[feature_index_start:feature_index_end]
            )
            feature_index_start = feature_index_end
            combination_index += 1
            n_channels_start = n_channels_end
    return biases

