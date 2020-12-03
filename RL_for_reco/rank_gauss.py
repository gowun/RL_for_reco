# Reference:
# 1. rankGauss by Micheal Jahrer
# https://github.com/michaeljahrer/rankGauss/blob/master/rankGaussMain.cpp
# 2. a beginning of scikit-learn compatible implementation of GaussRank
# https://github.com/zygmuntz/gaussrank/blob/master/gaussrank.py
# Extended by 오병화, 20180903
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn.utils import check_array, check_random_state
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.interpolate import interp1d
from collections import OrderedDict
from itertools import chain
import numpy as np
import pickle

try:
    from joblib import Parallel, delayed
except ImportError:
    from sklearn.externals.joblib import Parallel, delayed


class RankGaussScaler(BaseEstimator, TransformerMixin):
    def __init__(self, nan_to_val=None, extrapolate=False, num_storing=None, random_state=None, interp_params=None, n_jobs=None):
        nan_to_val = nan_to_val or True
        self.nan_to_val = 0.0 if isinstance(nan_to_val, bool) else nan_to_val
        self.force_all_finite = False
        if isinstance(nan_to_val, bool) and not nan_to_val:
            self.force_all_finite = True
        self.extrapolate = extrapolate
        num_storing = num_storing or np.iinfo(int).max
        self.num_storing = 2 if num_storing < 2 else num_storing
        self.random_state = check_random_state(random_state)
        self.interp_params = interp_params or dict(kind='linear')
        self.n_jobs = n_jobs

    '''
    change for working on pyspark
    '''
    def fit(self, X, y=None, cols=None, add_timeseries=False, save_path=None):
        X = self._check_array(X)
        X = self._to_2d_if_1d(X)
        self.codebooks_ = Parallel(n_jobs=self.n_jobs)(delayed(self._make_codebook)(*x) for x in enumerate(X.T))
        if add_timeseries and cols is not None:
            ts_keywords = sorted(set(map(lambda x: '_'.join(x.split('_')[:-1]) if x[-3:-1] == '_m' else '', cols)))[1:]
            ts_cols = []
            for key in ts_keywords:
                ts_cols.append(list(filter(lambda x: x.startswith(key), cols)))
        if save_path is not None and cols is not None:
            model_info = {'type': 'rank_gauss', 'model': self, 'input_cols': cols}
            if add_timeseries:
                model_info['output_cols'] = cols + list(map(lambda x: x + '_ts', chain(*ts_cols)))
                model_info['timeseries_cols'] = ts_cols
            else:
                model_info['output_cols'] = cols
            pickle.dump(model_info, open(save_path, 'wb'), 4)
        return self

    def transform(self, X):
        transformed = self._transform(X, self._transform_column)
        if not self.force_all_finite:
            transformed[~np.isfinite(transformed)] = self.nan_to_val
        return transformed

    def inverse_transform(self, X):
        return self._transform(X, self._inv_transform_column)

    def _transform(self, X, func_transform):
        X = self._check_before_transform(X)
        return_as_1d = True if len(X.shape) == 1 else False
        X = self._to_2d_if_1d(X)
        transformed = np.array(Parallel(n_jobs=self.n_jobs)(delayed(func_transform)(*x, **self.interp_params) for x in enumerate(X.T))).T
        return self._to_1d_if_single(transformed) if return_as_1d else transformed

    def _check_array(self, X):
        return check_array(X, dtype=FLOAT_DTYPES, ensure_2d=False, force_all_finite=self.force_all_finite)

    def _check_num_cols(self, X):
        num_features = 1 if len(X.shape) == 1 else X.shape[1]
        if num_features != len(self.codebooks_):
            raise ValueError('bad input shape {0}'.format(X.shape))

    def _check_before_transform(self, X):
        check_is_fitted(self, 'codebooks_')
        X = self._check_array(X)
        self._check_num_cols(X)
        return X

    def _make_codebook(self, col_index, x):
        codebook = build_rankguass_trafo(x)
        num_codes = len(codebook[0])
        if num_codes == 0:
            raise ValueError('column %d contains only null values' % col_index)
        elif num_codes > self.num_storing:
            chosen = self.random_state.choice(num_codes - 2, self.num_storing - 2, replace=False)
            return codebook[0][chosen], codebook[1][chosen]
        else:
            return codebook

    def _transform_column(self, index, x, **interp1d_params):
        return self._transform_with_interp(x, *self.codebooks_[index], **interp1d_params)

    def _inv_transform_column(self, index, x, **interp1d_params):
        return self._transform_with_interp(x, *reversed(self.codebooks_[index]), **interp1d_params)

    def _transform_with_interp(self, x, train_x, train_y, **interp1d_params):
        if len(train_x) == 1:
            return np.ones(x.shape) * train_y[0]
        f = interp1d(train_x, train_y, fill_value='extrapolate', **interp1d_params)
        return f(x) if self.extrapolate else f(np.clip(x, *minmax(train_x)))

    @staticmethod
    def _to_2d_if_1d(a):
        return a.reshape(-1, 1) if len(a.shape) == 1 else a

    @staticmethod
    def _to_1d_if_single(a):
        return a.ravel() if a.shape[1] == 1 else a


# function for simultaneous max() and min() (using numba)
# https://stackoverflow.com/a/33919126
def minmax(x):
    maximum = x[0]
    minimum = x[0]
    for i in x[1:]:
        if i > maximum:
            maximum = i
        elif i < minimum:
            minimum = i
    return minimum, maximum


# converted from [ref 1]
def norm_cdf_inv(p):
    sign = 1.0
    if p < 0.5:
        sign = -1.0
    else:
        p = 1.0 - p
    t = np.sqrt(-2.0 * np.log(p))
    return sign * (t - ((0.010328 * t + 0.802853) * t + 2.515517) / (((0.001308 * t + 0.189269) * t + 1.432788) * t + 1.0))


# converted from [ref 1]
def build_rankguass_trafo(x):
    finite_indices = np.isfinite(x)
    if np.sum(finite_indices) == 0:
        return np.array([]), np.array([])
    x_finite = x[np.isfinite(x)]
    hist = dict()
    for val in x_finite:
        hist[val] = hist.get(val, 0) + 1
    len_hist = len(hist)
    list_keys = list(hist.keys())
    if len_hist == 1:
        return np.array(list_keys), np.array([0.0])
    elif len_hist == 2:
        return np.array(list_keys), np.array([0.0, 1.0])
    else:
        hist = OrderedDict(sorted(hist.items()))
        n = float(x_finite.shape[0])
        cnt = 0.0
        mean = 0.0
        trafo_keys = list()
        trafo_values = list()
        for key, val in hist.items():
            rank_v = norm_cdf_inv(cnt / n * 0.998 + 1e-3) * 0.7
            trafo_keys.append(key)
            trafo_values.append(rank_v)
            mean += val * rank_v
            cnt += val
        mean /= n
        trafo_values = np.array(trafo_values)
        trafo_values -= mean
        return np.array(trafo_keys), trafo_values