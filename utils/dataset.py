from typing import List, Optional, Tuple

import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import make_column_selector as selector
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

numerical_col_selector = selector(dtype_include=np.number)
categorical_col_selector = selector(dtype_exclude=np.number)


class _CategoryShift:
    def fit(self, X: np.ndarray) -> "_CategoryShift":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X + 2

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def fetch_openml_dataset_by_id(
    data_id: int,
    frac: float = 0.75,
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # id: {2, 41138} --> {anneal, APSFailure}
    X, y = fetch_openml(data_id=data_id, return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, train_size=frac)
    return X_train, X_test, y_train, y_test


class BaseFeaturePreprocessing:
    def __init__(self):
        self._preprocs: Optional[List[BaseEstimator]] = None

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        assert self._preprocs is not None
        for preproc in self._preprocs:
            X = preproc.transform(X)

        return X

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        assert self._preprocs is not None
        for i, preproc in enumerate(self._preprocs):
            preproc = preproc.fit(X)
            X = preproc.transform(X)
            self._preprocs[i] = preproc

        return X


class CategoricalFeaturePreprocessing(BaseFeaturePreprocessing):
    """
    Omit minority coalescence
    The preprocessing mostly follows: https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/data_preprocessing/feature_type_categorical.py#L28-L35  # noqa:E501
    1. Apply ordinal encoding
    2. Impute nan by -1
    3. Add 2 to all the values
    4. Apply one-hot encoding
    """
    def __init__(self):
        self._preprocs = [
            OrdinalEncoder(categories="auto", handle_unknown='use_encoded_value', unknown_value=-1),
            SimpleImputer(strategy="constant", copy=False, fill_value=-1),
            _CategoryShift(),
            OneHotEncoder(categories="auto", sparse=False)
        ]


class NumericalFeaturePreprocessing(BaseFeaturePreprocessing):
    """
    Omit variance thresholding.
    The preprocessing mostly follows: https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/data_preprocessing/feature_type_numerical.py#L24-L30  # noqa:E501
    1. Impute by the mean value (default)
    2. Apply a scaling (e.g. standard scaler)
    """
    def __init__(self):
        self._preprocs = [
            SimpleImputer(strategy="mean", copy=False),
            StandardScaler(copy=False)
        ]


def _baseline_feature_preprocessing(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    cat_col_selector = selector(dtype_exclude=np.number)
    num_col_selector = selector(dtype_include=np.number)
    cat_cols_train = X_train[cat_col_selector(X_train)]
    num_cols_train = X_train[num_col_selector(X_train)]
    cat_cols_test = X_train[cat_col_selector(X_test)]
    num_cols_test = X_train[num_col_selector(X_test)]

    num_preproc = NumericalFeaturePreprocessing()
    num_cols_train = num_preproc.fit_transform(num_cols_train)
    num_cols_test = num_preproc.transform(num_cols_test)

    cat_preproc = CategoricalFeaturePreprocessing()
    cat_cols_train = cat_preproc.fit_transform(cat_cols_train)
    cat_cols_test = cat_preproc.transform(cat_cols_test)

    X_train = np.hstack([cat_cols_train, num_cols_train])
    X_test = np.hstack([cat_cols_test, num_cols_test])

    return X_train, X_test


def _baseline_target_preprocessing(y_train: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    enc = OrdinalEncoder(categories="auto")
    enc = enc.fit(y_train.to_numpy()[:, np.newaxis])
    y_train = enc.transform(y_train.to_numpy()[:, np.newaxis])
    y_test = enc.transform(y_test.to_numpy()[:, np.newaxis])
    return y_train.flatten(), y_test.flatten()


def baseline_preprocessing(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test = _baseline_feature_preprocessing(X_train, X_test)
    y_train, y_test = _baseline_target_preprocessing(y_train, y_test)

    cat, count = np.unique(y_train, return_counts=True)
    N, K = np.sum(count), len(cat)
    cat2weight = {c: N / (K * cnt) for c, cnt in zip(cat, count)}
    weights_train = np.array([cat2weight[c] for c in y_train])
    weights_test = np.array([cat2weight[c] for c in y_test])
    return X_train, X_test, y_train, y_test, weights_train, weights_test