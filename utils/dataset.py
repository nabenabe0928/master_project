from typing import List, Optional, Tuple

import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import make_column_selector as selector
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUM_COL_SELECTOR = selector(dtype_include=np.number)
CAT_COL_SELECTOR = selector(dtype_exclude=np.number)


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
        if X.shape[-1] == 0:  # no feature to preprocess
            return X

        for preproc in self._preprocs:
            X = preproc.transform(X)

        return X

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        assert self._preprocs is not None
        if X.shape[-1] == 0:  # no feature to preprocess
            return X

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


def _add_nan_flag_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nan_flags_train, nan_flags_test = X_train.isna(), X_test.isna()
    any_nan_flags = nan_flags_train.any()
    any_nan_cols = []
    for col in any_nan_flags.index:
        if any_nan_flags[col]:
            any_nan_cols.append(col)

    nan_flag_cols = {col: f"{col}_nan_flag" for col in any_nan_cols}
    X_train = pd.concat([X_train, nan_flags_train[any_nan_cols].rename(columns=nan_flag_cols)], axis=1)
    X_test = pd.concat([X_test, nan_flags_test[any_nan_cols].rename(columns=nan_flag_cols)], axis=1)

    return X_train, X_test


def _add_row_wise_nan_counts(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nan_counts_train, nan_counts_test = X_train.isna().sum(axis=1), X_test.isna().sum(axis=1)
    if nan_counts_train.sum() != 0:
        X_train["nan_counts"] = nan_counts_train
        X_test["nan_counts"] = nan_counts_test

    return X_train, X_test


def _baseline_feature_preprocessing(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    cat_cols, num_cols = CAT_COL_SELECTOR(X_train), NUM_COL_SELECTOR(X_train)
    cat_cols_train, num_cols_train = X_train[cat_cols], X_train[num_cols]
    cat_cols_test, num_cols_test = X_test[cat_cols], X_test[num_cols]

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
    weights_test /= weights_test.sum()

    return X_train, X_test, y_train, y_test, weights_train, weights_test


def nan_feat_preprocessing(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    X_train, X_test = _add_row_wise_nan_counts(X_train, X_test)
    X_train, X_test = _add_nan_flag_columns(X_train, X_test)
    return baseline_preprocessing(X_train, X_test, y_train, y_test)
