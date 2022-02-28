from typing import NamedTuple, Optional, Union

import numpy as np

import pandas as pd

import sklearn.ensemble
from sklearn.experimental import enable_hist_gradient_boosting  # noqa


class GradientBoostingHyperparameters(NamedTuple):
    # https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/classification/gradient_boosting.py
    learning_rate: float = 0.1  # (1e-2, 1.0)
    min_samples_leaf: int = 20  # (1, 200)
    max_depth: Optional[int] = None
    max_leaf_nodes: int = 31  # (3, 2047)
    l2_regularization: float = 1e-10  # (1e-10, 1.0)
    n_iter_no_change: Optional[int] = 10  # (1, 20)
    early_stopping: bool = True
    validation_fraction: Optional[Union[float, int]] = 0.1  # (0.01, 0.4)
    max_iter: int = 512  # Constant
    max_bins: int = 255  # Constant
    tol: float = 1e-7  # Constant
    scoring: str = "loss"  # Constant
    loss: "str" = "auto"  # Constant


class GradientBoostingClassifier:
    def __init__(
        self,
        hyperparameters: GradientBoostingHyperparameters = GradientBoostingHyperparameters(),
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        self._hyperparameters = hyperparameters._asdict()

        if self._hyperparameters["early_stopping"]:
            self._hyperparameters["n_iter_no_change"] = 0

        self._random_state = np.random.RandomState(seed)
        self._verbose = verbose
        self._estimator = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: Optional[np.ndarray] = None
    ) -> "GradientBoostingClassifier":

        self._estimator = sklearn.ensemble.HistGradientBoostingClassifier(
            **self._hyperparameters,
            verbose=self._verbose,
            random_state=self._random_state,
        )
        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.estimator is None:
            raise ValueError(f"{self.__class__.__name__} is not fitted")

        return self.estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.estimator is None:
            raise ValueError(f"{self.__class__.__name__} is not fitted")

        return self.estimator.predict_proba(X)

    @property
    def estimator(self) -> Optional[sklearn.ensemble.HistGradientBoostingClassifier]:
        return self._estimator
