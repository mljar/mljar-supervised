import copy
import logging
import warnings

import joblib
import numpy as np
import pandas as pd

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class SklearnAlgorithm(BaseAlgorithm):
    def __init__(self, params):
        super(SklearnAlgorithm, self).__init__(params)

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        X_validation=None,
        y_validation=None,
        sample_weight_validation=None,
        log_to_file=None,
        max_time=None,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore")
            self.model.fit(X, y, sample_weight=sample_weight)
            if self.params["ml_task"] != REGRESSION:
                self.classes_ = np.unique(y)

    def copy(self):
        return copy.deepcopy(self)

    def save(self, model_file_path):
        logger.debug("SklearnAlgorithm save to {0}".format(model_file_path))
        joblib.dump(self.model, model_file_path, compress=True)
        self.model_file_path = model_file_path

    def load(self, model_file_path):
        logger.debug("SklearnAlgorithm loading model from {0}".format(model_file_path))
        self.model = joblib.load(model_file_path)
        self.model_file_path = model_file_path

    def is_fitted(self):
        return (
            hasattr(self.model, "n_features_in_")
            and self.model.n_features_in_ is not None
            and self.model.n_features_in_ > 0
        )

    def predict(self, X):
        self.reload()
        if self.params["ml_task"] == BINARY_CLASSIFICATION:
            return self.model.predict_proba(X)[:, 1]
        elif self.params["ml_task"] == MULTICLASS_CLASSIFICATION:
            return self.model.predict_proba(X)
        return self.model.predict(X)


from supervised.utils.metric import Metric


def predict_proba_function_binary(estimator, X):
    return estimator.predict_proba(X)[:, 1]


def predict_proba_function_multiclass(estimator, X):
    return estimator.predict_proba(X)


class SklearnTreesEnsembleClassifierAlgorithm(SklearnAlgorithm):
    @staticmethod
    def tree_count(rows_count):
        """Use one fixed count for the complete training/validation split."""
        return 100 * min(3, max(1, rows_count // 1000))

    def __init__(self, params):
        super(SklearnTreesEnsembleClassifierAlgorithm, self).__init__(params)
        self.log_metric = Metric(
            {"name": self.params.get("eval_metric_name", "logloss")}
        )
        self.max_iters = 1  # Fit the complete fixed-size forest once per fold.
        if params.get("ml_task") == BINARY_CLASSIFICATION:
            self.predict_function = predict_proba_function_binary
        else:
            self.predict_function = predict_proba_function_multiclass

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        X_validation=None,
        y_validation=None,
        sample_weight_validation=None,
        log_to_file=None,
        max_time=None,
    ):
        rows_count = len(X) + (len(X_validation) if X_validation is not None else 0)
        n_estimators = self.tree_count(rows_count)
        # Validation labels do not select the forest size. Disable warm start
        # so fitting the same learner again cannot retain trees from old data.
        self.model.set_params(n_estimators=n_estimators, warm_start=False)
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore")
            self.model.fit(X, np.ravel(y), sample_weight=sample_weight)
        self.params["n_estimators"] = n_estimators

        if log_to_file is not None:
            train_score = self.log_metric(
                y, self.predict_function(self.model, X), sample_weight=sample_weight
            )
            validation_score = np.nan
            if X_validation is not None and y_validation is not None:
                validation_score = self.log_metric(
                    y_validation,
                    self.predict_function(self.model, X_validation),
                    sample_weight=sample_weight_validation,
                )
            # Log the final forest only; there is no selected validation peak.
            df_result = pd.DataFrame({
                "iteration": [n_estimators - 1],
                "train": [train_score],
                "validation": [validation_score],
            })
            if self.log_metric.is_negative():
                df_result["train"] *= -1.0
                df_result["validation"] *= -1.0
            df_result.to_csv(log_to_file, index=False, header=False)

        if self.params.get("ml_task") != REGRESSION:
            self.classes_ = self.model.classes_

    def get_metric_name(self):
        return self.params.get("eval_metric_name", "logloss")


def predict_function(estimator, X):
    return estimator.predict(X)


class SklearnTreesEnsembleRegressorAlgorithm(SklearnTreesEnsembleClassifierAlgorithm):
    def __init__(self, params):
        super(SklearnTreesEnsembleRegressorAlgorithm, self).__init__(params)
        self.log_metric = Metric({"name": self.params.get("eval_metric_name", "rmse")})
        self.predict_function = predict_function

    def get_metric_name(self):
        return self.params.get("eval_metric_name", "rmse")
