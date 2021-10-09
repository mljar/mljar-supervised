import copy
import logging
import numpy as np
import pandas as pd
import time
import joblib
import warnings

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
    def __init__(self, params):
        super(SklearnTreesEnsembleClassifierAlgorithm, self).__init__(params)
        self.log_metric = Metric(
            {"name": self.params.get("eval_metric_name", "logloss")}
        )
        self.max_iters = (
            1  # max iters is used by model_framework, max_steps is used internally
        )
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
        max_steps = self.max_steps
        n_estimators = 0

        min_val = 10e12
        min_e = 0

        p_tr, p_vd = None, None
        result = {"iteration": [], "train": [], "validation": []}

        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore")    
            
            for i in range(max_steps):
                self.model.fit(X, np.ravel(y), sample_weight=sample_weight)
                self.model.n_estimators += self.trees_in_step

                if X_validation is None or y_validation is None:
                    continue
                estimators = self.model.estimators_

                stop = False
                for e in range(n_estimators, len(estimators)):
                    p = self.predict_function(estimators[e], X)
                    if p_tr is None:
                        p_tr = p
                    else:
                        p_tr += p

                    p = self.predict_function(estimators[e], X_validation)
                    if p_vd is None:
                        p_vd = p
                    else:
                        p_vd += p

                    tr = self.log_metric(
                        y, p_tr / float(e + 1), sample_weight=sample_weight
                    )
                    vd = self.log_metric(
                        y_validation,
                        p_vd / float(e + 1),
                        sample_weight=sample_weight_validation,
                    )

                    if vd < min_val:  # optimize direction
                        min_val = vd
                        min_e = e

                    if e - min_e >= self.early_stopping_rounds:
                        stop = True
                        break

                    result["iteration"] += [e]
                    result["train"] += [tr]
                    result["validation"] += [vd]

                # disable for now ...
                # if max_time is not None and time.time()-start_time > max_time:
                #    stop = True

                if stop:
                    self.model.estimators_ = estimators[: (min_e + 1)]
                    break
                n_estimators = len(estimators)

        if log_to_file is not None:
            df_result = pd.DataFrame(result)
            if self.log_metric.is_negative():
                df_result["train"] *= -1.0
                df_result["validation"] *= -1.0
            df_result.to_csv(log_to_file, index=False, header=False)

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
