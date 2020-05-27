import copy
import logging
import numpy as np
import warnings
import joblib

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

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

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)

    def copy(self):
        return copy.deepcopy(self)

    def save(self, model_file_path):
        logger.debug("SklearnAlgorithm save to {0}".format(model_file_path))
        joblib.dump(self.model, model_file_path, compress=True)

    def load(self, model_file_path):
        logger.debug("SklearnAlgorithm loading model from {0}".format(model_file_path))
        self.model = joblib.load(model_file_path)

    def get_params(self):
        return {
            "library_version": self.library_version,
            "algorithm_name": self.algorithm_name,
            "algorithm_short_name": self.algorithm_short_name,
            "uid": self.uid,
            "params": self.params,
        }

    def set_params(self, json_desc):
        self.library_version = json_desc.get("library_version", self.library_version)
        self.algorithm_name = json_desc.get("algorithm_name", self.algorithm_name)
        self.algorithm_short_name = json_desc.get(
            "algorithm_short_name", self.algorithm_short_name
        )
        self.uid = json_desc.get("uid", self.uid)
        self.params = json_desc.get("params", self.params)

    def predict(self, X):
        if self.params["ml_task"] == BINARY_CLASSIFICATION:
            return self.model.predict_proba(X)[:, 1]
        elif self.params["ml_task"] == MULTICLASS_CLASSIFICATION:
            return self.model.predict_proba(X)
        return self.model.predict(X)


class SklearnTreesClassifierAlgorithm(SklearnAlgorithm):
    def __init__(self, params):
        super(SklearnTreesClassifierAlgorithm, self).__init__(params)

    def fit(self, X, y):
        self.model.fit(X, np.ravel(y))
        if hasattr(self.model, "n_estimators"):
            self.model.n_estimators += self.trees_in_step


# TODO refactor
class SklearnTreesRegressorAlgorithm(SklearnAlgorithm):
    def __init__(self, params):
        super(SklearnTreesRegressorAlgorithm, self).__init__(params)

    def fit(self, X, y):
        self.model.fit(X, np.ravel(y))
        if hasattr(self.model, "n_estimators"):
            self.model.n_estimators += self.trees_in_step
