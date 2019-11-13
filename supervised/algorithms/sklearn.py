import numpy as np

from supervised.algorithms.algorithm import BaseAlgorithm
from sklearn.externals import joblib
import copy

import logging

logger = logging.getLogger(__name__)
from supervised.config import LOG_LEVEL

logger.setLevel(LOG_LEVEL)


class SklearnAlgorithm(BaseAlgorithm):
    def __init__(self, params):
        super(SklearnAlgorithm, self).__init__(params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def copy(self):
        return copy.deepcopy(self)

    def save(self):
        logger.debug("SklearnAlgorithm save to {0}".format(self.model_file_path))
        joblib.dump(self.model, self.model_file_path, compress=True)

        json_desc = {
            "library_version": self.library_version,
            "algorithm_name": self.algorithm_name,
            "algorithm_short_name": self.algorithm_short_name,
            "uid": self.uid,
            "model_file": self.model_file,
            "model_file_path": self.model_file_path,
            "params": self.params,
        }

        return json_desc

    def load(self, json_desc):

        self.library_version = json_desc.get("library_version", self.library_version)
        self.algorithm_name = json_desc.get("algorithm_name", self.algorithm_name)
        self.algorithm_short_name = json_desc.get(
            "algorithm_short_name", self.algorithm_short_name
        )
        self.uid = json_desc.get("uid", self.uid)
        self.model_file = json_desc.get("model_file", self.model_file)
        self.model_file_path = json_desc.get("model_file_path", self.model_file_path)
        self.params = json_desc.get("params", self.params)

        logger.debug(
            "SklearnAlgorithm loading model from {0}".format(self.model_file_path)
        )
        self.model = joblib.load(self.model_file_path)


class SklearnTreesClassifierAlgorithm(SklearnAlgorithm):
    def __init__(self, params):
        super(SklearnTreesClassifierAlgorithm, self).__init__(params)

    def fit(self, X, y):
        logger.debug("SklearnTreesClassifierAlgorithm.fit")
        self.model.fit(X, np.ravel(y))
        self.model.n_estimators += self.trees_in_step

    def predict(self, X):
        logger.debug("SklearnTreesClassifierAlgorithm.predict")
        if "num_class" in self.params:
            return self.model.predict_proba(X)
        return self.model.predict_proba(X)[:, 1]
