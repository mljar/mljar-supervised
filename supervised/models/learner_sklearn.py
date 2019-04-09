import numpy as np
import logging
from supervised.models.learner import Learner
from sklearn.externals import joblib
import copy

logger = logging.getLogger(__name__)


class SklearnLearner(Learner):
    def __init__(self, params):
        super(SklearnLearner, self).__init__(params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def copy(self):
        return copy.deepcopy(self)

    def save(self):
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

        logger.debug("SklearnLearner save to {0}".format(self.model_file_path))
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

        self.model = joblib.load(self.model_file_path)

        logger.debug(
            "SklearnLearner loading model from {0}".format(self.model_file_path)
        )


class SklearnTreesClassifierLearner(SklearnLearner):
    def __init__(self, params):
        super(SklearnTreesClassifierLearner, self).__init__(params)

    def fit(self, X, y):
        self.model.fit(X, np.ravel(y))
        self.model.n_estimators += self.trees_in_step

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
