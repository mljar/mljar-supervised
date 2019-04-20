import logging
import copy
import numpy as np
import pandas as pd
import os

from supervised.config import storage_path
from supervised.models.learner import Learner
from supervised.tuner.registry import ModelsRegistry
from supervised.tuner.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)

import xgboost as xgb
import operator

log = logging.getLogger(__name__)


class XgbLearnerException(Exception):
    def __init__(self, message):
        super(XgbLearnerException, self).__init__(message)
        log.error(message)


class XgbLearner(Learner):
    """
    This is a wrapper over xgboost algorithm.
    """

    algorithm_name = "Extreme Gradient Boosting"
    algorithm_short_name = "Xgboost"

    def __init__(self, params):
        super(XgbLearner, self).__init__(params)
        self.library_version = xgb.__version__
        self.model_file = self.uid + ".xgb.model"
        self.model_file_path = os.path.join(storage_path, self.model_file)

        self.boosting_rounds = additional.get(
            "one_step", 50
        )  # params.get("boosting_rounds", 50)
        self.max_iters = additional.get("max_steps", 3)
        self.learner_params = {
            "booster": self.params.get("booster", "gbtree"),
            "objective": self.params.get("objective"),
            "eval_metric": self.params.get("eval_metric"),
            "eta": self.params.get("eta", 0.01),
            "max_depth": self.params.get("max_depth", 1),
            "min_child_weight": self.params.get("min_child_weight", 1),
            "subsample": self.params.get("subsample", 0.8),
            "colsample_bytree": self.params.get("colsample_bytree", 0.8),
            "silent": self.params.get("silent", 1),
            "seed": self.params.get("seed", 1),
        }
        log.debug("XgbLearner __init__")

    def update(self, update_params):
        # Dont need to update boosting rounds, it is adding rounds incrementally
        pass

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y, missing=np.NaN)
        self.model = xgb.train(
            self.learner_params, dtrain, self.boosting_rounds, xgb_model=self.model
        )

    def predict(self, X):
        if self.model is None:
            raise XgbLearnerException("Xgboost model is None")
        dtrain = xgb.DMatrix(X, missing=np.NaN)
        return self.model.predict(dtrain)

    def copy(self):
        return copy.deepcopy(self)

    def save(self):
        self.model.save_model(self.model_file_path)

        json_desc = {
            "library_version": self.library_version,
            "algorithm_name": self.algorithm_name,
            "algorithm_short_name": self.algorithm_short_name,
            "uid": self.uid,
            "model_file": self.model_file,
            "model_file_path": self.model_file_path,
            "params": self.params,
        }

        log.debug("XgbLearner save model to %s" % self.model_file_path)
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

        log.debug("XgbLearner load model from %s" % self.model_file_path)
        self.model = xgb.Booster()  # init model
        self.model.load_model(self.model_file_path)

    def importance(self, column_names, normalize=True):
        return None

    def get_params_key(self):
        params_key = "key"
        for p, v in self.params.items():
            params_key += "_{}_{}".format(p, str(v))
        return params_key


# For binary classification target should be 0, 1. There should be no NaNs in target.
XgbLearnerBinaryClassificationParams = {
    "booster": ["gbtree", "gblinear"],
    "objective": ["binary:logistic"],
    "eval_metric": ["auc", "logloss"],
    "eta": [0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1],
    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "min_child_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "subsample": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

XgbLearnerRegressionParams = dict(XgbLearnerBinaryClassificationParams)
XgbLearnerRegressionParams["booster"] = ["gbtree"]
XgbLearnerRegressionParams["objective"] = ["reg:linear", "reg:log"]
XgbLearnerRegressionParams["eval_metric"] = ["rmse", "mae"]

XgbLearnerMulticlassClassificationParams = dict(XgbLearnerBinaryClassificationParams)

additional = {
    "one_step": 50,
    "train_cant_improve_limit": 5,
    "max_steps": 500,
    "max_rows_limit": None,
    "max_cols_limit": None,
}
required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "target_preprocessing",
]

ModelsRegistry.add(
    BINARY_CLASSIFICATION,
    XgbLearner,
    XgbLearnerBinaryClassificationParams,
    required_preprocessing,
    additional,
)
ModelsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    XgbLearner,
    XgbLearnerMulticlassClassificationParams,
    required_preprocessing,
    additional,
)
ModelsRegistry.add(
    REGRESSION,
    XgbLearner,
    XgbLearnerRegressionParams,
    required_preprocessing,
    additional,
)
