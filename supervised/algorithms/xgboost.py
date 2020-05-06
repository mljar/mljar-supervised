import logging
import copy
import numpy as np
import pandas as pd
import os
import xgboost as xgb

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

import tempfile


class XgbAlgorithmException(Exception):
    def __init__(self, message):
        super(XgbAlgorithmException, self).__init__(message)
        logger.error(message)


class XgbAlgorithm(BaseAlgorithm):
    """
    This is a wrapper over xgboost algorithm.
    """

    algorithm_name = "Extreme Gradient Boosting"
    algorithm_short_name = "Xgboost"

    def __init__(self, params):
        super(XgbAlgorithm, self).__init__(params)
        self.library_version = xgb.__version__

        self.boosting_rounds = additional.get("trees_in_step", 50)
        self.max_iters = additional.get("max_steps", 500)
        self.learner_params = {
            "tree_method": "hist",
            "booster": "gbtree",
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

        if "num_class" in self.params:  # multiclass classification
            self.learner_params["num_class"] = self.params.get("num_class")

        logger.debug("XgbLearner __init__")

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y, missing=np.NaN)
        print(self.learner_params)
        print(self.learner_params["seed"])
        print(type(self.learner_params["seed"]))
        self.model = xgb.train(
            self.learner_params, dtrain, self.boosting_rounds, xgb_model=self.model
        )

        # fix high memory consumption in xgboost,
        # waiting for release with fix
        # https://github.com/dmlc/xgboost/issues/5474
        with tempfile.NamedTemporaryFile() as tmp:
            self.model.save_model(tmp.name)
            del self.model
            self.model = xgb.Booster()
            self.model.load_model(tmp.name)

    def predict(self, X):
        if self.model is None:
            raise XgbAlgorithmException("Xgboost model is None")
        dtrain = xgb.DMatrix(X, missing=np.NaN)
        a = self.model.predict(dtrain)
        return a

    def copy(self):
        return copy.deepcopy(self)

    def save(self, model_file_path):
        self.model.save_model(model_file_path)
        logger.debug("XgbAlgorithm save model to %s" % model_file_path)

    def load(self, model_file_path):
        logger.debug("XgbLearner load model from %s" % model_file_path)
        self.model = xgb.Booster()  # init model
        self.model.load_model(model_file_path)

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

    def file_extension(self):
        return "xgboost"


# For binary classification target should be 0, 1. There should be no NaNs in target.
xgb_bin_class_params = {
    "objective": ["binary:logistic"],
    "eval_metric": ["auc", "logloss"],
    "eta": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15],
    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "min_child_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "subsample": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

xgb_regression_params = dict(xgb_bin_class_params)
xgb_regression_params["objective"] = ["reg:squarederror"]
xgb_regression_params["eval_metric"] = ["rmse", "rmsle", "mae"]
xgb_regression_params["max_depth"] = [1, 2, 3, 4]


xgb_multi_class_params = dict(xgb_bin_class_params)
xgb_multi_class_params["objective"] = ["multi:softprob"]
xgb_multi_class_params["eval_metric"] = ["mlogloss"]


additional = {
    "trees_in_step": 10,
    "train_cant_improve_limit": 5,
    "min_steps": 5,
    "max_steps": 500,
    "max_rows_limit": None,
    "max_cols_limit": None,
}
required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "target_as_integer",
]

AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    XgbAlgorithm,
    xgb_bin_class_params,
    required_preprocessing,
    additional,
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    XgbAlgorithm,
    xgb_multi_class_params,
    required_preprocessing,
    additional,
)

regression_required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "target_scale",
]


AlgorithmsRegistry.add(
    REGRESSION,
    XgbAlgorithm,
    xgb_regression_params,
    regression_required_preprocessing,
    additional,
)
