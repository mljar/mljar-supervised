import logging
import copy
import numpy as np
import pandas as pd
import os
import contextlib
import multiprocessing
import lightgbm as lgb

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


class LightgbmAlgorithm(BaseAlgorithm):

    algorithm_name = "LightGBM"
    algorithm_short_name = "LightGBM"

    def __init__(self, params):
        super(LightgbmAlgorithm, self).__init__(params)
        self.library_version = lgb.__version__

        self.rounds = additional.get("trees_in_step", 50)
        self.max_iters = additional.get("max_steps", 500)
        self.learner_params = {
            "boosting_type": "gbdt",
            "objective": self.params.get("objective", "binary"),
            "metric": self.params.get("metric", "binary_logloss"),
            "num_threads": multiprocessing.cpu_count(),
            "num_leaves": self.params.get("num_leaves", 16),
            "learning_rate": self.params.get("learning_rate", 0.01),
            "feature_fraction": self.params.get("feature_fraction", 0.7),
            "bagging_fraction": self.params.get("bagging_fraction", 0.7),
            "bagging_freq": self.params.get("bagging_freq", 1),
            "verbose": -1,
            "seed": self.params.get("seed", 1),
        }
        if "num_class" in self.params:  # multiclass classification
            self.learner_params["num_class"] = self.params.get("num_class")

        logger.debug("LightgbmLearner __init__")

    def file_extension(self):
        return "lightgbm"

    def update(self, update_params):
        pass

    def fit(self, X, y):
        lgb_train = lgb.Dataset(X, y)
        self.model = lgb.train(
            self.learner_params,
            lgb_train,
            num_boost_round=self.rounds,
            init_model=self.model,
        )

    def predict(self, X):
        return self.model.predict(X)

    def copy(self):
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            return copy.deepcopy(self)

    def save(self, model_file_path):
        self.model.save_model(model_file_path)
        logger.debug("LightgbmAlgorithm save model to %s" % model_file_path)

    def load(self, model_file_path):
        logger.debug("LightgbmAlgorithm load model from %s" % model_file_path)
        self.model = lgb.Booster(model_file=model_file_path)

    def get_params(self):
        json_desc = {
            "library_version": self.library_version,
            "algorithm_name": self.algorithm_name,
            "algorithm_short_name": self.algorithm_short_name,
            "uid": self.uid,
            "params": self.params,
        }
        return json_desc

    def set_params(self, json_desc):
        self.library_version = json_desc.get("library_version", self.library_version)
        self.algorithm_name = json_desc.get("algorithm_name", self.algorithm_name)
        self.algorithm_short_name = json_desc.get(
            "algorithm_short_name", self.algorithm_short_name
        )
        self.uid = json_desc.get("uid", self.uid)
        self.params = json_desc.get("params", self.params)


lgbm_bin_params = {
    "objective": ["binary"],
    "metric": ["binary_logloss", "auc"],
    "num_leaves": [32, 64, 128, 256, 512],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1],
    "feature_fraction": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "bagging_fraction": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "bagging_freq": [0, 1, 2, 3, 4, 5],
}


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
    "target_scale",
]


lgbm_multi_params = copy.deepcopy(lgbm_bin_params)
lgbm_multi_params["objective"] = ["multiclass"]
lgbm_multi_params["metric"] = ["multi_logloss", "multi_error"]


lgbr_params = copy.deepcopy(lgbm_bin_params)
lgbr_params["objective"] = ["regression"]
lgbr_params["metric"] = ["l1", "l2"]

AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    LightgbmAlgorithm,
    lgbm_bin_params,
    required_preprocessing,
    additional,
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    LightgbmAlgorithm,
    lgbm_multi_params,
    required_preprocessing,
    additional,
)

AlgorithmsRegistry.add(
    REGRESSION, LightgbmAlgorithm, lgbr_params, required_preprocessing, additional
)
