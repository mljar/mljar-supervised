import logging
import copy
import numpy as np
import pandas as pd
import os

from supervised.config import storage_path
from supervised.models.learner import Learner
from supervised.tuner.registry import ModelsRegistry
from supervised.tuner.registry import BINARY_CLASSIFICATION

import multiprocessing
import lightgbm as lgb
import operator

log = logging.getLogger(__name__)


class LightgbmLearner(Learner):

    algorithm_name = "LightGBM"
    algorithm_short_name = "LightGBM"

    def __init__(self, params):
        super(LightgbmLearner, self).__init__(params)
        self.library_version = lgb.__version__
        self.model_file = self.uid + ".lgbm.model"
        self.model_file_path = os.path.join(storage_path, self.model_file)

        self.rounds = additional.get("one_step", 50)
        self.max_iters = additional.get("max_steps", 3)
        self.learner_params = {
            "boosting_type": "gbdt",
            "objective": "binary",
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

        log.debug("LightgbmLearner __init__")

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

        log.debug("LightgbmLearner save model to %s" % self.model_file_path)
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

        log.debug("LightgbmLearner load model from %s" % self.model_file_path)
        self.model = lgb.Booster(model_file=self.model_file_path)

    def importance(self, column_names, normalize=True):
        return None


LightgbmLearnerBinaryClassificationParams = {
    "objective": ["binary"],
    "metric": ["binary_logloss", "auc"],
    "num_leaves": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "learning_rate": [
        0.0025,
        0.005,
        0.0075,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
    ],
    "feature_fraction": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "bagging_fraction": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "bagging_freq": [0, 1, 2, 3, 4, 5],
}


additional = {
    "one_step": 10,
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
    LightgbmLearner,
    LightgbmLearnerBinaryClassificationParams,
    required_preprocessing,
    additional,
)
