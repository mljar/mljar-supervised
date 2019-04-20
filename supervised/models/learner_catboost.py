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
import operator

log = logging.getLogger(__name__)

from catboost import CatBoostClassifier
import catboost


class CatBoostLearner(Learner):

    algorithm_name = "CatBoost"
    algorithm_short_name = "CatBoost"

    def __init__(self, params):
        super(CatBoostLearner, self).__init__(params)
        self.library_version = catboost.__version__
        self.model_file = self.uid + ".cat.model"
        self.model_file_path = os.path.join(storage_path, self.model_file)
        self.snapshot_file_path = os.path.join(
            storage_path, "training_snapshot_" + self.model_file
        )
        self.rounds = additional.get("one_step", 50)
        self.max_iters = additional.get("max_steps", 10)
        self.learner_params = {
            "learning_rate": self.params.get("learning_rate", 0.025),
            "depth": self.params.get("depth", 6),
            "rsm": self.params.get("rsm", 1),
            "random_strength": self.params.get("random_strength", 1),
            "bagging_temperature": self.params.get("bagging_temperature", 1),
            "l2_leaf_reg": self.params.get("l2_leaf_reg", 3),
            "random_seed": self.params.get("seed", 1),
        }

        log.debug("CatBoostLearner __init__")

        self.model = CatBoostClassifier(
            iterations=0,
            learning_rate=self.learner_params.get("learning_rate"),
            depth=self.learner_params.get("depth"),
            rsm=self.learner_params.get("rsm"),
            random_strength=self.learner_params.get("random_strength"),
            bagging_temperature=self.learner_params.get("bagging_temperature"),
            l2_leaf_reg=self.learner_params.get("l2_leaf_reg"),
            loss_function="Logloss",
            verbose=False,
        )

    def update(self, update_params):
        pass
        # here should be update

    def fit(self, X, y):
        self.model._init_params["iterations"] += self.rounds
        self.model.fit(X, y, save_snapshot=True, snapshot_file=self.snapshot_file_path)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

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

        log.debug("CatBoostLearner save model to %s" % self.model_file_path)
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

        log.debug("CatBoostLearner load model from %s" % self.model_file_path)

        self.model = CatBoostClassifier()
        self.model.load_model(self.model_file_path)

    def importance(self, column_names, normalize=True):
        return None


CatBoostLearnerBinaryClassificationParams = {
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
    "depth": [2, 4, 6, 8],
    "rsm": [0.5, 0.6, 0.7, 0.8, 0.9, 1],  # random subspace method
    "random_strength": [1, 3, 5, 8, 10, 15, 20],
    "bagging_temperature": [0.5, 0.7, 0.9, 1],
    "l2_leaf_reg": [1, 3, 5, 7, 10],
}


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
    CatBoostLearner,
    CatBoostLearnerBinaryClassificationParams,
    required_preprocessing,
    additional,
)
