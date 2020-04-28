import logging
import copy
import numpy as np
import pandas as pd
import os

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import (BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, REGRESSION)
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

from catboost import CatBoostClassifier, CatBoostRegressor
import catboost

class CatBoostAlgorithm(BaseAlgorithm):

    algorithm_name = "CatBoost"
    algorithm_short_name = "CatBoost"

    def __init__(self, params):
        super(CatBoostAlgorithm, self).__init__(params)
        self.library_version = catboost.__version__
        self.snapshot_file_path = "training_snapshot"

        self.rounds = additional.get("trees_in_step", 10)
        self.max_iters = additional.get("max_steps", 500)

        Algo = CatBoostClassifier
        loss_function = "Logloss"
        if self.params["ml_task"] == MULTICLASS_CLASSIFICATION:
            loss_function = "MultiClass"
        elif self.params["ml_task"] == REGRESSION:
            loss_function = self.params.get("loss_function", "RMSE")
            Algo = CatBoostRegressor
        
        self.learner_params = {
            "learning_rate": self.params.get("learning_rate", 0.1),
            "depth": self.params.get("depth", 6),
            "rsm": self.params.get("rsm", 1),
            "l2_leaf_reg": self.params.get("l2_leaf_reg", 3),
            "random_seed": self.params.get("seed", 1),
            "loss_function": loss_function
        }

        self.model = Algo(
            iterations=self.rounds,
            learning_rate=self.learner_params["learning_rate"],
            depth=self.learner_params["depth"],
            rsm=self.learner_params["rsm"],
            l2_leaf_reg=self.learner_params["l2_leaf_reg"],
            loss_function=self.learner_params["loss_function"],
            verbose=False,
            allow_writing_files=False
        )
        self.cat_features = None

        logger.debug("CatBoostAlgorithm.__init__")

    def fit(self, X, y):
        if self.cat_features is None:
            self.cat_features = []
            for i in range(X.shape[1]):
                if PreprocessingUtils.is_categorical(X.iloc[:, i]):
                    self.cat_features += [i]

        self.model.fit(
            X,
            y,
            cat_features=self.cat_features,
            init_model=None if self.model.tree_count_ is None else self.model,
        )

    def predict(self, X):
        if self.params["ml_task"] == BINARY_CLASSIFICATION:
            return self.model.predict_proba(X)[:, 1]
        elif self.params["ml_task"] == MULTICLASS_CLASSIFICATION:
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def copy(self):
        return copy.deepcopy(self)

    def save(self, model_file_path):
        self.model.save_model(model_file_path)
        logger.debug("CatBoostAlgorithm save model to %s" % model_file_path)

    def load(self, model_file_path):
        logger.debug("CatBoostLearner load model from %s" % model_file_path)
        self.model = CatBoostClassifier()
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
        return "catboost"


classification_params = {
    "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2],
    "depth": [4, 5, 6, 7, 8],
    "rsm": [0.5, 0.6, 0.7, 0.8, 0.9, 1],  # random subspace method
    "l2_leaf_reg": [1, 3, 5, 7, 10],
}

additional = {
    "trees_in_step": 10,
    "train_cant_improve_limit": 5,
    "min_steps": 5,
    "max_steps": 500,
    "max_rows_limit": None,
    "max_cols_limit": None,
}
required_preprocessing = ["missing_values_inputation", "target_as_integer"]


AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    CatBoostAlgorithm,
    classification_params,
    required_preprocessing,
    additional,
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    CatBoostAlgorithm,
    classification_params,
    required_preprocessing,
    additional,
)

regression_params = copy.deepcopy(classification_params)
regression_params["loss_function"] = ["MAE", "RMSE"]

AlgorithmsRegistry.add(
    REGRESSION,
    CatBoostAlgorithm,
    regression_params,
    required_preprocessing,
    additional,
)