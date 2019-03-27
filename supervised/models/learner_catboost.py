import logging
import copy
import numpy as np
import pandas as pd

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
        self.model_file_path = "/tmp/" + self.model_file

        self.rounds = additional.get(
            "one_step", 10
        )
        self.max_iters = 1#additional.get("max_steps", 3)
        self.learner_params = {
            #"bagging_fraction": self.params.get("bagging_fraction", 0.7),
        }

        log.debug("CatBoostLearner __init__")

        self.model = CatBoostClassifier(iterations=150,
                                   depth=4,
                                   learning_rate=0.1,
                                   loss_function='Logloss',
                                   verbose=True)


    def update(self, update_params):
        print("CatBoost update", update_params)
        #self.rounds = update_params["iters"]

    def fit(self, data):
        log.debug("CatBoostLearner.fit")
        X = data.get("X")
        y = data.get("y")

        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:,1]

    def copy(self):
        return None

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
    ]
}


additional = {
    "one_step": 50,
    "train_cant_improve_limit": 5,
    "max_steps": 100,
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
