import json
import numpy as np
import pandas as pd

from supervised.models.learner_xgboost import XgbLearner
from supervised.iterative_learner_framework import IterativeLearner
from supervised.callbacks.early_stopping import EarlyStopping
from supervised.callbacks.metric_logger import MetricLogger
from supervised.metric import Metric
from supervised.tuner.random_parameters import RandomParameters
from supervised.tuner.registry import ModelsRegistry
from supervised.tuner.registry import BINARY_CLASSIFICATION
from supervised.tuner.preprocessing_tuner import PreprocessingTuner


class AutoML:

    def __init__(self, time_limit=60):
        self._time_limit = time_limit # time limit in seconds

        self._learners = []
        self._learners_params = []
        self._validation = {
            "validation_type": "kfold",
            "k_folds": 5,
            "shuffle": True,
        }

    def _get_model_params(self, X, y):
        available_models = list(ModelsRegistry.registry[BINARY_CLASSIFICATION].keys())
        model_type = np.random.permutation(available_models)[0]
        model_info = ModelsRegistry.registry[BINARY_CLASSIFICATION][model_type]
        model_params = RandomParameters.get(model_info["params"])
        required_preprocessing = model_info["required_preprocessing"]
        model_additional = model_info["additional"]
        preprocessing_params = PreprocessingTuner.get(
            required_preprocessing, {"train": {"X": X, "y":y}}, BINARY_CLASSIFICATION
        )

        return {
            "additional": model_additional,
            "preprocessing": preprocessing_params,
            "validation": self._validation,
            "learner": {
                "model_type": model_info["class"].algorithm_short_name,
                **model_params,
            },
        }

    def fit(self, X, y):
        # get preprocessing and model params
        for i in range(5):
            print("-"*21)
            params = self._get_model_params(X, y)
            self._learners_params += [params]

            early_stop = EarlyStopping({"metric": {"name": "logloss"}})
            metric_logger = MetricLogger({"metric_names": ["logloss", "auc"]})
            il = IterativeLearner(params, callbacks=[early_stop, metric_logger])
            il.train({"train":{"X":X, "y":y}})

            y_predicted = il.predict(self.data["train"]["X"])
            metric = Metric({"name": "logloss"})
            loss = metric(self.data["train"]["y"], y_predicted)

            self._learners += [il]


    def predict(self, X):
        pass

    def save(self, storage_path):
        pass

    def load(self, storage_path):
        pass

'''
        early_stop = EarlyStopping({"metric": {"name": "logloss"}})
        metric_logger = MetricLogger({"metric_names": ["logloss", "auc"]})
        il = IterativeLearner(self.train_params, callbacks=[early_stop, metric_logger])
        il.train(self.data)

        y_predicted = il.predict(self.data["train"]["X"])

        metric = Metric({"name": "logloss"})
        loss = metric(self.data["train"]["y"], y_predicted)
        self.assertTrue(loss < 0.6)
'''
