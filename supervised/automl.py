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
        self._time_limit = time_limit  # time limit in seconds
        self._learners = []
        self._validation = {"validation_type": "kfold", "k_folds": 5, "shuffle": True}

    def _get_model_params(self, X, y):
        available_models = list(ModelsRegistry.registry[BINARY_CLASSIFICATION].keys())
        model_type = np.random.permutation(available_models)[0]
        model_info = ModelsRegistry.registry[BINARY_CLASSIFICATION][model_type]
        model_params = RandomParameters.get(model_info["params"])
        required_preprocessing = model_info["required_preprocessing"]
        model_additional = model_info["additional"]
        preprocessing_params = PreprocessingTuner.get(
            required_preprocessing, {"train": {"X": X, "y": y}}, BINARY_CLASSIFICATION
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

        for i in range(5):
            print("-" * 21)
            params = self._get_model_params(X, y)
            early_stop = EarlyStopping({"metric": {"name": "logloss"}})
            il = IterativeLearner(params, callbacks=[early_stop])
            il.train({"train": {"X": X, "y": y}})
            self._learners += [il]
            print("early-----------------------")
            print(early_stop.best_loss)

        for il in self._learners:
            print("Learner")
            print(il.callbacks.callbacks[0].best_loss)
            print(il.callbacks.callbacks[0].final_loss)

    def predict(self, X):
        y_predicted_mean = None
        for il in self._learners:
            y_predicted = il.predict(X)
            y_predicted_mean = (
                y_predicted
                if y_predicted_mean is None
                else y_predicted_mean + y_predicted
            )
        # Make a mean
        y_predicted_mean /= float(len(self._learners))
        return y_predicted_mean

    def to_json(self):
        save_details = []
        for il in self._learners:
            save_details += [il.save()]
        return save_details

    def from_json(self, json_data):
        self._learners = []
        for save_detail in json_data:
            il = IterativeLearner()
            il.load(save_detail)
            self._learners += [il]
