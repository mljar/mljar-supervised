import json
import copy
import numpy as np
import pandas as pd

from supervised.models.learner_xgboost import XgbLearner
from supervised.iterative_learner_framework import IterativeLearner
from supervised.callbacks.early_stopping import EarlyStopping
from supervised.callbacks.metric_logger import MetricLogger
from supervised.callbacks.time_constraint import TimeConstraint
from supervised.metric import Metric
from supervised.tuner.random_parameters import RandomParameters
from supervised.tuner.registry import ModelsRegistry
from supervised.tuner.registry import BINARY_CLASSIFICATION
from supervised.tuner.preprocessing_tuner import PreprocessingTuner
from supervised.tuner.hill_climbing import HillClimbing
from supervised.models.ensemble import Ensemble


class AutoML:
    # "CatBoost", "Xgboost", "RF", "LightGBM"
    def __init__(self, time_limit=60, algorithms=["NN"]):
        self._time_limit = time_limit  # time limit in seconds
        self._models = []
        self._models_params_keys = []
        self._best_model = None
        self._validation = {"validation_type": "kfold", "k_folds": 5, "shuffle": True}

        self._start_random_models = 1
        self._hill_climbing_steps = 0
        self._top_models_to_improve = 0
        self._algorithms = algorithms
        if len(self._algorithms) == 0:
            self._algorithms = list(
                ModelsRegistry.registry[BINARY_CLASSIFICATION].keys()
            )

    def _get_model_params(self, model_type, X, y):
        # available_models = list(ModelsRegistry.registry[BINARY_CLASSIFICATION].keys())
        # print("available_models", available_models)
        # model_type = np.random.permutation(available_models)[0]
        print("model_type", model_type)
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

    def train_model(self, params, X, y):
        early_stop = EarlyStopping({"metric": {"name": "logloss"}})
        time_constraint = TimeConstraint({"train_seconds_time_limit": 10})
        il = IterativeLearner(params, callbacks=[early_stop, time_constraint])
        il_key = il.get_params_key()
        print("KEY", il_key)
        if il_key in self._models_params_keys:
            print("Already trained !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return None
        self._models_params_keys += [il_key]
        il.train({"train": {"X": X, "y": y}})
        print("oof--->>>", il.get_out_of_folds().shape, X.shape, y.shape)
        return il

    def fit(self, X, y):
        print("FIT", X.head())
        print("FIT", X.columns)
        print("automl fit", X.shape, y.shape)
        print(X.index[:10])
        print(y.index[:10])
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        print("FIT", X.columns)

        # start with not-so-random models
        for model_type in self._algorithms:
            for i in range(self._start_random_models):
                params = self._get_model_params(model_type, X, y)
                m = self.train_model(params, X, y)
                if m is not None:
                    self._models += [m]
        # perform hill climbing steps on best models
        for hill_climbing in range(self._hill_climbing_steps):
            # get models orderer by loss
            models = []
            for m in self._models:
                models += [(m.callbacks.callbacks[0].final_loss, m)]
            models = sorted(models, key=lambda x: x[0])
            print("models", models)
            for i in range(self._top_models_to_improve):
                print(">", models[i])
                m = models[i][1]
                if m is None:
                    continue
                print(m.params.get("learner"))
                print(m is None)
                params_1, params_2 = HillClimbing.get(m.params.get("learner"))
                for p in [params_1, params_2]:
                    if p is not None:
                        print(m is None)
                        all_params = copy.deepcopy(m.params)
                        all_params["learner"] = p
                        new_model = self.train_model(all_params, X, y)
                        if new_model is not None:
                            self._models += [new_model]

        self.ensemble = Ensemble()
        self.ensemble.fit(self._models, y)

        max_loss = 1000000.0
        for il in self._models:
            print(
                "Learner {} final loss {}".format(
                    il.learners[0].algorithm_short_name,
                    il.callbacks.callbacks[0].final_loss,
                )
            )
            if il.callbacks.callbacks[0].final_loss < max_loss:
                self._best_model = il
                max_loss = il.callbacks.callbacks[0].final_loss
        print("Best learner")
        print(self._best_model.uid, max_loss)
        print("FIT", X.columns)

    def predict(self, X):
        print("Predict", X.head())
        # return self._best_model.predict(X)
        return self.ensemble.predict(X)

    def to_json(self):
        save_details = []
        for il in self._models:
            save_details += [il.save()]
        return save_details

    def from_json(self, json_data):
        self._models = []
        for save_detail in json_data:
            il = IterativeLearner()
            il.load(save_detail)
            self._models += [il]
