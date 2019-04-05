import json
import copy
import time
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
    def __init__(
        self,
        total_time_limit=None,
        learner_time_limit=120,
        algorithms=["CatBoost", "Xgboost", "RF", "LightGBM", "NN"],
        start_random_models=10,
        hill_climbing_steps=3,
        top_models_to_improve=5,
        train_ensemble=True,
        verbose=True,
    ):
        self._total_time_limit = total_time_limit
        self._time_limit = (
            learner_time_limit
        )  # time limit in seconds for single learner
        self._train_ensemble = train_ensemble
        self._models = []
        self._models_params_keys = []
        self._best_model = None
        self._validation = {"validation_type": "kfold", "k_folds": 5, "shuffle": True}

        self._start_random_models = start_random_models
        self._hill_climbing_steps = hill_climbing_steps
        self._top_models_to_improve = top_models_to_improve
        self._algorithms = algorithms
        self._verbose = verbose

        if self._total_time_limit is not None:

            estimated_models_to_check = (
                len(self._algorithms)
                * (
                    self._start_random_models
                    + self._top_models_to_improve * self._hill_climbing_steps * 2
                )
                * 5
            )
            # set time limit for single model training
            # the 0.85 is safe scale factor, to not exceed time limit
            self._time_limit = self._total_time_limit * 0.85 / estimated_models_to_check
            print("single time limit->", self._time_limit)

        if len(self._algorithms) == 0:
            self._algorithms = list(
                ModelsRegistry.registry[BINARY_CLASSIFICATION].keys()
            )
        self._fit_time = None
        self._models_train_time = {}

    def _get_model_params(self, model_type, X, y):
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
        time_constraint = TimeConstraint({"train_seconds_time_limit": self._time_limit})
        il = IterativeLearner(params, callbacks=[early_stop, time_constraint])
        il_key = il.get_params_key()
        if il_key in self._models_params_keys:
            return None
        self._models_params_keys += [il_key]
        if self.should_train_next(il.get_name()):
            il.train({"train": {"X": X, "y": y}})
            return il
        return None

    def verbose_print(self, msg):
        if self._verbose:
            print(msg)

    def log_train_time(self, model_type, train_time):
        if model_type in self._models_train_time:
            self._models_train_time[model_type] += [train_time]
        else:
            self._models_train_time[model_type] = [train_time]

    def should_train_next(self, model_type):
        # no time limit, just train, dont ask
        if self._total_time_limit is None:
            return True
        print(self._models_train_time)
        total_time_already_spend = (
            0
            if model_type not in self._models_train_time
            else np.sum(self._models_train_time[model_type])
        )
        mean_time_already_spend = (
            0
            if model_type not in self._models_train_time
            else np.mean(self._models_train_time[model_type])
        )

        print("Total time already spend", total_time_already_spend)
        print("Mean time already spend", mean_time_already_spend)
        print(">", 0.85 * self._total_time_limit / float(len(self._algorithms)))

        if (
            total_time_already_spend + mean_time_already_spend
            < 0.85 * self._total_time_limit / float(len(self._algorithms))
        ):
            return True
        return False

    def not_so_random_step(self, X, y):
        for model_type in self._algorithms:
            for i in range(self._start_random_models):
                params = self._get_model_params(model_type, X, y)
                m = self.train_model(params, X, y)
                if m is not None:
                    self._models += [m]
                    self.verbose_print(
                        "Learner {} final loss {} time {}".format(
                            m.get_name(), m.get_final_loss(), m.get_train_time()
                        )
                    )
                    self.log_train_time(m.get_name(), m.get_train_time())

    def hill_climbing_step(self, X, y):
        for hill_climbing in range(self._hill_climbing_steps):
            # get models orderer by loss
            models = []
            for m in self._models:
                models += [(m.callbacks.callbacks[0].final_loss, m)]
            models = sorted(models, key=lambda x: x[0])
            for i in range(min(self._top_models_to_improve, len(models))):
                m = models[i][1]
                for p in HillClimbing.get(m.params.get("learner")):
                    if p is not None:
                        all_params = copy.deepcopy(m.params)
                        all_params["learner"] = p
                        new_model = self.train_model(all_params, X, y)
                        if new_model is not None:
                            self._models += [new_model]
                            self.verbose_print(
                                "Learner {} final loss {} time {}".format(
                                    new_model.get_name(),
                                    new_model.get_final_loss(),
                                    new_model.get_train_time(),
                                )
                            )
                            self.log_train_time(
                                new_model.get_name(), new_model.get_train_time()
                            )

    def ensemble_step(self, y):
        if self._train_ensemble:
            self.ensemble = Ensemble()
            X_oof = self.ensemble.get_oof_matrix(self._models)
            self.ensemble.fit(X_oof, y)
            self._models += [self.ensemble]
            self.verbose_print(
                "Learner {} final loss {} time {}".format(
                    self.ensemble.get_name(),
                    self.ensemble.get_final_loss(),
                    self.ensemble.get_train_time(),
                )
            )
            self.log_train_time(
                self.ensemble.get_name(), self.ensemble.get_train_time()
            )

    def fit(self, X, y):
        start_time = time.time()
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        # start with not-so-random models
        self.not_so_random_step(X, y)

        # perform hill climbing steps on best models
        self.hill_climbing_step(X, y)

        # train ensemble
        self.ensemble_step(y)

        max_loss = 10e12
        for i, m in enumerate(self._models):
            if m.get_final_loss() < max_loss:
                self._best_model = m
                max_loss = m.get_final_loss()
        self._fit_time = time.time() - start_time

    def predict(self, X):
        return self._best_model.predict(X)

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
