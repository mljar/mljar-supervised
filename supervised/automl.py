import sys
import json
import copy
import time
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

tqdm.pandas()

from supervised.models.learner_xgboost import XgbLearner
from supervised.iterative_learner_framework import IterativeLearner
from supervised.callbacks.early_stopping import EarlyStopping
from supervised.callbacks.metric_logger import MetricLogger
from supervised.callbacks.time_constraint import TimeConstraint
from supervised.metric import Metric
from supervised.tuner.random_parameters import RandomParameters
from supervised.tuner.registry import ModelsRegistry
from supervised.tuner.registry import BINARY_CLASSIFICATION
from supervised.tuner.registry import MULTICLASS_CLASSIFICATION
from supervised.tuner.registry import REGRESSION
from supervised.tuner.preprocessing_tuner import PreprocessingTuner
from supervised.tuner.hill_climbing import HillClimbing
from supervised.models.ensemble import Ensemble
from supervised.models.compute_additional_metrics import ComputeAdditionalMetrics
from supervised.preprocessing.preprocessing_exclude_missing import (
    PreprocessingExcludeMissingValues,
)


class AutoML:
    def __init__(
        self,
        total_time_limit=60 * 60,
        learner_time_limit=120,
        algorithms=["CatBoost", "Xgboost", "RF", "LightGBM", "NN"],
        start_random_models=10,
        hill_climbing_steps=3,
        top_models_to_improve=5,
        train_ensemble=True,
        verbose=True,
        optimize_metric=None,
        ml_task = None,
        seed=1,
    ):
        self._total_time_limit = total_time_limit
        self._time_limit = (
            learner_time_limit
        )  # time limit in seconds for single learner
        self._train_ensemble = train_ensemble
        self._models = []  # instances of iterative learner framework or ensemble
        self._models_params_keys = []
        self._best_model = (
            None
        )  # it is instance of iterative learner framework or ensemble
        self._validation = {"validation_type": "kfold", "k_folds": 5, "shuffle": True}

        self._start_random_models = start_random_models
        self._hill_climbing_steps = hill_climbing_steps
        self._top_models_to_improve = top_models_to_improve
        self._algorithms = algorithms
        self._verbose = verbose

        self._fit_time = None
        self._models_train_time = {}
        self._threshold, self._metrics_details, self._max_metrics, self._confusion_matrix = (
            None,
            None,
            None,
            None,
        )
        self._seed = seed
        self._user_set_optimize_metric = optimize_metric
        self.ml_task = ml_task

    def estimate_training_times(self):
        # single models including models in the folds
        self._estimated_models_to_check = (
            len(self._algorithms) * self._start_random_models
            + self._top_models_to_improve * self._hill_climbing_steps * 2
        ) * 5

        if self._total_time_limit is not None:
            # set time limit for single model training
            # the 0.85 is safe scale factor, to not exceed time limit
            self._time_limit = (
                self._total_time_limit * 0.85 / self._estimated_models_to_check
            )

    def get_leaderboard(self):
        ldb = {
            "uid": [],
            "model_type": [],
            "metric_type": [],
            "metric_value": [],
            "train_time": [],
        }
        for m in self._models:
            ldb["uid"] += [m.uid]
            ldb["model_type"] += [m.get_name()]
            ldb["metric_type"] += [self._optimize_metric]
            ldb["metric_value"] += [m.get_final_loss()]
            ldb["train_time"] += [m.get_train_time()]
        return pd.DataFrame(ldb)

    def get_additional_metrics(self):
        # 'target' - the target after processing used for model training
        # 'prediction' - out of folds predictions of model
        oof_predictions = self._best_model.get_out_of_folds()
        prediction_cols = [c for c in oof_predictions.columns if "prediction" in c]
        self._metrics_details, self._max_metrics, self._confusion_matrix = ComputeAdditionalMetrics.compute(
            oof_predictions["target"], oof_predictions[prediction_cols], self.ml_task
        )
        if self.ml_task == BINARY_CLASSIFICATION:
            self._threshold = float(
                self._max_metrics["f1"]["threshold"]
            )  # TODO: do need conversion
        # print(self._metrics_details, self._max_metrics, self._confusion_matrix)

    def _get_model_params(self, model_type, X, y):
        model_info = ModelsRegistry.registry[self.ml_task][model_type]
        model_params = RandomParameters.get(
            model_info["params"], len(self._models) + self._seed
        )
        required_preprocessing = model_info["required_preprocessing"]
        model_additional = model_info["additional"]
        preprocessing_params = PreprocessingTuner.get(
            required_preprocessing, {"train": {"X": X, "y": y}}, self.ml_task
        )

        model_params = {
            "additional": model_additional,
            "preprocessing": preprocessing_params,
            "validation": self._validation,

            "learner": {
                "model_type": model_info["class"].algorithm_short_name,
                **model_params,
            },
        }
        num_class = len(np.unique(y)) if self.ml_task == MULTICLASS_CLASSIFICATION else None
        if num_class is not None:
            model_params["learner"]["num_class"] = num_class

        return model_params

    def keep_model(self, model):
        if model is None:
            return
        self._models += [model]
        self.verbose_print(
            "Learner {} final loss {} time {} seconds".format(
                model.get_name(),
                model.get_final_loss(),
                np.round(model.get_train_time(), 2),
            )
        )
        self.log_train_time(model.get_name(), model.get_train_time())

    def train_model(self, params, X, y):

        early_stop = EarlyStopping({"metric": {"name": self._optimize_metric}})
        time_constraint = TimeConstraint({"train_seconds_time_limit": self._time_limit})
        params["ml_task"] = self.ml_task

        il = IterativeLearner(params, callbacks=[early_stop, time_constraint])
        il_key = il.get_params_key()
        if il_key in self._models_params_keys:
            self._progress_bar.update(1)
            return None
        self._models_params_keys += [il_key]
        if self.should_train_next(il.get_name()):
            il.train({"train": {"X": X, "y": y}})
            self._progress_bar.update(1)
            return il
        self._progress_bar.update(1)
        return None

    def verbose_print(self, msg):
        if self._verbose:
            self._progress_bar.write(msg)

    def log_train_time(self, model_type, train_time):
        if model_type in self._models_train_time:
            self._models_train_time[model_type] += [train_time]
        else:
            self._models_train_time[model_type] = [train_time]

    def should_train_next(self, model_type):
        # no time limit, just train, dont ask
        if self._total_time_limit is None:
            return True

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
                self.keep_model(m)

    def hill_climbing_step(self, X, y):
        for hill_climbing in range(self._hill_climbing_steps):
            # get models orderer by loss
            models = []
            for m in self._models:
                models += [(m.callbacks.callbacks[0].final_loss, m)]
            models = sorted(models, key=lambda x: x[0])
            for i in range(min(self._top_models_to_improve, len(models))):
                m = models[i][1]
                for p in HillClimbing.get(
                    m.params.get("learner"), self.ml_task, len(self._models) + self._seed
                ):
                    if p is not None:
                        all_params = copy.deepcopy(m.params)
                        all_params["learner"] = p
                        new_model = self.train_model(all_params, X, y)
                        self.keep_model(new_model)
                    else:
                        self._progress_bar.update(1)

    def ensemble_step(self, y):
        if self._train_ensemble:
            self.ensemble = Ensemble(self._optimize_metric, self.ml_task)
            oofs = self.ensemble.get_oof_matrix(self._models)
            self.ensemble.fit(oofs, y)
            self.keep_model(self.ensemble)
            self._progress_bar.update(1)

    def set_ml_task(self, y):
        target_unique_cnt = len(np.unique(y))
        if target_unique_cnt == 2:
            self.ml_task = BINARY_CLASSIFICATION
        elif target_unique_cnt <= 20:
            self.ml_task = MULTICLASS_CLASSIFICATION
        else:
            self.ml_task = REGRESSION

    def fit(self, X, y):
        start_time = time.time()

        X.reset_index(drop=True, inplace=True)
        y = np.array(y)
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame({"target": y})
        y.reset_index(drop=True, inplace=True)
        y = y["target"]

        # drops rows with missing target
        # TODO needs a better name for this preprocessing
        X, y = PreprocessingExcludeMissingValues.transform(X, y)

        # define the ml_task
        if self.ml_task is None:
            self.set_ml_task(y)

        supported_ml_tasks = [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]
        if self.ml_task not in supported_ml_tasks:
            raise Exception(
                "Unknow Machine Learning task {}. Supported tasks are: {}".format(
                    self.ml_task, supported_ml_tasks
                )
            )
        print("ML task: {}".format(self.ml_task))
        # define available algorithms
        if len(self._algorithms) == 0:
            self._algorithms = list(ModelsRegistry.registry[self.ml_task].keys())

        for a in self._algorithms:
            if a not in list(ModelsRegistry.registry[self.ml_task].keys()):
                raise Exception(
                    "The algorithm {} is not allowed to use for ML task: {}.".format(
                        a, self.ml_task
                    )
                )

        # set metric to be optimized, TODO move this to registry with dict
        if self.ml_task == BINARY_CLASSIFICATION:
            if self._user_set_optimize_metric is None:
                self._optimize_metric = "logloss"
            elif self._user_set_optimize_metric not in ["logloss", "auc"]:
                raise Exception(
                    "Metric {} is not allowed in ML task: {}".format(
                        self._user_set_optimize_metric, self.ml_task
                    )
                )
            else:
                self._optimize_metric = self._user_set_optimize_metric
        elif self.ml_task == MULTICLASS_CLASSIFICATION:
            if self._user_set_optimize_metric is None:
                self._optimize_metric = "logloss"
            elif self._user_set_optimize_metric not in ["logloss"]:
                raise Exception(
                    "Metric {} is not allowed in ML task: {}".format(
                        self._user_set_optimize_metric, self.ml_task
                    )
                )
            else:
                self._optimize_metric = self._user_set_optimize_metric
        elif self.ml_task == REGRESSION:
            if self._user_set_optimize_metric is None:
                self._optimize_metric = "mse"
            elif self._user_set_optimize_metric not in ["mse"]:
                raise Exception(
                    "Metric {} is not allowed in ML task: {}".format(
                        self._user_set_optimize_metric, self.ml_task
                    )
                )
            else:
                self._optimize_metric = self._user_set_optimize_metric

        # estimate training time
        self.estimate_training_times()

        self._progress_bar = tqdm(
            total=int(self._estimated_models_to_check / 5),
            desc="MLJAR AutoML",
            unit="model",
        )

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

        self.get_additional_metrics()
        self._fit_time = time.time() - start_time
        self._progress_bar.close()

    def predict(self, X):
        if self._best_model is not None:

            predictions = self._best_model.predict(X)

            if self.ml_task == BINARY_CLASSIFICATION:
                # need to predict the label based on predictions and threshold
                neg_label, pos_label = (
                    predictions.columns[0][2:],
                    predictions.columns[1][2:],
                )
                if neg_label == "0" and pos_label == "1":
                    neg_label, pos_label = 0, 1
                # assume that it is binary classification
                predictions["label"] = predictions.iloc[:, 1] > self._threshold
                predictions["label"] = predictions["label"].map(
                    {True: pos_label, False: neg_label}
                )
                return predictions
            elif self.ml_task == MULTICLASS_CLASSIFICATION:

                return predictions
            else:
                return predictions

        return None

    def to_json(self):
        if self._best_model is None:
            return None

        return {
            "best_model": self._best_model.to_json(),
            "threshold": self._threshold,
            "ml_task": self.ml_task,
        }

    def from_json(self, json_data):
        # pretty sure that this can be easily refactored
        if json_data["best_model"]["algorithm_short_name"] == "Ensemble":
            self._best_model = Ensemble()
            self._best_model.from_json(json_data["best_model"])
        else:
            self._best_model = IterativeLearner(json_data["best_model"].get("params"))
            self._best_model.from_json(json_data["best_model"])
        self._threshold = json_data.get("threshold")

        self.ml_task = json_data.get("ml_task")
