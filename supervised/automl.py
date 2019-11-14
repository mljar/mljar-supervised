import sys
import json
import copy
import time
import numpy as np
import pandas as pd
import logging

from tqdm.auto import tqdm
tqdm.pandas()

from supervised.model_framework import ModelFramework
from supervised.callbacks.early_stopping import EarlyStopping
from supervised.callbacks.metric_logger import MetricLogger
from supervised.callbacks.time_constraint import TimeConstraint
from supervised.utils.metric import Metric
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.tuner.preprocessing_tuner import PreprocessingTuner
from supervised.tuner.hill_climbing import HillClimbing
from supervised.tuner.mljar_tuner import MljarTuner
from supervised.algorithms.ensemble import Ensemble
from supervised.utils.compute_additional_metrics import ComputeAdditionalMetrics
from supervised.preprocessing.preprocessing_exclude_missing import (
    PreprocessingExcludeMissingValues,
)
from supervised.utils.config import LOG_LEVEL


logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.ERROR
)
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class AutoMLException(Exception):
    def __init__(self, message):
        super(AutoMLException, self).__init__(message)
        logger.error(message)


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
        ml_task=None,
        seed=1,
    ):
        logger.debug("AutoML.__init__")

        self._total_time_limit = total_time_limit
        # time limit in seconds for single learner
        self._time_limit = learner_time_limit
        
        self._train_ensemble = train_ensemble
        self._models = []  # instances of iterative learner framework or ensemble
        self._models_params_keys = []
        # it is instance of model framework or ensemble
        self._best_model = None
        # default validation
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

    def _estimate_training_times(self):
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
        target_cols = [c for c in oof_predictions.columns if "target" in c]
        self._metrics_details, self._max_metrics, self._confusion_matrix = ComputeAdditionalMetrics.compute(
            oof_predictions[target_cols], oof_predictions[prediction_cols], self.ml_task
        )
        if self.ml_task == BINARY_CLASSIFICATION:
            self._threshold = float(
                self._max_metrics["f1"]["threshold"]
            )  # TODO: do need conversion
            logger.info(
                "Metric details:\n{}\nConfusion matrix:\n{}".format(
                    self._max_metrics.transpose(), self._confusion_matrix
                )
            )
        elif self.ml_task == MULTICLASS_CLASSIFICATION:
            logger.info(
                "Metric details:\n{}\nConfusion matrix:\n{}".format(
                    self._max_metrics, self._confusion_matrix
                )
            )


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

        il = ModelFramework(params, callbacks=[early_stop, time_constraint])
        # get learner unique hash
        il_key = il.get_params_key()
        # if already trained model with such paramaters, just skip it
        if il_key in self._models_params_keys:
            il = None
        else:  # unique hash, train the model
            self._models_params_keys += [il_key]
            if self._enough_time_to_train(il.get_name()):
                il.train({"train": {"X": X, "y": y}})
            else:
                il = None

        self._progress_bar.update(1)
        return il

    def verbose_print(self, msg):
        if self._verbose:
            self._progress_bar.write(msg)

    def log_train_time(self, model_type, train_time):
        if model_type in self._models_train_time:
            self._models_train_time[model_type] += [train_time]
        else:
            self._models_train_time[model_type] = [train_time]

    def _enough_time_to_train(self, model_type):
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

    def ensemble_step(self):
        if self._train_ensemble:
            self.ensemble = Ensemble(self._optimize_metric, self.ml_task)
            oofs, target = self.ensemble.get_oof_matrix(self._models)
            self.ensemble.fit(oofs, target)
            self.keep_model(self.ensemble)
            self._progress_bar.update(1)

    def _set_ml_task(self, y):
        """ Set and validate the ML task.
        
        If ML task is not set, it trys to guess ML task based on count of unique values in the target. 
        Then it performs validation.
        """
        # if not set, guess
        if self.ml_task is None:
            target_unique_cnt = len(np.unique(y[~pd.isnull(y)]))
            if target_unique_cnt == 2:
                self.ml_task = BINARY_CLASSIFICATION
            elif target_unique_cnt <= 20:
                self.ml_task = MULTICLASS_CLASSIFICATION
            else:
                self.ml_task = REGRESSION
        # validation
        if self.ml_task not in AlgorithmsRegistry.get_supported_ml_tasks():
            raise Exception(
                "Unknow Machine Learning task {}."
                " Supported tasks are: {}".format(self.ml_task, AlgorithmsRegistry.get_supported_ml_tasks())
            )
        logger.info("AutoML task to be solved: {}".format(self.ml_task))

    def _set_algorithms(self):
        """ Set and validate available algorithms.

        If algorithms are not set, all algorithms from registry are used.
        Then perform vadlidation of algorithms.
        """
        if len(self._algorithms) == 0:
            self._algorithms = list(AlgorithmsRegistry.registry[self.ml_task].keys())

        for a in self._algorithms:
            if a not in list(AlgorithmsRegistry.registry[self.ml_task].keys()):
                raise AutoMLException(
                    "The algorithm {} is not allowed to use for ML task: {}.".format(
                        a, self.ml_task
                    )
                )
        logger.info("AutoML will use algorithms: {}".format(self._algorithms))

    def _set_metric(self):
        """ Set and validate the metric to be optimized. """
        if self.ml_task == BINARY_CLASSIFICATION:
            if self._user_set_optimize_metric is None:
                self._optimize_metric = "logloss"
            elif self._user_set_optimize_metric not in ["logloss", "auc"]:
                raise AutoMLException(
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
                raise AutoMLException(
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
                raise AutoMLException(
                    "Metric {} is not allowed in ML task: {}".format(
                        self._user_set_optimize_metric, self.ml_task
                    )
                )
            else:
                self._optimize_metric = self._user_set_optimize_metric
        logger.info("AutoML will optimize metric: {0}".format(self._optimize_metric))

    def fit(self, X, y):
        logger.debug("AutoML.fit, X data {0}, y data {1}".format(X.shape, y.shape))
        start_time = time.time()

        X.reset_index(drop=True, inplace=True)
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame({"target": np.array(y)})
        y.reset_index(drop=True, inplace=True)
        y = y["target"]
        
        #X, y = PreprocessingExcludeMissingValues.transform(X, y)

        self._set_ml_task(y)
        self._set_algorithms()
        self._set_metric()
        self._estimate_training_times()

        self._progress_bar = tqdm(
            total=int(self._estimated_models_to_check / 5), # 5 ?
            desc="MLJAR AutoML",
            unit="model",
        )

        tuner = MljarTuner(
            {
                "start_random_models": self._start_random_models,
                "hill_climbing_steps": self._hill_climbing_steps,
                "top_models_to_improve": self._top_models_to_improve,
            },
            self._algorithms,
            self.ml_task,
            self._validation,
            self._seed,
        )

        for params in tuner.get_params(X, y, self._models):
            if params is not None:
                new_model = self.train_model(params, X, y)
                self.keep_model(new_model)
            # always update progessbar, even for empty parameters, because they are counted in the progressbar
            self._progress_bar.update(1)

        self.ensemble_step()

        max_loss = 10e12
        for m in self._models:
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
            self._best_model = ModelFramework(json_data["best_model"].get("params"))
            self._best_model.from_json(json_data["best_model"])
        self._threshold = json_data.get("threshold")

        self.ml_task = json_data.get("ml_task")
