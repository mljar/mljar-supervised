import os
import sys
import json
import copy
import time
import numpy as np
import pandas as pd
import logging

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
from supervised.tuner.mljar_tuner import MljarTuner
from supervised.ensemble import Ensemble
from supervised.utils.additional_metrics import AdditionalMetrics
from supervised.utils.config import LOG_LEVEL

from supervised.preprocessing.exclude_missing_target import ExcludeRowsMissingTarget

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.ERROR
)
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

from supervised.exceptions import AutoMLException

import gc
from supervised.utils.config import mem

from tabulate import tabulate


class AutoML:
    """
    Automated Machine Learning for supervised tasks (binary classification, multiclass classification, regression).
    """

    def __init__(
        self,
        results_path=None,
        total_time_limit=60 * 60,
        model_time_limit=None,
        algorithms=["Decision Tree"],  # ["Random Forest", "Xgboost"],
        tuning_mode="Sport",
        train_ensemble=True,
        optimize_metric=None,
        validation={"validation_type": "kfold", "k_folds": 5, "shuffle": True},
        verbose=True,
        ml_task=None,
        seed=1,
    ):
        """
        Create the AutoML object. Initialize directory for results.

        :param results_path: The path where all results will be saved. 
        If left `None` then the name of directory will be generated, with schema: AutoML_{number},
        where number can be from 1 to 100 - depends which direcory name will be available.

        If the `results_path` will point to directory with AutoML results, then all models will be loaded.
        
        :param total_time_limit: The time limit in seconds for AutoML training. It is not used when `model_time_limit` is not `None`.
        
        :param model_time_limit: The time limit in seconds for training single model. 
        If `model_time_limit` is set, the `total_time_limit` is not respected. 
        Single model can contain several learners, for example in the case of 10-fold cross-validation, one model will have 10 learners.
        Based on `model_time_limit` the time limit for single learner is computed.
        
        :param algorithms: The list of algorithms that will be used in the training.
        
        :param tuning_mode: The mode for tuning. It can be: `Normal`, `Sport`, `Insane`, `Perfect`. The names are kept the same as in https://mljar.com application.
        
        Each mode describe how many models will be checked:
        
        - `Normal` - about 5-10 models of each algorithm will be trained,
        - `Sport` - about 10-15 models of each algorithm will be trained,
        - `Insane` - about 15-20 models of each algorithm will be trained,
        - `Perfect` - about 25-35 models of each algorithm will be trained.
        
        You can also set how many models will be trained with `set_advanced` method.
        
        :param train_ensemble: If true then at the end of models training the ensemble will be created.
        
        :param optimize_metric: The metric to be optimized. (not implemented yet, please left `None`)
        
        :param validation: The JSON with validation type. Right now only Cross-Validation is supported. 
        The example JSON parameters for validation:
        ```
        {"validation_type": "kfold", "k_folds": 5, "shuffle": True, "stratify": True, "random_seed": 123}
        ```
        :param verbose: Not implemented yet.
        :param ml_task: The machine learning task that will be solved. Can be: `"binary_classification", "multiclass_classification", "regression"`.
        If left `None` AutoML will try to guess the task based on target values. 
        If there will be only 2 values in the target, then task will be set to `"binary_classification"`.
        If number of values in the target will be between 2 and 20 (included), then task will be set to `"multiclass_classification"`.
        In all other casses, the task is set to `"regression"`.
        
        :param seed: The seed for random generator.
        
        """
        logger.debug("AutoML.__init__")

        # total_time_limit is the time for computing for all models
        # model_time_limit is the time for computing a single model
        # if model_time_limit is None then its value is computed from total_time_limit
        # if total_time_limit is set and model_time_limit is set, then total_time_limit constraint will be omitted
        self._total_time_limit = total_time_limit
        self._model_time_limit = model_time_limit
        # time limit in seconds for single learner (model consists of learners)
        # the value is computed before fit, initilize with any number
        self._time_limit = 1

        self._train_ensemble = train_ensemble
        self._models = []  # instances of iterative learner framework or ensemble

        # it is instance of model framework or ensemble
        self._best_model = None
        self._validation = validation
        self.set_tuning_mode("Sport")

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
        self._ml_task = ml_task
        self._tuner_params = {
            "start_random_models": self._start_random_models,
            "hill_climbing_steps": self._hill_climbing_steps,
            "top_models_to_improve": self._top_models_to_improve,
        }

        self._X_train_path, self._y_train_path = None, None
        self._X_validation_path, self._y_validation_path = None, None

        self._data_info = None
        self._model_paths = []

        self._results_path = results_path
        self._set_results_dir()

    def set_tuning_mode(self, mode="Normal"):
        if mode == "Sport":
            self._start_random_models = 10
            self._hill_climbing_steps = 2
            self._top_models_to_improve = 3
        if mode == "Insane":
            self._start_random_models = 15
            self._hill_climbing_steps = 3
            self._top_models_to_improve = 4
        if mode == "Perfect":
            self._start_random_models = 25
            self._hill_climbing_steps = 5
            self._top_models_to_improve = 5
        else:  # Normal
            self._start_random_models = 5
            self._hill_climbing_steps = 1
            self._top_models_to_improve = 2

    def set_advanced(
        self, start_random_models=1, hill_climbing_steps=0, top_models_to_improve=0
    ):
        """
        Advanced set of tuning parameters. 

        :param start_random_models: Number of not-so-random models to check for each algorithm.
        :param hill_climbing_steps: Number of hill climbing steps during tuning.
        :param top_models_to_improve: Number of top models (of each algorithm) which will be considered for improving in hill climbing steps.
        """
        self._start_random_models = start_random_models
        self._hill_climbing_steps = hill_climbing_steps
        self._top_models_to_improve = top_models_to_improve
        self._tuner_params = {
            "start_random_models": self._start_random_models,
            "hill_climbing_steps": self._hill_climbing_steps,
            "top_models_to_improve": self._top_models_to_improve,
        }

    def _set_results_dir(self):
        if self._results_path is None:
            found = False
            for i in range(1, 101):
                self._results_path = f"AutoML_{i}"
                if not os.path.exists(self._results_path):
                    found = True
                    break
            if not found:
                raise AutoMLException("Cannot create directory for AutoML results")

        if os.path.exists(self._results_path) and os.path.exists(
            os.path.join(self._results_path, "params.json")
        ):
            print(f"Directory {self._results_path} already exists")
            self.load()
        elif self._results_path is not None:

            if not os.path.exists(self._results_path):
                print(f"Create directory {self._results_path}")
                try:
                    os.mkdir(self._results_path)
                except Exception as e:
                    raise AutoMLException(
                        f"Cannot create directory {self._results_path}"
                    )
            elif os.path.exists(self._results_path) and len(
                os.listdir(self._results_path)
            ):
                raise AutoMLException(
                    f"Cannot set directory for AutoML. Directory {self._results_path} is not empty."
                )
        else:
            raise AutoMLException("Cannot set directory for AutoML results")

    def load(self):
        logger.info("Loading AutoML models ...")
        try:
            params = json.load(open(os.path.join(self._results_path, "params.json")))

            self._model_paths = params["saved"]
            self._ml_task = params["ml_task"]
            self._optimize_metric = params["optimize_metric"]

            models_map = {}
            for model_path in self._model_paths:
                if model_path.endswith("ensemble"):
                    ens = Ensemble.load(model_path, models_map)
                    models_map[ens.get_name()] = ens
                else:
                    m = ModelFramework.load(model_path)
                    self._models += [m]
                    models_map[m.get_name()] = m

            best_model_name = None
            with open(os.path.join(self._results_path, "best_model.txt"), "r") as fin:
                best_model_name = fin.read()

            self._best_model = models_map[best_model_name]

            data_info_path = os.path.join(self._results_path, "data_info.json")
            self._data_info = json.load(open(data_info_path))
        except Exception as e:
            raise AutoMLException(f"Cannot load AutoML directory. {str(e)}")

    def _estimate_training_times(self):
        # single models including models in the folds
        self._estimated_models_to_check = (
            len(self._algorithms) * self._start_random_models
            + self._top_models_to_improve * self._hill_climbing_steps * 2
        )
        if self._model_time_limit is not None:
            k = self._validation.get("k_folds", 1.0)
            self._time_limit = self._model_time_limit / k
        elif self._total_time_limit is not None:
            # set time limit for single model training
            # the 0.85 is safe scale factor, to not exceed time limit
            # scaling is added because number of models to be trained are estimate
            k = self._validation.get("k_folds", 1.0)
            self._time_limit = (
                self._total_time_limit * 0.85 / self._estimated_models_to_check / k
            )
        print(
            f"AutoML will try to check about {int(self._estimated_models_to_check)} models"
        )

    def get_leaderboard(self):
        ldb = {
            "name": [],
            "model_type": [],
            "metric_type": [],
            "metric_value": [],
            "train_time": [],
        }
        for m in self._models:
            ldb["name"] += [m.get_name()]
            ldb["model_type"] += [m.get_type()]
            ldb["metric_type"] += [self._optimize_metric]
            ldb["metric_value"] += [m.get_final_loss()]
            ldb["train_time"] += [np.round(m.get_train_time(), 2)]
        return pd.DataFrame(ldb)

    def get_additional_metrics(self):

        additional_metrics = self._best_model.get_additional_metrics()
        # AdditionalMetrics.compute(
        #    oof_predictions[target_cols],
        #    oof_predictions[prediction_cols],
        #    self._ml_task,
        # )
        if self._ml_task == BINARY_CLASSIFICATION:

            self._metrics_details = additional_metrics["metric_details"]
            self._max_metrics = additional_metrics["max_metrics"]
            self._confusion_matrix = additional_metrics["confusion_matrix"]
            self._threshold = additional_metrics["threshold"]
            logger.info(
                "Metric details:\n{}\n\nConfusion matrix:\n{}".format(
                    self._max_metrics.transpose(), self._confusion_matrix
                )
            )
            with open(
                os.path.join(self._results_path, "best_model_metrics.txt"), "w"
            ) as fout:
                fout.write(
                    "Metric details:\n{}\n\nConfusion matrix:\n{}".format(
                        self._max_metrics.transpose(), self._confusion_matrix
                    )
                )

        elif self._ml_task == MULTICLASS_CLASSIFICATION:

            max_metrics = additional_metrics["max_metrics"]
            confusion_matrix = additional_metrics["confusion_matrix"]

            logger.info(
                "Metric details:\n{}\nConfusion matrix:\n{}".format(
                    max_metrics, confusion_matrix
                )
            )
            with open(
                os.path.join(self._results_path, "best_model_metrics.txt"), "w"
            ) as fout:
                fout.write("Metric details:\n{}\n\n".format(max_metrics.transpose()))
                fout.write("Confusion matrix:\n{}".format(confusion_matrix))

    def keep_model(self, model):
        if model is None:
            return
        self._models += [model]
        self.verbose_print(
            "{} final {} {} time {} seconds".format(
                model.get_type(),
                self._optimize_metric,
                model.get_final_loss(),
                np.round(model.get_train_time(), 2),
            )
        )
        self.log_train_time(model.get_type(), model.get_train_time())

    def train_model(self, params):

        model_path = os.path.join(self._results_path, params["name"])

        early_stop = EarlyStopping(
            {"metric": {"name": self._optimize_metric}, "log_to_dir": model_path}
        )
        time_constraint = TimeConstraint({"train_seconds_time_limit": self._time_limit})
        mf = ModelFramework(params, callbacks=[early_stop, time_constraint])

        if self._enough_time_to_train(mf.get_type()):

            logger.info(
                f"Train model #{len(self._models)+1} / Model name: {params['name']}"
            )

            try:
                os.mkdir(model_path)
            except Exception as e:
                raise AutoMLException(f"Cannot create directory {model_path}")

            mf.train(model_path)

            mf.save(model_path)
            self._model_paths += [model_path]

            self.keep_model(mf)

        else:
            logger.info(
                f"Cannot check more models of {mf.get_type()} because of time constraint"
            )
        # self._progress_bar.update(1)

    def verbose_print(self, msg):
        if self._verbose:
            # self._progress_bar.write(msg)
            print(msg)

    def log_train_time(self, model_type, train_time):
        if model_type in self._models_train_time:
            self._models_train_time[model_type] += [train_time]
        else:
            self._models_train_time[model_type] = [train_time]

    def _enough_time_to_train(self, model_type):
        # if model_time_limit is set, train every model
        # do not apply total_time_limit
        if self._model_time_limit is not None:
            return True
        # no total time limit, just train, dont ask
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
            self.ensemble = Ensemble(self._optimize_metric, self._ml_task)
            oofs, target = self.ensemble.get_oof_matrix(self._models)
            self.ensemble.fit(oofs, target)
            self.keep_model(self.ensemble)

            ensemble_path = os.path.join(self._results_path, "ensemble")
            try:
                os.mkdir(ensemble_path)
            except Exception as e:
                raise AutoMLException(f"Cannot create directory {ensemble_path}")
            self.ensemble.save(ensemble_path)
            self._model_paths += [ensemble_path]

    def _set_ml_task(self, y):
        """ Set and validate the ML task.
        
        If ML task is not set, it trys to guess ML task based on count of unique values in the target. 
        Then it performs validation.
        """
        # if not set, guess
        if self._ml_task is None:
            target_unique_cnt = len(np.unique(y[~pd.isnull(y)]))
            if target_unique_cnt == 2:
                self._ml_task = BINARY_CLASSIFICATION
            elif target_unique_cnt <= 20:
                self._ml_task = MULTICLASS_CLASSIFICATION
            else:
                self._ml_task = REGRESSION
        # validation
        if self._ml_task not in AlgorithmsRegistry.get_supported_ml_tasks():
            raise Exception(
                "Unknow Machine Learning task {}."
                " Supported tasks are: {}".format(
                    self._ml_task, AlgorithmsRegistry.get_supported_ml_tasks()
                )
            )
        logger.info("AutoML task to be solved: {}".format(self._ml_task))
        print(f"AutoML task to be solved: { self._ml_task}")

    def _set_algorithms(self):
        """ Set and validate available algorithms.

        If algorithms are not set, all algorithms from registry are used.
        Then perform vadlidation of algorithms.
        """
        if len(self._algorithms) == 0:
            self._algorithms = list(AlgorithmsRegistry.registry[self._ml_task].keys())

        for a in self._algorithms:
            if a not in list(AlgorithmsRegistry.registry[self._ml_task].keys()):
                raise AutoMLException(
                    "The algorithm {} is not allowed to use for ML task: {}. Allowed algorithms: {}".format(
                        a,
                        self._ml_task,
                        list(AlgorithmsRegistry.registry[self._ml_task].keys()),
                    )
                )
        logger.info("AutoML will use algorithms: {}".format(self._algorithms))
        print(f"AutoML will use algorithms: {self._algorithms}")

    def _set_metric(self):
        """ Set and validate the metric to be optimized. """
        if self._ml_task == BINARY_CLASSIFICATION:
            if self._user_set_optimize_metric is None:
                self._optimize_metric = "logloss"
            elif self._user_set_optimize_metric not in ["logloss", "auc"]:
                raise AutoMLException(
                    "Metric {} is not allowed in ML task: {}".format(
                        self._user_set_optimize_metric, self._ml_task
                    )
                )
            else:
                self._optimize_metric = self._user_set_optimize_metric
        elif self._ml_task == MULTICLASS_CLASSIFICATION:
            if self._user_set_optimize_metric is None:
                self._optimize_metric = "logloss"
            elif self._user_set_optimize_metric not in ["logloss"]:
                raise AutoMLException(
                    "Metric {} is not allowed in ML task: {}".format(
                        self._user_set_optimize_metric, self._ml_task
                    )
                )
            else:
                self._optimize_metric = self._user_set_optimize_metric
        elif self._ml_task == REGRESSION:
            if self._user_set_optimize_metric is None:
                self._optimize_metric = "mse"
            elif self._user_set_optimize_metric not in ["mse"]:
                raise AutoMLException(
                    "Metric {} is not allowed in ML task: {}".format(
                        self._user_set_optimize_metric, self._ml_task
                    )
                )
            else:
                self._optimize_metric = self._user_set_optimize_metric
        logger.info(
            "AutoML will optimize for metric: {0}".format(self._optimize_metric)
        )
        print(f"AutoML will optimize for metric: {self._optimize_metric}")

    def _check_imbalanced(self, y):
        v = y.value_counts()
        # at least 10 samples of each class
        ii = v < 10
        if np.sum(ii):
            raise AutoMLException(
                f"There need to be at least 10 samples of each class, for class {list(v[ii].index)} there is {v[ii].values} samples"
            )
        # at least 1% of all samples for each class
        v = y.value_counts(normalize=True) * 100.0
        ii = v < 1.0
        if np.sum(ii):
            raise AutoMLException(
                f"There need to be at least 1% of samples of each class, for class {list(v[ii].index)} there is {v[ii].values} % of samples"
            )

    def _initial_prep(self, X_train, y_train, X_validation=None, y_validation=None):

        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        if not isinstance(X_train.columns[0], str):
            X_train.columns = [str(c) for c in X_train.columns]

        X_train.reset_index(drop=True, inplace=True)

        y_train = pd.Series(np.array(y_train), name="target")

        X_train, y_train = ExcludeRowsMissingTarget.transform(
            X_train, y_train, warn=True
        )

        return X_train, y_train, X_validation, y_validation

    def _save_data(self, X_train, y_train, X_validation=None, y_validation=None):

        self._X_train_path = os.path.join(self._results_path, "X_train.parquet")
        self._y_train_path = os.path.join(self._results_path, "y_train.parquet")

        X_train.to_parquet(self._X_train_path, index=False)

        pd.DataFrame({"target": y_train}).to_parquet(self._y_train_path, index=False)

        self._validation["X_train_path"] = self._X_train_path
        self._validation["y_train_path"] = self._y_train_path
        self._validation["results_path"] = self._results_path

        self._data_info = {
            "columns": X_train.columns.tolist(),
            "rows": X_train.shape[0],
            "cols": X_train.shape[1],
            "target_is_numeric": pd.api.types.is_numeric_dtype(y_train),
        }
        data_info_path = os.path.join(self._results_path, "data_info.json")
        with open(data_info_path, "w") as fout:
            fout.write(json.dumps(self._data_info, indent=4))

    def _del_data_variables(self, X_train, y_train):

        X_train.drop(X_train.columns, axis=1, inplace=True)

    def _load_data_variables(self, X_train):
        X = pd.read_parquet(self._X_train_path)

        for c in X.columns:
            X_train.insert(loc=X_train.shape[1], column=c, value=X[c])

        os.remove(self._X_train_path)
        os.remove(self._y_train_path)

    def fit(self, X_train, y_train, X_validation=None, y_validation=None):
        """
        Fit AutoML
        
        :param X_train: Pandas DataFrame with training data.
        :param y_train: Numpy Array with target training data.
        
        :param X_validation: Pandas DataFrame with validation data. (Not implemented yet)
        :param y_validation: Numpy Array with target of validation data. (Not implemented yet)
        
        """
        try:

            if self._best_model is not None:
                print("Best model is already set, no need to run fit. Skipping ...")
                return

            start_time = time.time()
            if not isinstance(X_train, pd.DataFrame):
                raise AutoMLException(
                    "AutoML needs X_train matrix to be a Pandas DataFrame"
                )

            if X_train is not None:
                X_train = X_train.copy(deep=False)

            X_train, y_train, X_validation, y_validation = self._initial_prep(
                X_train, y_train, X_validation, y_validation
            )
            self._save_data(X_train, y_train, X_validation, y_validation)

            self._set_ml_task(y_train)
            self._set_algorithms()
            self._set_metric()
            self._estimate_training_times()

            if self._ml_task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:
                self._check_imbalanced(y_train)

            tuner = MljarTuner(
                self._tuner_params,
                self._algorithms,
                self._ml_task,
                self._validation,
                self._seed,
            )

            # not so random step
            generated_params = tuner.get_not_so_random_params(X_train, y_train)
            self._del_data_variables(X_train, y_train)

            for params in generated_params:
                self.train_model(params)
            # hill climbing
            for params in tuner.get_hill_climbing_params(self._models):
                self.train_model(params)

            self.ensemble_step()

            max_loss = 10e12
            for i, m in enumerate(self._models):
                if m.get_final_loss() < max_loss:
                    self._best_model = m
                    max_loss = m.get_final_loss()

            self.get_additional_metrics()
            self._fit_time = time.time() - start_time
            # self._progress_bar.close()

            with open(os.path.join(self._results_path, "best_model.txt"), "w") as fout:
                fout.write(f"{self._best_model.get_name()}")

            with open(os.path.join(self._results_path, "params.json"), "w") as fout:
                params = {
                    "ml_task": self._ml_task,
                    "optimize_metric": self._optimize_metric,
                    "saved": self._model_paths,
                }
                fout.write(json.dumps(params, indent=4))

            ldb = self.get_leaderboard()
            ldb.to_csv(os.path.join(self._results_path, "leaderboard.csv"), index=False)

            # save report
            ldb["Link"] = [f"[Results link]({m}/README.md)" for m in ldb["name"].values]
            ldb.insert(loc=0, column="Best model", value="")
            ldb.loc[
                ldb.name == self._best_model.get_name(), "Best model"
            ] = "*** the best ***"
            with open(os.path.join(self._results_path, "README.md"), "w") as fout:
                fout.write(f"# AutoML Leaderboard\n\n")
                fout.write(tabulate(ldb.values, ldb.columns, tablefmt="pipe"))
        except Exception as e:
            raise e
        finally:
            if self._X_train_path is not None:
                self._load_data_variables(X_train)

    def predict(self, X):
        """
        Computes predictions from AutoML best model.

        :param X: The Pandas DataFrame with input data. The input data should have the same columns as data used for training, otherwise the `AutoMLException` will be raised.
        """
        if self._best_model is None:
            return None

        if not isinstance(X.columns[0], str):
            X.columns = [str(c) for c in X.columns]

        input_columns = X.columns.tolist()
        for column in self._data_info["columns"]:
            if column not in input_columns:
                raise AutoMLException(
                    f"Missing column: {column} in input data. Cannot predict"
                )
        X = X[self._data_info["columns"]]

        predictions = self._best_model.predict(X)

        if self._ml_task == BINARY_CLASSIFICATION:
            # need to predict the label based on predictions and threshold
            neg_label, pos_label = (
                predictions.columns[0][11:],
                predictions.columns[1][11:],
            )

            if neg_label == "0" and pos_label == "1":
                neg_label, pos_label = 0, 1
            target_is_numeric = self._data_info.get("target_is_numeric", False)
            if target_is_numeric:
                neg_label = int(neg_label)
                pos_label = int(pos_label)
            # assume that it is binary classification
            predictions["label"] = predictions.iloc[:, 1] > self._best_model._threshold
            predictions["label"] = predictions["label"].map(
                {True: pos_label, False: neg_label}
            )
            return predictions
        elif self._ml_task == MULTICLASS_CLASSIFICATION:
            target_is_numeric = self._data_info.get("target_is_numeric", False)
            if target_is_numeric:
                predictions["label"] = predictions["label"].astype(int)
            return predictions
        else:
            return predictions

    def to_json(self):
        if self._best_model is None:
            return None

        return {
            "best_model": self._best_model.to_json(),
            "threshold": self._threshold,
            "ml_task": self._ml_task,
        }

    def from_json(self, json_data):

        if json_data["best_model"]["algorithm_short_name"] == "Ensemble":
            self._best_model = Ensemble()
            self._best_model.from_json(json_data["best_model"])
        else:
            self._best_model = ModelFramework(json_data["best_model"].get("params"))
            self._best_model.from_json(json_data["best_model"])
        self._threshold = json_data.get("threshold")

        self._ml_task = json_data.get("ml_task")
