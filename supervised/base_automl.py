import os
import sys
import json
import copy
import time
import numpy as np
import pandas as pd
import logging
from tabulate import tabulate
from abc import ABC
from copy import deepcopy

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from sklearn.metrics import r2_score, accuracy_score

from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.registry import REGRESSION
from supervised.callbacks.early_stopping import EarlyStopping
from supervised.callbacks.metric_logger import MetricLogger
from supervised.callbacks.learner_time_constraint import LearnerTimeConstraint
from supervised.callbacks.total_time_constraint import TotalTimeConstraint
from supervised.ensemble import Ensemble
from supervised.exceptions import AutoMLException
from supervised.model_framework import ModelFramework
from supervised.preprocessing.exclude_missing_target import ExcludeRowsMissingTarget
from supervised.tuner.data_info import DataInfo
from supervised.tuner.mljar_tuner import MljarTuner
from supervised.utils.additional_metrics import AdditionalMetrics
from supervised.utils.config import mem
from supervised.utils.config import LOG_LEVEL
from supervised.utils.leaderboard_plots import LeaderboardPlots
from supervised.utils.metric import Metric
from supervised.preprocessing.eda import EDA
from supervised.tuner.time_controller import TimeController
from supervised.utils.data_validation import (
    check_positive_integer,
    check_greater_than_zero_integer,
    check_bool,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class BaseAutoML(BaseEstimator, ABC):
    """
    Automated Machine Learning for supervised tasks (binary classification, multiclass classification, regression).
    Warning: This class should not be used directly. Use derived classes instead.
    """

    def __init__(self):
        logger.debug("BaseAutoML.__init__")
        self._mode = None
        self._ml_task = None
        self._results_path = None
        self._total_time_limit = None
        self._model_time_limit = None
        self._algorithms = []
        self._train_ensemble = False
        self._stack_models = False
        self._eval_metric = None
        self._validation_strategy = None
        self._verbose = None
        self._explain_level = None
        self._golden_features = None
        self._features_selection = None
        self._start_random_models = None
        self._hill_climbing_steps = None
        self._top_models_to_improve = None
        self._random_state = 1234
        self._models = []  # instances of iterative learner framework or ensemble
        self._best_model = None
        self._verbose = True
        self._threshold = None  # used only in classification
        self._metrics_details = None
        self._max_metrics = None
        self._confusion_matrix = None
        self._X_path, self._y_path = None, None
        self._data_info = None
        self._model_paths = []
        self._stacked_models = None
        self._fit_level = None
        self._start_time = time.time()
        self._time_ctrl = None
        self._all_params = {}
        # https://scikit-learn.org/stable/developers/develop.html#universal-attributes
        self.n_features_in_ = None  # for scikit-learn api

    def _get_tuner_params(
        self, start_random_models, hill_climbing_steps, top_models_to_improve
    ):
        return {
            "start_random_models": start_random_models,
            "hill_climbing_steps": hill_climbing_steps,
            "top_models_to_improve": top_models_to_improve,
        }

    def _check_can_load(self):
        """ Checks if AutoML can be loaded from a folder"""
        if self.results_path is not None:
            # Dir exists and can be loaded
            if os.path.exists(self.results_path) and os.path.exists(
                os.path.join(self.results_path, "params.json")
            ):
                self.load(self.results_path)
                self._results_path = self.results_path

    def load(self, path):
        logger.info("Loading AutoML models ...")
        try:
            params = json.load(open(os.path.join(path, "params.json")))

            self._model_paths = params["saved"]
            self._mode = params.get("mode", self._mode)
            self._ml_task = params.get("ml_task", self._ml_task)
            self._results_path = params.get("results_path", self._results_path)
            self._total_time_limit = params.get(
                "total_time_limit", self._total_time_limit
            )
            self._model_time_limit = params.get(
                "model_time_limit", self._model_time_limit
            )
            self._algorithms = params.get("algorithms", self._algorithms)
            self._train_ensemble = params.get("train_ensemble", self._train_ensemble)
            self._stack_models = params.get("stack_models", self._stack_models)
            self._eval_metric = params.get("eval_metric", self._eval_metric)
            self._validation_strategy = params.get(
                "validation_strategy", self._validation_strategy
            )
            self._verbose = params.get("verbose", self._verbose)
            self._explain_level = params.get("explain_level", self._explain_level)
            self._golden_features = params.get("golden_features", self._golden_features)
            self._features_selection = params.get(
                "features_selectiom", self._features_selection
            )
            self._start_random_models = params.get(
                "start_random_models", self._start_random_models
            )
            self._hill_climbing_steps = params.get(
                "hill_climbing_steps", self._hill_climbing_steps
            )
            self._top_models_to_improve = params.get(
                "top_models_to_improve", self._top_models_to_improve
            )
            self._random_state = params.get("random_state", self._random_state)
            stacked_models = params.get("stacked")

            models_map = {}
            for model_path in self._model_paths:
                if model_path.endswith("Ensemble") or model_path.endswith(
                    "Ensemble_Stacked"
                ):
                    ens = Ensemble.load(model_path, models_map)
                    self._models += [ens]
                    models_map[ens.get_name()] = ens
                else:
                    m = ModelFramework.load(model_path)
                    self._models += [m]
                    models_map[m.get_name()] = m

            if stacked_models is not None:
                self._stacked_models = []
                for stacked_model_name in stacked_models:
                    self._stacked_models += [models_map[stacked_model_name]]

            best_model_name = None
            with open(os.path.join(path, "best_model.txt"), "r") as fin:
                best_model_name = fin.read()

            self._best_model = models_map[best_model_name]

            data_info_path = os.path.join(path, "data_info.json")
            self._data_info = json.load(open(data_info_path))
            self.n_features_in_ = self._data_info["n_features"]

            if "n_classes" in self._data_info:
                self.n_classes = self._data_info["n_classes"]

            self._fit_level = "finished"
        except Exception as e:
            raise AutoMLException(f"Cannot load AutoML directory. {str(e)}")

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
            ldb["metric_type"] += [self._eval_metric]
            ldb["metric_value"] += [m.get_final_loss()]
            ldb["train_time"] += [np.round(m.get_train_time(), 2)]
        return pd.DataFrame(ldb)

    def keep_model(self, model, model_path):
        if model is None:
            return
        self._models += [model]
        self._model_paths += [model_path]
        self.select_and_save_best()

        self.verbose_print(
            "{} {} {} trained in {} seconds".format(
                model.get_name(),
                self._eval_metric,
                np.round(model.get_final_loss(), 6),
                np.round(model.get_train_time(), 2),
            )
        )
        self._time_ctrl.log_time(
            model.get_name(), model.get_type(), self._fit_level, model.get_train_time()
        )

    def create_dir(self, model_path):
        if not os.path.exists(model_path):
            try:
                os.mkdir(model_path)
            except Exception as e:
                raise AutoMLException(f"Cannot create directory {model_path}. {str(e)}")

    def train_model(self, params):

        # do we have enough time to train?
        # if not, skip
        if not self._time_ctrl.enough_time(
            params["learner"]["model_type"], self._fit_level
        ):
            logger.info(f"Cannot train {params['name']} because of the time constraint")
            return False

        # let's create directory to log all training artifacts
        model_path = os.path.join(self._results_path, params["name"])
        self.create_dir(model_path)

        # prepare callbacks
        early_stop = EarlyStopping(
            {"metric": {"name": self._eval_metric}, "log_to_dir": model_path}
        )

        learner_time_constraint = LearnerTimeConstraint(
            {
                "learner_time_limit": self._time_ctrl.learner_time_limit(
                    params["learner"]["model_type"],
                    self._fit_level,
                    self._validation_strategy.get("k_folds", 1.0),
                ),
                "min_steps": params["additional"].get("min_steps"),
            }
        )

        total_time_constraint = TotalTimeConstraint(
            {
                "total_time_limit": self._total_time_limit
                if self._model_time_limit is None
                else None,
                "total_time_start": self._start_time,
            }
        )

        # create model framework
        mf = ModelFramework(
            params,
            callbacks=[early_stop, learner_time_constraint, total_time_constraint],
        )

        # start training
        logger.info(
            f"Train model #{len(self._models)+1} / Model name: {params['name']}"
        )
        mf.train(model_path)

        # save the model
        mf.save(model_path)

        # and keep info about the model
        self.keep_model(mf, model_path)
        return True

    def verbose_print(self, msg):
        if self._verbose > 0:
            # self._progress_bar.write(msg)
            print(msg)

    def ensemble_step(self, is_stacked=False):
        if self._train_ensemble and len(self._models) > 1:

            ensemble_path = os.path.join(
                self._results_path, "Ensemble_Stacked" if is_stacked else "Ensemble"
            )
            self.create_dir(ensemble_path)

            self.ensemble = Ensemble(
                self._eval_metric, self._ml_task, is_stacked=is_stacked
            )
            oofs, target = self.ensemble.get_oof_matrix(self._models)
            self.ensemble.fit(oofs, target)
            self.ensemble.save(ensemble_path)
            self.keep_model(self.ensemble, ensemble_path)
            return True
        return False

    def can_we_stack_them(self, y):
        # if multiclass and too many classes then No
        return True

    def get_stacked_data(self, X, mode="training"):
        # mode can be `training` or `predict`
        if self._stacked_models is None:
            return X
        all_oofs = []
        for m in self._stacked_models:
            oof = None
            if mode == "training":
                oof = m.get_out_of_folds()
            else:
                oof = m.predict(X)
                if self._ml_task == BINARY_CLASSIFICATION:
                    cols = [f for f in oof.columns if "prediction" in f]
                    if len(cols) == 2:
                        oof = pd.DataFrame({"prediction": oof[cols[1]]})

            cols = [f for f in oof.columns if "prediction" in f]
            oof = oof[cols]
            oof.columns = [f"{m.get_name()}_{c}" for c in cols]
            all_oofs += [oof]

        org_index = X.index.copy()
        X.reset_index(drop=True, inplace=True)
        X_stacked = pd.concat(all_oofs + [X], axis=1)

        X_stacked.index = org_index.copy()
        X.index = org_index.copy()
        return X_stacked

    def _perform_model_stacking(self):

        if self._stacked_models is not None:
            return

        ldb = self.get_leaderboard()
        ldb = ldb.sort_values(by="metric_value", ascending=True)

        models_map = {m.get_name(): m for m in self._models if not m._is_stacked}
        self._stacked_models = []
        models_limit = 10

        for model_type in np.unique(ldb.model_type):
            if model_type in ["Baseline"]:
                continue
            ds = ldb[ldb.model_type == model_type].copy()
            ds.sort_values(by="metric_value", inplace=True)

            for n in list(ds.name.iloc[:models_limit].values):
                self._stacked_models += [models_map[n]]

        scores = [m.get_final_loss() for m in self._stacked_models]
        self._stacked_models = [
            self._stacked_models[i] for i in np.argsort(scores).tolist()
        ]

    def prepare_for_stacking(self):
        # print("Stacked models ....")
        # do we have enough models?
        if len(self._models) < 5:
            return
        # do we have time?
        if self._total_time_limit is not None:
            time_left = self._total_time_limit - (time.time() - self._start_time)
            # we need at least 60 seconds to do anything
            if time_left < 60:
                return

        self._perform_model_stacking()

        X_stacked_path = os.path.join(self._results_path, "X_stacked.parquet")
        if os.path.exists(X_stacked_path):
            return

        X = pd.read_parquet(self._X_path)
        org_columns = X.columns.tolist()
        X_stacked = self.get_stacked_data(X)
        new_columns = X_stacked.columns.tolist()
        added_columns = [c for c in new_columns if c not in org_columns]

        # save stacked train data
        X_stacked.to_parquet(X_stacked_path, index=False)

        """
        # resue old params
        for m in self._stacked_models:
            # print(m.get_type())
            # use only Xgboost, LightGBM and CatBoost as stacked models
            if m.get_type() not in ["Xgboost", "LightGBM", "CatBoost"]:
                continue
            params = copy.deepcopy(m.params)
            params["validation"]["X_train_path"] = X_train_stacked_path
            params["name"] = params["name"] + "_Stacked"
            params["is_stacked"] = True
            # print(params)
            if "model_architecture_json" in params["learner"]:
                # the new model will be created with wider input size
                del params["learner"]["model_architecture_json"]
            if self._ml_task == REGRESSION:
                # scale added predictions in regression if the target was scaled (in the case of NN)
                target_preprocessing = params["preprocessing"]["target_preprocessing"]
                scale = None
                if "scale_log_and_normal" in target_preprocessing:
                    scale = "scale_log_and_normal"
                elif "scale_normal" in target_preprocessing:
                    scale = "scale_normal"
                if scale is not None:
                    for col in added_columns:
                        params["preprocessing"]["columns_preprocessing"][col] = [
                            scale]
            self.train_model(params)
        """

    def _save_data(self, X, y):

        self._X_path = os.path.join(self._results_path, "X.parquet")
        self._y_path = os.path.join(self._results_path, "y.parquet")

        X.to_parquet(self._X_path, index=False)

        # let's check before any conversions
        target_is_numeric = pd.api.types.is_numeric_dtype(y)
        if self._ml_task == MULTICLASS_CLASSIFICATION:
            y = y.astype(str)

        pd.DataFrame({"target": y}).to_parquet(self._y_path, index=False)

        self._validation_strategy["X_path"] = self._X_path
        self._validation_strategy["y_path"] = self._y_path
        self._validation_strategy["results_path"] = self._results_path

        columns_and_target_info = DataInfo.compute(X, y, self._ml_task)

        self.n_features_in_ = X.shape[1]
        self.n_classes = len(np.unique(y[~pd.isnull(y)]))

        self._data_info = {
            "columns": X.columns.tolist(),
            "rows": y.shape[0],
            "cols": X.shape[1],
            "target_is_numeric": target_is_numeric,
            "columns_info": columns_and_target_info["columns_info"],
            "target_info": columns_and_target_info["target_info"],
            "n_features": self.n_features_in_,
        }
        # Add n_classes if not regression
        if self._ml_task != REGRESSION:
            self._data_info["n_classes"] = self.n_classes

        if columns_and_target_info.get("num_class") is not None:
            self._data_info["num_class"] = columns_and_target_info["num_class"]
        data_info_path = os.path.join(self._results_path, "data_info.json")
        with open(data_info_path, "w") as fout:
            fout.write(json.dumps(self._data_info, indent=4))

        self._drop_data_variables(X)

    def _drop_data_variables(self, X):

        X.drop(X.columns, axis=1, inplace=True)

    def _load_data_variables(self, X_train):
        if X_train.shape[1] == 0:
            X = pd.read_parquet(self._X_path)
            for c in X.columns:
                X_train.insert(loc=X_train.shape[1], column=c, value=X[c])

        os.remove(self._X_path)
        os.remove(self._y_path)

    def save_progress(self, step=None, generated_params=None):

        if step is not None and generated_params is not None:
            self._all_params[step] = generated_params

        state = {}

        state["fit_level"] = self._fit_level
        state["time_controller"] = self._time_ctrl.to_json()
        state["all_params"] = self._all_params

        fname = os.path.join(self._results_path, "progress.json")
        with open(fname, "w") as fout:
            fout.write(json.dumps(state, indent=4))

    def load_progress(self):
        state = {}
        fname = os.path.join(self._results_path, "progress.json")
        if not os.path.exists(fname):
            return
        state = json.load(open(fname, "r"))
        self._fit_level = state.get("fit_level", self._fit_level)
        self._all_params = state.get("all_params", self._all_params)
        self._time_ctrl = TimeController.from_json(state.get("time_controller"))

    def _validate_X_predict(self, X):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        # X = check_array(X, ensure_2d=False)
        X = np.atleast_2d(X)
        n_features = X.shape[1]
        if self.n_features_in_ != n_features:
            raise ValueError(
                f"Number of features of the model must match the input. Model n_features_in_ is {self.n_features_in_} and input n_features is {n_features}. Reshape your data."
            )

    # This method builds pandas.Dataframe from input. The input can be numpy.ndarray, matrix, or pandas.Dataframe
    # This method is used to build dataframes in `fit()` and in `predict`. That's the reason y can be None (`predict()` method)
    def _build_dataframe(self, X, y=None):
        if X is None or X.shape[0] == 0:
            raise AutoMLException("Empty input dataset")
        # If Inputs are not pandas dataframes use scikit-learn validation for X array
        if not isinstance(X, pd.DataFrame):
            # Validate X as array
            X = check_array(X, ensure_2d=False)
            # Force X to be 2D
            X = np.atleast_2d(X)
            # Create Pandas dataframe from np.arrays, columns get names with the schema: feature_{index}
            X = pd.DataFrame(
                X, columns=["feature_" + str(i) for i in range(1, len(X[0]) + 1)]
            )
        # Enforce column names
        # Enforce X_train columns to be string
        X.columns = X.columns.astype(str)

        X.reset_index(drop=True, inplace=True)

        if y is None:
            return X

        # Check if y is np.ndarray, transform to pd.Series
        if isinstance(y, np.ndarray):
            y = check_array(y, ensure_2d=False)
            y = pd.Series(np.array(y), name="target")
        # if pd.DataFrame, slice first column
        elif isinstance(y, pd.DataFrame):
            y = np.array(y.iloc[:, 0])
            y = check_array(y, ensure_2d=False)
            y = pd.Series(np.array(y), name="target")

        X, y = ExcludeRowsMissingTarget.transform(X, y, warn=True)

        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        return X, y

    def _fit(self, X, y):
        """Fits the AutoML model with data"""
        if self._fit_level == "finished":
            print(
                "This model has already been fitted. You can use predict methods or select a new 'results_path' for a new a 'fit()'."
            )
            return
        # Validate input and build dataframes
        X, y = self._build_dataframe(X, y)

        self.n_features_in_ = X.shape[1]
        self.n_classes = len(np.unique(y[~pd.isnull(y)]))

        # Get attributes (__init__ params)
        self._mode = self._get_mode()
        self._ml_task = self._get_ml_task()
        self._results_path = self._get_results_path()
        self._total_time_limit = self._get_total_time_limit()
        self._model_time_limit = self._get_model_time_limit()
        self._algorithms = self._get_algorithms()
        self._train_ensemble = self._get_train_ensemble()
        self._stack_models = self._get_stack_models()
        self._eval_metric = self._get_eval_metric()
        self._validation_strategy = self._get_validation_strategy()
        self._verbose = self._get_verbose()
        self._explain_level = self._get_explain_level()
        self._golden_features = self._get_golden_features()
        self._features_selection = self._get_features_selection()
        self._start_random_models = self._get_start_random_models()
        self._hill_climbing_steps = self._get_hill_climbing_steps()
        self._top_models_to_improve = self._get_top_models_to_improve()
        self._random_state = self._get_random_state()

        try:

            self.load_progress()
            if self._fit_level == "finished":
                print(
                    "This model has already been fitted. You can use predict methods or select a new 'results_path' for a new 'fit()'."
                )
                return
            self._check_can_load()

            self.verbose_print(f"AutoML directory: {self._results_path}")
            self.verbose_print(
                f"The task is {self._ml_task} with evaluation metric {self._eval_metric}"
            )
            self.verbose_print(f"AutoML will use algorithms: {self._algorithms}")
            if self._stack_models:
                self.verbose_print("AutoML will stack models")
            if self._train_ensemble:
                self.verbose_print("AutoML will ensemble availabe models")

            self._start_time = time.time()
            if self._time_ctrl is not None:
                self._start_time -= self._time_ctrl.already_spend()

            # Automatic Exloratory Data Analysis
            if self._explain_level == 2:
                EDA.compute(X, y, os.path.join(self._results_path, "EDA"))

            # Save data
            self._save_data(X.copy(deep=False), y)

            tuner = MljarTuner(
                self._get_tuner_params(
                    self._start_random_models,
                    self._hill_climbing_steps,
                    self._top_models_to_improve,
                ),
                self._algorithms,
                self._ml_task,
                self._validation_strategy,
                self._explain_level,
                self._data_info,
                self._golden_features,
                self._features_selection,
                self._train_ensemble,
                self._stack_models,
                self._random_state,
            )
            self.tuner = tuner

            steps = tuner.steps()
            self.verbose_print(f"AutoML steps: {steps}")
            if self._time_ctrl is None:
                self._time_ctrl = TimeController(
                    self._start_time,
                    self._total_time_limit,
                    self._model_time_limit,
                    steps,
                    self._algorithms,
                )

            self._time_ctrl.log_time(
                "prepare_data",
                "prepare_data",
                "prepare_data",
                time.time() - self._start_time,
            )

            for step in steps:
                self._fit_level = step
                start = time.time()
                # self._time_start[step] = start

                if step == "stack":
                    self.prepare_for_stacking()

                generated_params = []
                if step in self._all_params:
                    generated_params = self._all_params[step]
                else:
                    generated_params = tuner.generate_params(
                        step, self._models, self._results_path, self._stacked_models
                    )

                if generated_params is None or not generated_params:
                    self.verbose_print(
                        f"Skip {step} because no parameters were generated."
                    )
                    continue
                if generated_params:
                    if "learner" in generated_params[
                        0
                    ] and not self._time_ctrl.enough_time(
                        generated_params[0]["learner"]["model_type"], self._fit_level
                    ):
                        self.verbose_print(f"Skip {step} because of the time limit.")
                    else:
                        model_str = "models" if len(generated_params) > 1 else "model"
                        self.verbose_print(
                            f"* Step {step} will try to check up to {len(generated_params)} {model_str}"
                        )

                for params in generated_params:
                    if params.get("status", "") in ["trained", "skipped", "error"]:
                        self.verbose_print(f"{params['name']}: {params['status']}.")
                        continue

                    try:
                        trained = False
                        if "ensemble" in step:
                            trained = self.ensemble_step(
                                is_stacked=params["is_stacked"]
                            )
                        else:
                            trained = self.train_model(params)
                        params["status"] = "trained" if trained else "skipped"
                        params["final_loss"] = self._models[-1].get_final_loss()
                        params["train_time"] = self._models[-1].get_train_time()
                    except Exception as e:
                        self._update_errors_report(params.get("name"), str(e))
                        params["status"] = "error"

                    self.save_progress(step, generated_params)

            self._fit_level = "finished"
            self.save_progress()

            self.verbose_print(
                f"AutoML fit time: {np.round(time.time() - self._start_time,2)} seconds"
            )

        except Exception as e:
            raise e
        finally:
            if self._X_path is not None:
                self._load_data_variables(X)

        return self

    def _update_errors_report(self, model_name, error_msg):
        """Append error message to errors.md file. """
        errors_filename = os.path.join(self._get_results_path(), "errors.md")
        with open(errors_filename, "a") as fout:
            self.verbose_print(f"There was an error during {model_name} training.")
            self.verbose_print(f"Please check {errors_filename} for details.")
            fout.write(f"## Error for {model_name}\n\n")
            fout.write(error_msg)
            link = "https://github.com/mljar/mljar-supervised/issues/new"
            fout.write(
                f"\n\nPlease set a GitHub issue with above error message at: {link}"
            )
            fout.write("\n\n")

    def select_and_save_best(self):
        # Select best model based on the lowest loss
        self._best_model = min(
            [m for m in self._models if m.is_valid()], key=lambda x: x.get_final_loss()
        )

        with open(os.path.join(self._results_path, "best_model.txt"), "w") as fout:
            fout.write(f"{self._best_model.get_name()}")

        with open(os.path.join(self._results_path, "params.json"), "w") as fout:
            params = {
                "mode": self._mode,
                "ml_task": self._ml_task,
                "results_path": self._results_path,
                "total_time_limit": self._total_time_limit,
                "model_time_limit": self._model_time_limit,
                "algorithms": self._algorithms,
                "train_ensemble": self._train_ensemble,
                "stack_models": self._stack_models,
                "eval_metric": self._eval_metric,
                "validation_strategy": self._validation_strategy,
                "verbose": self._verbose,
                "explain_level": self._explain_level,
                "golden_features": self._golden_features,
                "features_selection": self._features_selection,
                "start_random_models": self._start_random_models,
                "hill_climbing_steps": self._hill_climbing_steps,
                "top_models_to_improve": self._top_models_to_improve,
                "random_state": self._random_state,
                "saved": self._model_paths,
            }
            if self._stacked_models is not None:
                params["stacked"] = [m.get_name() for m in self._stacked_models]
            fout.write(json.dumps(params, indent=4))

        ldb = self.get_leaderboard()
        ldb.to_csv(os.path.join(self._results_path, "leaderboard.csv"), index=False)

        # save report
        ldb["Link"] = [f"[Results link]({m}/README.md)" for m in ldb["name"].values]
        ldb.insert(loc=0, column="Best model", value="")
        ldb.loc[ldb.name == self._best_model.get_name(), "Best model"] = "**the best**"

        with open(os.path.join(self._results_path, "README.md"), "w") as fout:
            fout.write(f"# AutoML Leaderboard\n\n")
            fout.write(tabulate(ldb.values, ldb.columns, tablefmt="pipe"))
            LeaderboardPlots.compute(ldb, self._results_path, fout)

    def _check_is_fitted(self):
        # First check if model can be loaded
        self._check_can_load()
        # Check if fitted
        if self._fit_level != "finished":
            raise AutoMLException(
                "This model has not been fitted yet. Please call `fit()` first."
            )

    def _base_predict(self, X):
        self._check_is_fitted()

        X = self._build_dataframe(X)
        if not isinstance(X.columns[0], str):
            X.columns = [str(c) for c in X.columns]

        input_columns = X.columns.tolist()
        for column in self._data_info["columns"]:
            if column not in input_columns:
                raise AutoMLException(
                    f"Missing column: {column} in input data. Cannot predict"
                )

        X = X[self._data_info["columns"]]
        self._validate_X_predict(X)

        # is stacked model
        if self._best_model._is_stacked:
            self._perform_model_stacking()
            X_stacked = self.get_stacked_data(X, mode="predict")

            if self._best_model.get_type() == "Ensemble":
                # Ensemble is using both original and stacked data
                predictions = self._best_model.predict(X, X_stacked)
            else:
                predictions = self._best_model.predict(X_stacked)
        else:
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
                neg_label = int()
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
                try:
                    predictions["label"] = predictions["label"].astype(np.int32)
                except Exception as e:
                    predictions["label"] = predictions["label"].astype(np.float)
            return predictions
        # Regression
        else:
            return predictions

    def _predict(self, X):

        predictions = self._base_predict(X)
        # Return predictions
        # If classification task the result is in column 'label'
        # If regression task the result is in column 'prediction'
        return (
            predictions["label"].to_numpy()
            if self._ml_task != REGRESSION
            else predictions["prediction"].to_numpy()
        )

    def _predict_proba(self, X):
        # Check is task type is correct
        if self._ml_task == REGRESSION:
            raise AutoMLException(
                f"Method `predict_proba()` can only be used when in classification tasks. Current task: '{self._ml_task}'."
            )

        # Make and return predictions
        # If classification task the result is in column 'label'
        # Need to drop `label` column.
        return self._base_predict(X).drop(["label"], axis=1).to_numpy()

    def _predict_all(self, X):
        # Make and return predictions
        return self._base_predict(X)

    def _score(self, X, y=None):
        # y default must be None for scikit-learn compatibility

        # Check if y is None
        if y is None:
            raise AutoMLException("y must be specified.")

        predictions = self._predict(X)
        return (
            r2_score(y, predictions)
            if self._ml_task == REGRESSION
            else accuracy_score(y, predictions)
        )

    def _get_mode(self):
        """ Gets the current mode"""
        self._validate_mode()
        return deepcopy(self.mode)

    def _get_ml_task(self):
        """ Gets the current ml_task. If "auto" it is determined"""
        self._validate_ml_task()
        if self.ml_task == "auto":
            classes_number = self.n_classes
            if classes_number == 2:
                self._estimator_type = "classifier"  # for sk-learn api
                return BINARY_CLASSIFICATION
            elif classes_number <= 20:
                self._estimator_type = "classifier"  # for sk-learn api
                return MULTICLASS_CLASSIFICATION
            else:
                self._estimator_type = "regressor"  # for sk-learn api
                return REGRESSION
        else:
            return deepcopy(self.ml_task)

    def _get_results_path(self):
        """ Gets the current results_path"""
        # if we already have the results path set, please return it
        if self._results_path is not None:
            return self._results_path

        self._validate_results_path()

        path = self.results_path

        if path is None:
            for i in range(1, 10001):
                name = f"AutoML_{i}"
                if not os.path.exists(name):
                    self.create_dir(name)
                    self._results_path = name
                    return name
            # If it got here, could not create, raise expection
            raise AutoMLException("Cannot create directory for AutoML results")
        elif os.path.exists(self.results_path) and os.path.exists(
            os.path.join(self.results_path, "params.json")
        ):  # AutoML already loaded, return path
            self._results_path = path
            return path
        # Dir does not exist, create it
        elif not os.path.exists(path):
            self.create_dir(path)
            self._results_path = path
            return path
        # Dir exists and is empty, use it
        elif os.path.exists(path) and not len(os.listdir(path)):
            self._results_path = path
            return path
        elif os.path.exists(path) and len(os.listdir(path)):
            raise AutoMLException(
                f"Cannot set directory for AutoML. Directory '{path}' is not empty."
            )

        raise AutoMLException("Cannot set directory for AutoML results")

    def _get_total_time_limit(self):
        """ Gets the current total_time_limit"""
        self._validate_total_time_limit()
        return deepcopy(self.total_time_limit)

    def _get_model_time_limit(self):
        """ Gets the current model_time_limit"""
        self._validate_model_time_limit()
        return deepcopy(self.model_time_limit)

    def _get_algorithms(self):
        """ Gets the current algorithms. If "auto" it is determined"""
        self._validate_algorithms()
        if self.algorithms == "auto":
            if self._get_mode() == "Explain":
                return [
                    "Baseline",
                    "Linear",
                    "Decision Tree",
                    "Random Forest",
                    "Xgboost",
                    "Neural Network",
                ]
            if self._get_mode() == "Perform":
                return [
                    "Linear",
                    "Random Forest",
                    "LightGBM",
                    "Xgboost",
                    "CatBoost",
                    "Neural Network",
                ]
            if self._get_mode() == "Compete":
                return [
                    "Linear",
                    "Decision Tree",
                    "Random Forest",
                    "Extra Trees",
                    "LightGBM",
                    "Xgboost",
                    "CatBoost",
                    "Neural Network",
                    "Nearest Neighbors",
                ]
        else:
            return deepcopy(self.algorithms)

    def _get_train_ensemble(self):
        """ Gets the current train_ensemble"""
        self._validate_train_ensemble()
        return deepcopy(self.train_ensemble)

    def _get_stack_models(self):
        """ Gets the current stack_models"""
        self._validate_stack_models()
        if self.stack_models == "auto":
            return True if self.mode == "Compete" else False
        else:
            return deepcopy(self.stack_models)

    def _get_eval_metric(self):
        """ Gets the current eval_metric"""
        self._validate_eval_metric()
        if self.eval_metric == "auto":
            if self._get_ml_task() == BINARY_CLASSIFICATION:
                return "logloss"
            elif self._get_ml_task() == MULTICLASS_CLASSIFICATION:
                return "logloss"
            elif self._get_ml_task() == REGRESSION:
                return "rmse"
        else:
            return deepcopy(self.eval_metric)

    def _get_validation_strategy(self):
        """ Gets the current validation_strategy"""
        strat = {}
        self._validate_validation_strategy()
        if self.validation_strategy == "auto":
            if self._get_mode() == "Explain":
                strat = {
                    "validation_type": "split",
                    "train_ratio": 0.75,
                    "shuffle": True,
                    "stratify": True,
                }
            elif self._get_mode() == "Perform":
                strat = {
                    "validation_type": "kfold",
                    "k_folds": 5,
                    "shuffle": True,
                    "stratify": True,
                }
            elif self._get_mode() == "Compete":
                strat = {
                    "validation_type": "kfold",
                    "k_folds": 10,
                    "shuffle": True,
                    "stratify": True,
                }
            if self._get_ml_task() == REGRESSION:
                if "stratify" in strat:
                    # it's better to always check
                    # before delete (trust me)
                    del strat["stratify"]
            return strat
        else:
            strat = deepcopy(self.validation_strategy)
            if "stratify" in strat:
                del strat["stratify"]
            return strat

    def _get_verbose(self):
        """Gets the current verbose"""
        self._validate_verbose()
        return deepcopy(self.verbose)

    def _get_explain_level(self):
        """ Gets the current explain_level"""
        self._validate_explain_level()
        if self.explain_level == "auto":
            if self._get_mode() == "Explain":
                return 2
            if self._get_mode() == "Perform":
                return 1
            if self._get_mode() == "Compete":
                return 0
        else:
            return deepcopy(self.explain_level)

    def _get_golden_features(self):
        self._validate_golden_features()
        if self.golden_features == "auto":
            if self._get_mode() == "Explain":
                return False
            if self._get_mode() == "Perform":
                return True
            if self._get_mode() == "Compete":
                return True
        else:
            return deepcopy(self.golden_features)

    def _get_features_selection(self):
        """ Gets the current features_selection"""
        self._validate_features_selection()
        if self.features_selection == "auto":
            if self._get_mode() == "Explain":
                return False
            if self._get_mode() == "Perform":
                return True
            if self._get_mode() == "Compete":
                return True
        else:
            return deepcopy(self.features_selection)

    def _get_start_random_models(self):
        """ Gets the current start_random_models"""
        self._validate_start_random_models()
        if self.start_random_models == "auto":
            if self._get_mode() == "Explain":
                return 1
            if self._get_mode() == "Perform":
                return 5
            if self._get_mode() == "Compete":
                return 10
        else:
            return deepcopy(self.start_random_models)

    def _get_hill_climbing_steps(self):
        """ Gets the current hill_climbing_steps"""
        self._validate_hill_climbing_steps()
        if self.hill_climbing_steps == "auto":
            if self._get_mode() == "Explain":
                return 0
            if self._get_mode() == "Perform":
                return 2
            if self._get_mode() == "Compete":
                return 2
        else:
            return deepcopy(self.hill_climbing_steps)

    def _get_top_models_to_improve(self):
        """ Gets the current top_models_to_improve"""
        self._validate_top_models_to_improve()
        if self.top_models_to_improve == "auto":
            if self._get_mode() == "Explain":
                return 0
            if self._get_mode() == "Perform":
                return 2
            if self._get_mode() == "Compete":
                return 3
        else:
            return deepcopy(self.top_models_to_improve)

    def _get_random_state(self):
        """ Gets the current random_state"""
        self._validate_random_state()
        return deepcopy(self.random_state)

    def _validate_mode(self):
        """ Validates mode parameter"""
        valid_modes = ["Explain", "Perform", "Compete"]
        if self.mode not in valid_modes:
            raise ValueError(
                f"Expected 'mode' to be {' or '.join(valid_modes)}, got '{self.mode}'"
            )

    def _validate_ml_task(self):
        """ Validates ml_task parameter"""
        if isinstance(self.ml_task, str) and self.ml_task == "auto":
            return

        if self.ml_task not in AlgorithmsRegistry.get_supported_ml_tasks():
            raise ValueError(
                f"Expected 'ml_task' to be {' or '.join(AlgorithmsRegistry.get_supported_ml_tasks())}, got '{self.ml_task}''"
            )

    def _validate_results_path(self):
        """ Validates path parameter"""
        if self.results_path is None or isinstance(self.results_path, str):
            return

        raise ValueError(
            f"Expected 'results_path' to be of type string, got '{type(self.results_path)}''"
        )

    def _validate_total_time_limit(self):
        """ Validates total_time_limit parameter"""
        check_greater_than_zero_integer(self.total_time_limit, "total_time_limit")

    def _validate_model_time_limit(self):
        """ Validates model_time_limit parameter"""
        if self.model_time_limit is not None:
            check_greater_than_zero_integer(self.model_time_limit, "model_time_limit")

    def _validate_algorithms(self):
        """ Validates algorithms parameter"""
        if isinstance(self.algorithms, str) and self.algorithms == "auto":
            return

        for algo in self.algorithms:
            if algo not in list(AlgorithmsRegistry.registry[self._ml_task].keys()):
                raise ValueError(
                    f"The algorithm {algo} is not allowed to use for ML task: {self._ml_task}. Allowed algorithms: {list(AlgorithmsRegistry.registry[self._ml_task].keys())}"
                )

    def _validate_train_ensemble(self):
        """ Validates train_ensemble parameter"""
        # `train_ensemble` defaults to True, no further checking required
        check_bool(self.train_ensemble, "train_ensemble")

    def _validate_stack_models(self):
        """ Validates stack_models parameter"""
        # `stack_models` defaults to "auto". If "auto" return, else check if is valid bool
        if isinstance(self.stack_models, str) and self.stack_models == "auto":
            return

        check_bool(self.stack_models, "stack_models")

    def _validate_eval_metric(self):
        """ Validates eval_metric parameter"""
        # `stack_models` defaults to "auto". If not "auto", check if is valid bool
        if isinstance(self.eval_metric, str) and self.eval_metric == "auto":
            return

        if (
            self._get_ml_task() == BINARY_CLASSIFICATION
            or self._get_ml_task() == MULTICLASS_CLASSIFICATION
        ) and self.eval_metric != "logloss":
            raise ValueError(
                f"Metric {self.eval_metric} is not allowed in ML task: {self._get_ml_task()}. \
                    Use 'log_loss'"
            )

        elif self._get_ml_task() == REGRESSION and self.eval_metric != "rmse":
            raise ValueError(
                f"Metric {self.eval_metric} is not allowed in ML task: {self._get_ml_task()}. \
                Use 'rmse'"
            )

    def _validate_validation_strategy(self):
        """ Validates validation parameter"""
        if (
            isinstance(self.validation_strategy, str)
            and self.validation_strategy == "auto"
        ):
            return

        # only validation_type is mandatory
        # other parameters of validations
        # have defaults set in their constructors
        required_keys = ["validation_type"]
        if type(self.validation_strategy) is not dict:
            raise ValueError(
                f"Expected 'validation_strategy' to be a dict, got '{type(self.validation_strategy)}'"
            )
        if not all(key in self.validation_strategy for key in required_keys):
            raise ValueError(f"Expected dict with keys: {' , '.join(required_keys)}")

    def _validate_verbose(self):
        """ Validates verbose parameter"""
        check_positive_integer(self.verbose, "verbose")

    def _validate_explain_level(self):
        """ Validates explain_level parameter"""
        if isinstance(self.explain_level, str) and self.explain_level == "auto":
            return
        valid_explain_levels = [0, 1, 2]
        # Check if explain level is 0 or greater integer
        if not (
            isinstance(self.explain_level, int)
            and self.explain_level in valid_explain_levels
        ):
            raise ValueError(
                f"Expected 'explain_level' to be {' or '.join([str(x) for x in valid_explain_levels])}, got '{self.explain_level}'"
            )

    def _validate_golden_features(self):
        """ Validates golden_features parameter"""
        if isinstance(self.golden_features, str) and self.golden_features == "auto":
            return
        check_bool(self.golden_features, "golden_features")

    def _validate_features_selection(self):
        """ Validates features_selection parameter"""
        if (
            isinstance(self.features_selection, str)
            and self.features_selection == "auto"
        ):
            return
        check_bool(self.features_selection, "features_selection")

    def _validate_start_random_models(self):
        """ Validates start_random_models parameter"""
        if (
            isinstance(self.start_random_models, str)
            and self.start_random_models == "auto"
        ):
            return
        check_greater_than_zero_integer(self.start_random_models, "start_random_models")

    def _validate_hill_climbing_steps(self):
        """ Validates hill_climbing_steps parameter"""
        if (
            isinstance(self.hill_climbing_steps, str)
            and self.hill_climbing_steps == "auto"
        ):
            return
        check_positive_integer(self.hill_climbing_steps, "hill_climbing_steps")

    def _validate_top_models_to_improve(self):
        """ Validates top_models_to_improve parameter"""
        if (
            isinstance(self.top_models_to_improve, str)
            and self.top_models_to_improve == "auto"
        ):
            return
        check_positive_integer(self.top_models_to_improve, "top_models_to_improve")

    def _validate_random_state(self):
        """ Validates random_state parameter"""
        check_positive_integer(self.random_state, "random_state")

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
