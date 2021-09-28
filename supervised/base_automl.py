import os
import gc
import sys
import json
import copy
import time
import types
import numpy as np
import pandas as pd
import logging
import shutil
import joblib
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
from supervised.exceptions import NotTrainedException
from supervised.model_framework import ModelFramework
from supervised.preprocessing.exclude_missing_target import ExcludeRowsMissingTarget
from supervised.tuner.data_info import DataInfo
from supervised.tuner.mljar_tuner import MljarTuner
from supervised.utils.config import mem
from supervised.utils.config import LOG_LEVEL
from supervised.utils.leaderboard_plots import LeaderboardPlots
from supervised.utils.metric import Metric
from supervised.utils.metric import UserDefinedEvalMetric
from supervised.utils.automl_plots import AutoMLPlots
from supervised.preprocessing.eda import EDA
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.tuner.time_controller import TimeController
from supervised.utils.data_validation import (
    check_positive_integer,
    check_greater_than_zero_integer,
    check_bool,
    check_greater_than_zero_integer_or_float,
    check_integer,
)
from supervised.utils.utils import dump_data, load_data

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
        self._model_subpaths = []
        self._stacked_models = None
        self._fit_level = None
        self._start_time = time.time()
        self._time_ctrl = None
        self._all_params = {}
        # https://scikit-learn.org/stable/developers/develop.html#universal-attributes
        self.n_features_in_ = None  # for scikit-learn api
        self.tuner = None
        self._boost_on_errors = None
        self._kmeans_features = None
        self._mix_encoding = None
        self._max_single_prediction_time = None
        self._optuna_time_budget = None
        self._optuna_init_params = {}
        self._optuna_verbose = True
        self._n_jobs = -1

    def _get_tuner_params(
        self, start_random_models, hill_climbing_steps, top_models_to_improve
    ):
        return {
            "start_random_models": start_random_models,
            "hill_climbing_steps": hill_climbing_steps,
            "top_models_to_improve": top_models_to_improve,
        }

    def _check_can_load(self):
        """Checks if AutoML can be loaded from a folder"""
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

            self._model_subpaths = params["saved"]
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
            self._boost_on_errors = params.get("boost_on_errors", self._boost_on_errors)
            self._kmeans_features = params.get("kmeans_features", self._kmeans_features)
            self._mix_encoding = params.get("mix_encoding", self._mix_encoding)
            self._max_single_prediction_time = params.get(
                "max_single_prediction_time", self._max_single_prediction_time
            )
            self._n_jobs = params.get("n_jobs", self._n_jobs)
            self._random_state = params.get("random_state", self._random_state)
            stacked_models = params.get("stacked")

            best_model_name = params.get("best_model")
            load_on_predict = params.get("load_on_predict")
            self._fit_level = params.get("fit_level")
            lazy_load = not (
                self._fit_level is not None and self._fit_level == "finished"
            )
            load_models = self._model_subpaths
            if load_on_predict is not None and self._fit_level == "finished":
                load_models = load_on_predict
                # just in case there is check for which models should be loaded
                # fix https://github.com/mljar/mljar-supervised/issues/395
                models_needed = self.models_needed_on_predict(best_model_name)
                # join them and return unique list
                load_models = list(np.unique(load_models + models_needed))

            models_map = {}

            for model_subpath in load_models:
                if model_subpath.endswith("Ensemble") or model_subpath.endswith(
                    "Ensemble_Stacked"
                ):
                    ens = Ensemble.load(path, model_subpath, models_map)
                    self._models += [ens]
                    models_map[ens.get_name()] = ens
                else:
                    m = ModelFramework.load(path, model_subpath, lazy_load)
                    self._models += [m]
                    models_map[m.get_name()] = m

            self._best_model = None
            if best_model_name is not None:
                self._best_model = models_map.get(best_model_name)

            if stacked_models is not None and (
                self._best_model._is_stacked or self._fit_level != "finished"
            ):
                self._stacked_models = []
                for stacked_model_name in stacked_models:
                    self._stacked_models += [models_map[stacked_model_name]]

            data_info_path = os.path.join(path, "data_info.json")
            self._data_info = json.load(open(data_info_path))
            self.n_features_in_ = self._data_info["n_features"]

            if "n_classes" in self._data_info:
                self.n_classes = self._data_info["n_classes"]

        except Exception as e:
            raise AutoMLException(f"Cannot load AutoML directory. {str(e)}")

    def get_leaderboard(
        self, filter_random_feature=False, original_metric_values=False
    ):
        ldb = {
            "name": [],
            "model_type": [],
            "metric_type": [],
            "metric_value": [],
            "train_time": [],
        }
        if self._max_single_prediction_time is not None:
            ldb["single_prediction_time"] = []
        for m in self._models:
            # filter model with random feature
            if filter_random_feature and "RandomFeature" in m.get_name():
                continue
            ldb["name"] += [m.get_name()]
            ldb["model_type"] += [m.get_type()]
            ldb["metric_type"] += [self._eval_metric]
            ldb["metric_value"] += [m.get_final_loss()]
            ldb["train_time"] += [np.round(m.get_train_time(), 2)]
            if self._max_single_prediction_time is not None:
                if m._single_prediction_time is not None:
                    ldb["single_prediction_time"] += [
                        np.round(m._single_prediction_time, 4)
                    ]
                else:
                    ldb["single_prediction_time"] += [None]

        ldb = pd.DataFrame(ldb)
        # need to add argument for sorting
        # minimize_direction = m.get_metric().get_minimize_direction()
        # ldb = ldb.sort_values("metric_value", ascending=minimize_direction)

        if original_metric_values:
            if Metric.optimize_negative(self._eval_metric):
                ldb["metric_value"] *= -1.0

        return ldb

    def keep_model(self, model, model_subpath):
        if model is None:
            return

        if self._max_single_prediction_time is not None:
            # let's check the prediction time ...
            # load 2x because of model reloading during the training
            for _ in range(2):
                start_time = time.time()
                self._base_predict(self._one_sample, model)
                model._single_prediction_time = (
                    time.time() - start_time
                )  # prediction time on single sample
            # again release learners from models
            if "Ensemble" not in model.get_type():
                model.release_learners()

        self._models += [model]
        self._model_subpaths += [model_subpath]
        self.select_and_save_best()

        sign = -1.0 if Metric.optimize_negative(self._eval_metric) else 1.0
        msg = "{} {} {} trained in {} seconds".format(
            model.get_name(),
            self._eval_metric,
            np.round(sign * model.get_final_loss(), 6),
            np.round(model.get_train_time(), 2),
        )
        if model._single_prediction_time is not None:
            msg += f" (1-sample predict time {np.round(model._single_prediction_time,4)} seconds)"
        self.verbose_print(msg)
        self._time_ctrl.log_time(
            model.get_name(), model.get_type(), self._fit_level, model.get_train_time()
        )

        self.tuner.add_key(model)

    def create_dir(self, model_path):
        if not os.path.exists(model_path):
            try:
                os.mkdir(model_path)
            except Exception as e:
                raise AutoMLException(f"Cannot create directory {model_path}. {str(e)}")

    def _expected_learners_cnt(self):
        try:
            repeats = self._validation_strategy.get("repeats", 1)
            folds = self._validation_strategy.get("k_folds", 1)
            return repeats * folds
        except Exception as e:
            pass
        return 1

    def train_model(self, params):

        # do we have enough time to train?
        # if not, skip
        if not self._time_ctrl.enough_time(
            params["learner"]["model_type"], self._fit_level
        ):
            logger.info(f"Cannot train {params['name']} because of the time constraint")
            return False
        # let's create directory to log all training artifacts
        results_path, model_subpath = self._results_path, params["name"]
        model_path = os.path.join(results_path, model_subpath)
        self.create_dir(model_path)

        # prepare callbacks
        early_stop = EarlyStopping(
            {"metric": {"name": self._eval_metric}, "log_to_dir": model_path}
        )

        # disable for now
        max_time_for_learner = 3600
        if self._total_time_limit is not None:
            k_folds = self._validation_strategy.get("k_folds", 1.0)
            at_least_algorithms = 10.0

            max_time_for_learner = max(
                self._total_time_limit / k_folds / at_least_algorithms, 60
            )

        params["max_time_for_learner"] = max_time_for_learner

        total_time_constraint = TotalTimeConstraint(
            {
                "total_time_limit": self._total_time_limit
                if self._model_time_limit is None
                else None,
                "total_time_start": self._start_time,
                "expected_learners_cnt": self._expected_learners_cnt(),
            }
        )

        # create model framework
        mf = ModelFramework(
            params,
            callbacks=[early_stop, total_time_constraint],
        )

        # start training
        logger.info(
            f"Train model #{len(self._models)+1} / Model name: {params['name']}"
        )
        mf.train(results_path, model_subpath)

        # keep info about the model
        self.keep_model(mf, model_subpath)

        # save the model
        mf.save(results_path, model_subpath)

        return True

    def verbose_print(self, msg):
        if self._verbose > 0:
            # self._progress_bar.write(msg)
            print(msg)

    def ensemble_step(self, is_stacked=False):
        if self._train_ensemble and len(self._models) > 1:

            ensemble_subpath = "Ensemble_Stacked" if is_stacked else "Ensemble"
            ensemble_path = os.path.join(self._results_path, ensemble_subpath)
            self.create_dir(ensemble_path)

            self.ensemble = Ensemble(
                self._eval_metric,
                self._ml_task,
                is_stacked=is_stacked,
                max_single_prediction_time=self._max_single_prediction_time,
            )
            oofs, target, sample_weight = self.ensemble.get_oof_matrix(self._models)
            self.ensemble.fit(oofs, target, sample_weight)
            self.keep_model(self.ensemble, ensemble_subpath)
            self.ensemble.save(self._results_path, ensemble_subpath)
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
        X_stacked = pd.concat([X] + all_oofs, axis=1)

        X_stacked.index = org_index.copy()
        X.index = org_index.copy()
        return X_stacked

    def _perform_model_stacking(self):

        if self._stacked_models is not None:
            return

        ldb = self.get_leaderboard(filter_random_feature=True)
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

    def get_stacking_minimum_time_needed(self):
        try:
            ldb = self.get_leaderboard(filter_random_feature=True)
            ldb = ldb.sort_values(by="metric_value", ascending=True)
            return min(2.0 * ldb.iloc[0]["train_time"], 60)
        except Exception as e:
            return 60

    def prepare_for_stacking(self):
        # print("Stacked models ....")
        # do we have enough models?
        if len(self._models) < 5:
            return
        # do we have time?
        if self._total_time_limit is not None:
            time_left = self._total_time_limit - (time.time() - self._start_time)
            # we need some time to start stacking
            # it should be at least 60 seconds for larger data
            # but for small data it can be less
            if time_left < self.get_stacking_minimum_time_needed():
                return
        # too many classes and models
        if self._ml_task == MULTICLASS_CLASSIFICATION:
            if self.n_classes * len(self._models) > 1000:
                return

        self._perform_model_stacking()

        X_stacked_path = os.path.join(self._results_path, "X_stacked.data")
        if os.path.exists(X_stacked_path):
            return

        X = load_data(self._X_path)
        org_columns = X.columns.tolist()
        X_stacked = self.get_stacked_data(X)
        new_columns = X_stacked.columns.tolist()
        added_columns = [c for c in new_columns if c not in org_columns]

        # save stacked train data
        dump_data(X_stacked_path, X_stacked)

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

    def _save_data(self, X, y, sample_weight=None, cv=None):
        # save information about original data
        self._save_data_info(X, y, sample_weight)

        # handle drastic imbalance
        # assure at least 20 samples of each class
        # for binary and multiclass classification
        self._handle_drastic_imbalance(X, y, sample_weight)

        # prepare path for saving files
        self._X_path = os.path.join(self._results_path, "X.data")
        self._y_path = os.path.join(self._results_path, "y.data")
        self._sample_weight_path = None
        if sample_weight is not None:
            self._sample_weight_path = os.path.join(
                self._results_path, "sample_weight.data"
            )
            dump_data(
                self._sample_weight_path, pd.DataFrame({"sample_weight": sample_weight})
            )

        dump_data(self._X_path, X)

        if self._ml_task == MULTICLASS_CLASSIFICATION:
            y = y.astype(str)

        dump_data(self._y_path, pd.DataFrame({"target": y}))

        # set paths in validation parameters
        self._validation_strategy["X_path"] = self._X_path
        self._validation_strategy["y_path"] = self._y_path
        self._validation_strategy["results_path"] = self._results_path
        if sample_weight is not None:
            self._validation_strategy["sample_weight_path"] = self._sample_weight_path

        if cv is not None:
            self._validation_strategy["cv_path"] = os.path.join(
                self._results_path, "cv.data"
            )
            joblib.dump(cv, self._validation_strategy["cv_path"])

        if self._max_single_prediction_time is not None:
            self._one_sample = X.iloc[:1].copy(deep=True)

    def _handle_drastic_imbalance(self, X, y, sample_weight=None):
        if self._ml_task == REGRESSION:
            return
        classes, cnts = np.unique(y, return_counts=True)
        min_samples_per_class = 20
        if self._validation_strategy is not None:
            min_samples_per_class = max(
                min_samples_per_class, self._validation_strategy.get("k_folds", 0)
            )
        for i in range(len(classes)):
            if cnts[i] < min_samples_per_class:
                append_samples = min_samples_per_class - cnts[i]
                new_X = (
                    X[y == classes[i]]
                    .sample(n=append_samples, replace=True, random_state=1)
                    .reset_index(drop=True)
                )
                if sample_weight is not None:
                    new_sample_weight = (
                        sample_weight[y == classes[i]]
                        .sample(n=append_samples, replace=True, random_state=1)
                        .reset_index(drop=True)
                    )
                for j in range(new_X.shape[0]):
                    X.loc[X.shape[0]] = new_X.loc[j]
                    y.loc[y.shape[0]] = classes[i]
                    if sample_weight is not None:
                        sample_weight.loc[
                            sample_weight.shape[0]
                        ] = new_sample_weight.loc[j]

    def _save_data_info(self, X, y, sample_weight=None):

        target_is_numeric = pd.api.types.is_numeric_dtype(y)
        if self._ml_task == MULTICLASS_CLASSIFICATION:
            y = y.astype(str)

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
            "is_sample_weighted": sample_weight is not None,
        }
        # Add n_classes if not regression
        if self._ml_task != REGRESSION:
            self._data_info["n_classes"] = self.n_classes

        if columns_and_target_info.get("num_class") is not None:
            self._data_info["num_class"] = columns_and_target_info["num_class"]
        data_info_path = os.path.join(self._results_path, "data_info.json")
        with open(data_info_path, "w") as fout:
            fout.write(json.dumps(self._data_info, indent=4))

    def save_progress(self, step=None, generated_params=None):
        if step is not None and generated_params is not None:
            self._all_params[step] = generated_params

        state = {}

        state["fit_level"] = self._fit_level
        state["time_controller"] = self._time_ctrl.to_json()
        state["all_params"] = self._all_params
        state["adjust_validation"] = self._adjust_validation

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
        self._adjust_validation = state.get("adjust_validation", False)

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
    def _build_dataframe(self, X, y=None, sample_weight=None):
        if X is None or X.shape[0] == 0:
            raise AutoMLException("Empty input dataset")
        # If Inputs are not pandas dataframes use scikit-learn validation for X array
        if not isinstance(X, pd.DataFrame):
            # Validate X as array
            X = check_array(X, ensure_2d=False, force_all_finite=False)
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
            y = check_array(
                y,
                ensure_2d=False,
                dtype="str" if PreprocessingUtils.is_categorical(y) else "numeric",
            )
            y = pd.Series(np.array(y), name="target")
        # if pd.DataFrame, slice first column
        elif isinstance(y, pd.DataFrame):
            y = np.array(y.iloc[:, 0])
            y = check_array(y, ensure_2d=False)
            y = pd.Series(np.array(y), name="target")

        if sample_weight is not None:
            if isinstance(sample_weight, np.ndarray):
                sample_weight = check_array(sample_weight, ensure_2d=False)
                sample_weight = pd.Series(np.array(sample_weight), name="sample_weight")
            elif isinstance(sample_weight, pd.DataFrame):
                sample_weight = np.array(sample_weight.iloc[:, 0])
                sample_weight = check_array(sample_weight, ensure_2d=False)
                sample_weight = pd.Series(np.array(sample_weight), name="sample_weight")

        X, y, sample_weight = ExcludeRowsMissingTarget.transform(
            X, y, sample_weight, warn=True
        )

        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        if sample_weight is not None:
            sample_weight.reset_index(drop=True, inplace=True)

        return X, y, sample_weight

    def _apply_constraints(self):

        if "Neural Network" in self._algorithms and self._n_jobs != -1:
            self._algorithms.remove("Neural Network")
            self.verbose_print(
                "Neural Network algorithm was disabled because it doesn't support n_jobs parameter."
            )
        if "Linear" in self._algorithms and not (
            self.n_rows_in_ < 10000 and self.n_features_in_ < 1000
        ):
            self._algorithms.remove("Linear")
            self.verbose_print("Linear algorithm was disabled.")

        # remove algorithms in the case of multiclass
        # and too many classes and columns
        if self._ml_task == MULTICLASS_CLASSIFICATION:

            if self.n_classes >= 10 and self.n_features_in_ * self.n_classes > 500:
                if self.algorithms == "auto":
                    for a in ["CatBoost"]:
                        if a in self._algorithms:
                            self._algorithms.remove(a)

            if self.n_features_in_ * self.n_classes > 1000:

                if self.algorithms == "auto":
                    for a in ["Xgboost", "CatBoost"]:
                        if a in self._algorithms:
                            self._algorithms.remove(a)
                if self.validation_strategy == "auto":
                    self._validation_strategy = {
                        "validation_type": "split",
                        "train_ratio": 0.9,
                        "shuffle": True,
                    }
                    if self._get_ml_task() != REGRESSION:
                        self._validation_strategy["stratify"] = True

            if self.n_features_in_ * self.n_classes > 10000:
                if self.algorithms == "auto":
                    for a in ["Random Forest", "Extra Trees"]:
                        if a in self._algorithms:
                            self._algorithms.remove(a)

        # Adjust the validation type based on speed of Decision Tree learning
        if (
            self._get_mode() == "Compete"
            and self._total_time_limit is not None
            and self.validation_strategy == "auto"
            and self._validation_strategy["validation_type"]
            != "split"  # split is the fastest validation type, no need to change
        ):
            # the validation will be adjusted after first Decision Tree learning on
            # train/test split (1-fold)
            self._adjust_validation = True
            self._validation_strategy = self._fastest_validation()

    def _fastest_validation(self):
        strategy = {"validation_type": "split", "train_ratio": 0.9, "shuffle": True}
        if self._get_ml_task() != REGRESSION:
            strategy["stratify"] = True
        return strategy

    def _set_adjusted_validation(self):
        if self._validation_strategy["validation_type"] != "split":
            return
        train_time = self._models[-1].get_train_time()
        # the time of Decision Tree training multiply by 5.0
        # to get the rough estimation how much time is needed for
        # other algorithms
        one_fold_time = train_time * 5.0
        # it will be good to train at least 10 models
        min_model_cnt = 10.0
        # the number of folds we can afford during the training
        folds_cnt = np.round(self._total_time_limit / one_fold_time / min_model_cnt)

        # adjust the validation if possible
        if folds_cnt >= 5.0:
            self.verbose_print(f"Adjust validation. Remove: {self._model_subpaths[0]}")
            k_folds = 5
            if folds_cnt >= 15:
                k_folds = 10
            # too small dataset for stacking
            if self.n_rows_in_ < 500:
                self._stack_models = False
                self.verbose_print(
                    "*** Disable stacking for small dataset (nrows < 500)"
                )

            self._validation_strategy["validation_type"] = "kfold"
            del self._validation_strategy["train_ratio"]
            self._validation_strategy["k_folds"] = k_folds
            self.tuner._validation_strategy = self._validation_strategy
            shutil.rmtree(
                os.path.join(self._results_path, self._model_subpaths[0]),
                ignore_errors=True,
            )
            del self._models[0]
            del self._model_subpaths[0]
            del self.tuner._unique_params_keys[0]
            self._adjust_validation = False
            cv = []
            if self._validation_strategy.get("shuffle", False):
                cv += ["Shuffle"]
            if self._validation_strategy.get("stratify", False):
                cv += ["Stratify"]
            self.select_and_save_best()  # save validation strategy

            self.verbose_print(f"Validation strategy: {k_folds}-fold CV {','.join(cv)}")
        else:
            # cant stack models for train/test split
            self._stack_models = False
            self.verbose_print("Disable stacking for split validation")

        self._apply_constraints_stack_models()

    def _apply_constraints_stack_models(self):

        if self._validation_strategy["validation_type"] == "split":
            if self._stack_models:
                self.verbose_print("Disable stacking for split validation")
            self._stack_models = False
            self._boost_on_errors = False
        if "repeats" in self._validation_strategy:
            if self._stack_models:
                self.verbose_print("Disable stacking for repeated validation")
            self._stack_models = False
            self._boost_on_errors = False

        # update Tuner
        if self.tuner is not None:
            self.tuner._stack_models = self._stack_models
            self.tuner._boost_on_errors = self._boost_on_errors

        # update Time Controler
        if self._time_ctrl is not None:
            self._time_ctrl._is_stacking = self._stack_models

            if "stack" in self._time_ctrl._steps and not self._stack_models:
                self._time_ctrl._steps.remove("stack")
            if (
                "boost_on_errors" in self._time_ctrl._steps
                and not self._boost_on_errors
            ):
                self._time_ctrl._steps.remove("boost_on_errors")

    def _fit(self, X, y, sample_weight=None, cv=None):
        """Fits the AutoML model with data"""
        if self._fit_level == "finished":
            print(
                "This model has already been fitted. You can use predict methods or select a new 'results_path' for a new a 'fit()'."
            )
            return
        # Validate input and build dataframes
        X, y, sample_weight = self._build_dataframe(X, y, sample_weight)

        self.n_rows_in_ = X.shape[0]
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
        self._boost_on_errors = self._get_boost_on_errors()
        self._kmeans_features = self._get_kmeans_features()
        self._mix_encoding = self._get_mix_encoding()
        self._max_single_prediction_time = self._get_max_single_prediction_time()
        self._optuna_time_budget = self._get_optuna_time_budget()
        self._optuna_init_params = self._get_optuna_init_params()
        self._optuna_verbose = self._get_optuna_verbose()
        self._n_jobs = self._get_n_jobs()
        self._random_state = self._get_random_state()

        self._adjust_validation = False
        self._apply_constraints()
        if not self._adjust_validation:
            # if there is no validation adjustement
            # then we can apply stack_models constraints immediately
            # if there is validation adjustement
            # then we will apply contraints after the adjustement
            self._apply_constraints_stack_models()

        try:
            self.load_progress()
            if self._fit_level == "finished":
                print(
                    "This model has already been fitted. You can use predict methods or select a new 'results_path' for a new 'fit()'."
                )
                return
            self._check_can_load()

            self.verbose_print(f"AutoML directory: {self._results_path}")
            if self._mode == "Optuna":
                ttl = int(len(self._algorithms) * self._optuna_time_budget)
                self.verbose_print("Expected computing time:")
                self.verbose_print(
                    f"Time for tuning with Optuna: len(algorithms) * optuna_time_budget = {int(len(self._algorithms) * self._optuna_time_budget)} seconds"
                )
                self.verbose_print(
                    f"There is no time limit for ML model training after Optuna tuning (total_time_limit parameter is ignored)."
                )

            self.verbose_print(
                f"The task is {self._ml_task} with evaluation metric {self._eval_metric}"
            )
            self.verbose_print(f"AutoML will use algorithms: {self._algorithms}")
            if self._stack_models:
                self.verbose_print("AutoML will stack models")
            if self._train_ensemble:
                self.verbose_print("AutoML will ensemble available models")

            self._start_time = time.time()
            if self._time_ctrl is not None:
                self._start_time -= self._time_ctrl.already_spend()

            # Automatic Exloratory Data Analysis
            if self._explain_level == 2:
                EDA.compute(X, y, os.path.join(self._results_path, "EDA"))

            # Save data
            if sample_weight is not None:
                self._save_data(
                    X.copy(deep=False),
                    y.copy(deep=False),
                    sample_weight.copy(deep=False),
                    cv,
                )
            else:
                self._save_data(X.copy(deep=False), y.copy(deep=False), None, cv)

            tuner = MljarTuner(
                self._get_tuner_params(
                    self._start_random_models,
                    self._hill_climbing_steps,
                    self._top_models_to_improve,
                ),
                self._algorithms,
                self._ml_task,
                self._eval_metric,
                self._validation_strategy,
                self._explain_level,
                self._data_info,
                self._golden_features,
                self._features_selection,
                self._train_ensemble,
                self._stack_models,
                self._adjust_validation,
                self._boost_on_errors,
                self._kmeans_features,
                self._mix_encoding,
                self._optuna_time_budget,
                self._optuna_init_params,
                self._optuna_verbose,
                self._n_jobs,
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

                if step in ["stack", "ensemble_stacked"] and not self._stack_models:
                    continue

                if step == "stack":
                    self.prepare_for_stacking()
                if "hill_climbing" in step or step in ["ensemble", "stack"]:
                    if len(self._models) == 0:
                        raise AutoMLException(
                            "No models produced. \nPlease check your data or"
                            " submit a Github issue at https://github.com/mljar/mljar-supervised/issues/new."
                        )

                generated_params = []
                if step in self._all_params:
                    generated_params = self._all_params[step]
                else:
                    generated_params = tuner.generate_params(
                        step,
                        self._models,
                        self._results_path,
                        self._stacked_models,
                        self._total_time_limit,
                    )

                if generated_params is None or not generated_params:
                    self.verbose_print(
                        f"Skip {step} because no parameters were generated."
                    )
                    continue
                if generated_params:
                    if not self._time_ctrl.enough_time_for_step(self._fit_level):
                        self.verbose_print(f"Skip {step} because of the time limit.")
                        continue
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

                        if (
                            self._adjust_validation
                            and len(self._models) == 1
                            and step == "adjust_validation"
                        ):
                            self._set_adjusted_validation()

                    except NotTrainedException as e:
                        params["status"] = "error"
                        self.verbose_print(
                            params.get("name") + " not trained. " + str(e)
                        )
                    except Exception as e:
                        import traceback

                        self._update_errors_report(
                            params.get("name"), str(e) + "\n" + traceback.format_exc()
                        )
                        params["status"] = "error"

                    self.save_progress(step, generated_params)

            if not self._models:
                raise AutoMLException("No models produced.")
            self._fit_level = "finished"
            self.save_progress()
            self.select_and_save_best(show_warnings=True)

            self.verbose_print(
                f"AutoML fit time: {np.round(time.time() - self._start_time,2)} seconds"
            )
            self.verbose_print(f"AutoML best model: {self._best_model.get_name()}")

        except Exception as e:
            raise e

        return self

    def _update_errors_report(self, model_name, error_msg):
        """Append error message to errors.md file."""
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

    def select_and_save_best(self, show_warnings=False):
        # Select best model based on the lowest loss
        self._best_model = None
        if self._models:
            model_list = [
                m
                for m in self._models
                if m.is_valid() and m.is_fast_enough(self._max_single_prediction_time)
            ]
            if model_list:
                self._best_model = min(
                    model_list,
                    key=lambda x: x.get_final_loss(),
                )
        # if none selected please select again and warn the user
        if (
            len(self._models)
            and self._best_model is None
            and self._max_single_prediction_time is not None
        ):
            if show_warnings:
                msg = (
                    "*" * 64
                    + "\nThere were no model with prediction time smaller than the limit.\n"
                    + "Please increase the prediction time for single sample,\n"
                    + "or please to use train/test split for validation\n"
                    + "*" * 64
                )
                self.verbose_print(msg)

            self._best_model = min(
                [m for m in self._models if m.is_valid()],
                key=lambda x: x.get_final_loss(),
            )

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
                "boost_on_errors": self._boost_on_errors,
                "kmeans_features": self._kmeans_features,
                "mix_encoding": self._mix_encoding,
                "max_single_prediction_time": self._max_single_prediction_time,
                "n_jobs": self._n_jobs,
                "random_state": self._random_state,
                "saved": self._model_subpaths,
                "fit_level": self._fit_level,
            }
            if self._best_model is not None:
                params["best_model"] = self._best_model.get_name()
                load_on_predict = []
                load_on_predict += self._best_model.involved_model_names()
                if self._best_model._is_stacked and self._stacked_models is not None:
                    for m in self._stacked_models:
                        load_on_predict += m.involved_model_names()
                params["load_on_predict"] = list(np.unique(load_on_predict))

            if self._stacked_models is not None:
                params["stacked"] = [m.get_name() for m in self._stacked_models]
            fout.write(json.dumps(params, indent=4))

        if self._models:
            ldb = self.get_leaderboard(original_metric_values=True)
            ldb.to_csv(os.path.join(self._results_path, "leaderboard.csv"), index=False)

            # save report
            ldb.insert(loc=0, column="Best model", value="")
            ldb.loc[
                ldb.name == self._best_model.get_name(), "Best model"
            ] = "**the best**"
            ldb["name"] = [f"[{m}]({m}/README.md)" for m in ldb["name"].values]

            with open(os.path.join(self._results_path, "README.md"), "w") as fout:
                fout.write(f"# AutoML Leaderboard\n\n")
                fout.write(tabulate(ldb.values, ldb.columns, tablefmt="pipe"))
                LeaderboardPlots.compute(ldb, self._results_path, fout)

                if self._fit_level == "finished":
                    AutoMLPlots.add(self._results_path, self._models, fout)

    def get_ensemble_models(self, ensemble_name="Ensemble"):
        try:
            params = json.load(
                open(os.path.join(self.results_path, ensemble_name, "ensemble.json"))
            )
            return [m["model"] for m in params["selected_models"]]
        except Exception as e:
            return []

    def models_needed_on_predict(self, required_model_name):
        params = json.load(open(os.path.join(self.results_path, "params.json")))
        saved_models = params.get("saved", [])
        stacked_models = params.get("stacked", [])

        if required_model_name not in saved_models:
            raise AutoMLException(
                f"Can't load model {required_model_name}. Please check if the model's name is correct."
            )
        # single model needed
        if (
            "Stacked" not in required_model_name
            and "Ensemble" not in required_model_name
        ):
            return [required_model_name]
        ensemble_models = self.get_ensemble_models("Ensemble")
        # ensemble of single models
        if required_model_name == "Ensemble":
            return ensemble_models + [required_model_name]
        # single model on stacked data
        if required_model_name != "Stacked_Ensemble":
            return list(
                np.unique(
                    ensemble_models
                    + ["Ensemble"]
                    + stacked_models
                    + [required_model_name]
                )
            )
        # must be stacked ensemble
        stacked_ensemble_models = self.get_ensemble_models("Stacked_Ensemble")
        return list(
            np.unique(
                ensemble_models
                + ["Ensemble"]
                + stacked_models
                + stacked_ensemble_models
                + [required_model_name]
            )
        )

    def _base_predict(self, X, model=None):

        if model is None:
            if self._best_model is None:
                self.load(self.results_path)
            model = self._best_model

        if model is None:
            raise AutoMLException(
                "This model has not been fitted yet. Please call `fit()` first."
            )

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
        if model._is_stacked:
            self._perform_model_stacking()
            X_stacked = self.get_stacked_data(X, mode="predict")

            if model.get_type() == "Ensemble":
                # Ensemble is using both original and stacked data
                predictions = model.predict(X, X_stacked)
            else:
                predictions = model.predict(X_stacked)
        else:
            predictions = model.predict(X)

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
            predictions["label"] = predictions.iloc[:, 1] > model._threshold
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

    def _score(self, X, y=None, sample_weight=None):
        # y default must be None for scikit-learn compatibility

        # Check if y is None
        if y is None:
            raise AutoMLException("y must be specified.")

        predictions = self._predict(X)
        return (
            r2_score(y, predictions, sample_weight=sample_weight)
            if self._ml_task == REGRESSION
            else accuracy_score(y, predictions, sample_weight=sample_weight)
        )

    def _get_mode(self):
        """Gets the current mode"""
        self._validate_mode()
        return deepcopy(self.mode)

    def _get_ml_task(self):
        """Gets the current ml_task. If "auto" it is determined"""
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
        """Gets the current results_path"""
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
        """Gets the current total_time_limit"""
        self._validate_total_time_limit()
        if self._get_mode() == "Optuna":
            return None  # there no training limit for model in the Optuna mode
            # just train and be happy with super models :)
        return deepcopy(self.total_time_limit)

    def _get_model_time_limit(self):
        """Gets the current model_time_limit"""
        self._validate_model_time_limit()
        return deepcopy(self.model_time_limit)

    def _get_algorithms(self):
        """Gets the current algorithms. If "auto" it is determined"""
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
                    "Decision Tree",
                    "Linear",
                    "Random Forest",
                    "Extra Trees",
                    "LightGBM",
                    "Xgboost",
                    "CatBoost",
                    "Neural Network",
                    "Nearest Neighbors",
                ]
            if self._get_mode() == "Optuna":
                return [
                    "Random Forest",
                    "Extra Trees",
                    "LightGBM",
                    "Xgboost",
                    "CatBoost",
                    "Neural Network",
                ]
        else:
            return deepcopy(self.algorithms)

    def _get_train_ensemble(self):
        """Gets the current train_ensemble"""
        self._validate_train_ensemble()
        return deepcopy(self.train_ensemble)

    def _get_stack_models(self):
        """Gets the current stack_models"""
        self._validate_stack_models()
        if self.stack_models == "auto":
            val = self._get_validation_strategy()
            if val.get("validation_type", "") == "custom":
                return False
            return True if self.mode in ["Compete", "Optuna"] else False
        else:
            return deepcopy(self.stack_models)

    def _get_eval_metric(self):
        """Gets the current eval_metric"""
        self._validate_eval_metric()
        if isinstance(self.eval_metric, types.FunctionType):
            UserDefinedEvalMetric().set_metric(self.eval_metric)
            return "user_defined_metric"

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
        """Gets the current validation_strategy"""
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
            elif self._get_mode() in ["Compete", "Optuna"]:
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
            if self._get_ml_task() == REGRESSION:
                if "stratify" in strat:
                    del strat["stratify"]
            return strat

    def _get_verbose(self):
        """Gets the current verbose"""
        self._validate_verbose()
        return deepcopy(self.verbose)

    def _get_explain_level(self):
        """Gets the current explain_level"""
        self._validate_explain_level()
        if self.explain_level == "auto":
            if self._get_mode() == "Explain":
                return 2
            if self._get_mode() == "Perform":
                return 1
            if self._get_mode() == "Compete":
                return 0
            if self._get_mode() == "Optuna":
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
            if self._get_mode() == "Optuna":
                return False
        else:
            return deepcopy(self.golden_features)

    def _get_features_selection(self):
        """Gets the current features_selection"""
        self._validate_features_selection()
        if self.features_selection == "auto":
            if self._get_mode() == "Explain":
                return False
            if self._get_mode() == "Perform":
                return True
            if self._get_mode() == "Compete":
                return True
            if self._get_mode() == "Optuna":
                return False
        else:
            return deepcopy(self.features_selection)

    def _get_start_random_models(self):
        """Gets the current start_random_models"""
        self._validate_start_random_models()
        if self.start_random_models == "auto":
            if self._get_mode() == "Explain":
                return 1
            if self._get_mode() == "Perform":
                return 5
            if self._get_mode() == "Compete":
                return 10
            if self._get_mode() == "Optuna":
                return 1  # just 1, because it will be tuned by Optuna
        else:
            return deepcopy(self.start_random_models)

    def _get_hill_climbing_steps(self):
        """Gets the current hill_climbing_steps"""
        self._validate_hill_climbing_steps()
        if self.hill_climbing_steps == "auto":
            if self._get_mode() == "Explain":
                return 0
            if self._get_mode() == "Perform":
                return 2
            if self._get_mode() == "Compete":
                return 2
            if self._get_mode() == "Optuna":
                return 0  # all tuning is done in Optuna
        else:
            return deepcopy(self.hill_climbing_steps)

    def _get_top_models_to_improve(self):
        """Gets the current top_models_to_improve"""
        self._validate_top_models_to_improve()
        if self.top_models_to_improve == "auto":
            if self._get_mode() == "Explain":
                return 0
            if self._get_mode() == "Perform":
                return 2
            if self._get_mode() == "Compete":
                return 3
            if self._get_mode() == "Optuna":
                return 0
        else:
            return deepcopy(self.top_models_to_improve)

    def _get_boost_on_errors(self):
        """Gets the current boost_on_errors"""
        self._validate_boost_on_errors()
        if self.boost_on_errors == "auto":
            val = self._get_validation_strategy()
            if val.get("validation_type", "") == "custom":
                return False
            if self._get_mode() == "Explain":
                return False
            if self._get_mode() == "Perform":
                return False
            if self._get_mode() == "Compete":
                return True
            if self._get_mode() == "Optuna":
                return False
        else:
            return deepcopy(self.boost_on_errors)

    def _get_kmeans_features(self):
        """Gets the current kmeans_features"""
        self._validate_kmeans_features()
        if self.kmeans_features == "auto":
            if self._get_mode() == "Explain":
                return False
            if self._get_mode() == "Perform":
                return False
            if self._get_mode() == "Compete":
                return True
            if self._get_mode() == "Optuna":
                return False
        else:
            return deepcopy(self.kmeans_features)

    def _get_mix_encoding(self):
        """Gets the current mix_encoding"""
        self._validate_mix_encoding()
        if self.mix_encoding == "auto":
            if self._get_mode() == "Explain":
                return False
            if self._get_mode() == "Perform":
                return False
            if self._get_mode() == "Compete":
                return True
            if self._get_mode() == "Optuna":
                return False
        else:
            return deepcopy(self.mix_encoding)

    def _get_max_single_prediction_time(self):
        """Gets the current max_single_prediction_time"""
        self._validate_max_single_prediction_time()
        if self.max_single_prediction_time is None:
            if self._get_mode() == "Perform":
                return 0.5  # prediction time should be under 0.5 second
            return None
        else:
            return deepcopy(self.max_single_prediction_time)

    def _get_optuna_time_budget(self):
        """Gets the current optuna_time_budget"""
        self._validate_optuna_time_budget()

        if self.optuna_time_budget is None:
            if self._get_mode() == "Optuna":
                return 3600
            return None
        else:
            if self._get_mode() != "Optuna":
                # use only for mode Optuna
                return None
            return deepcopy(self.optuna_time_budget)

    def _get_optuna_init_params(self):
        """Gets the current optuna_init_params"""
        self._validate_optuna_init_params()
        if self._get_mode() != "Optuna":
            # use only for mode Optuna
            return {}
        return deepcopy(self.optuna_init_params)

    def _get_optuna_verbose(self):
        """Gets the current optuna_verbose"""
        self._validate_optuna_verbose()
        # use only for mode Optuna
        if self._get_mode() != "Optuna":
            return True
        return deepcopy(self.optuna_verbose)

    def _get_n_jobs(self):
        """Gets the current n_jobs"""
        self._validate_n_jobs()
        return deepcopy(self.n_jobs)

    def _get_random_state(self):
        """Gets the current random_state"""
        self._validate_random_state()
        return deepcopy(self.random_state)

    def _validate_mode(self):
        """Validates mode parameter"""
        valid_modes = ["Explain", "Perform", "Compete", "Optuna"]
        if self.mode not in valid_modes:
            raise ValueError(
                f"Expected 'mode' to be {' or '.join(valid_modes)}, got '{self.mode}'"
            )

    def _validate_ml_task(self):
        """Validates ml_task parameter"""
        if isinstance(self.ml_task, str) and self.ml_task == "auto":
            return

        if self.ml_task not in AlgorithmsRegistry.get_supported_ml_tasks():
            raise ValueError(
                f"Expected 'ml_task' to be {' or '.join(AlgorithmsRegistry.get_supported_ml_tasks())}, got '{self.ml_task}''"
            )

    def _validate_results_path(self):
        """Validates path parameter"""
        if self.results_path is None or isinstance(self.results_path, str):
            return

        raise ValueError(
            f"Expected 'results_path' to be of type string, got '{type(self.results_path)}''"
        )

    def _validate_total_time_limit(self):
        """Validates total_time_limit parameter"""
        if self.total_time_limit is None:
            return
        if self.total_time_limit is not None:
            check_greater_than_zero_integer(self.total_time_limit, "total_time_limit")

    def _validate_model_time_limit(self):
        """Validates model_time_limit parameter"""
        if self.model_time_limit is not None:
            check_greater_than_zero_integer(self.model_time_limit, "model_time_limit")

    def _validate_algorithms(self):
        """Validates algorithms parameter"""
        if isinstance(self.algorithms, str) and self.algorithms == "auto":
            return

        for algo in self.algorithms:
            if algo not in list(AlgorithmsRegistry.registry[self._ml_task].keys()):
                raise ValueError(
                    f"The algorithm {algo} is not allowed to use for ML task: {self._ml_task}. Allowed algorithms: {list(AlgorithmsRegistry.registry[self._ml_task].keys())}"
                )

    def _validate_train_ensemble(self):
        """Validates train_ensemble parameter"""
        # `train_ensemble` defaults to True, no further checking required
        check_bool(self.train_ensemble, "train_ensemble")

    def _validate_stack_models(self):
        """Validates stack_models parameter"""
        # `stack_models` defaults to "auto". If "auto" return, else check if is valid bool
        if isinstance(self.stack_models, str) and self.stack_models == "auto":
            return

        check_bool(self.stack_models, "stack_models")

    def _validate_eval_metric(self):
        """Validates eval_metric parameter"""
        if isinstance(self.eval_metric, types.FunctionType):
            return

        if isinstance(self.eval_metric, str) and self.eval_metric == "auto":
            return

        if (self._get_ml_task() == BINARY_CLASSIFICATION) and self.eval_metric not in [
            "logloss",
            "auc",
            "f1",
            "average_precision",
            "accuracy",
        ]:
            raise ValueError(
                f"Metric {self.eval_metric} is not allowed in ML task: {self._get_ml_task()}. \
                    Use 'logloss', 'auc', 'f1', 'average_precision', or 'accuracy'"
            )

        elif (
            self._get_ml_task() == MULTICLASS_CLASSIFICATION
        ) and self.eval_metric not in ["logloss", "f1", "accuracy"]:
            raise ValueError(
                f"Metric {self.eval_metric} is not allowed in ML task: {self._get_ml_task()}. \
                    Use 'logloss', 'f1', or 'accuracy'"
            )

        elif self._get_ml_task() == REGRESSION and self.eval_metric not in [
            "rmse",
            "mse",
            "mae",
            "r2",
            "mape",
            "spearman",
            "pearson",
        ]:
            raise ValueError(
                f"Metric {self.eval_metric} is not allowed in ML task: {self._get_ml_task()}. \
                Use 'rmse', 'mse', 'mae', 'r2', 'mape', 'spearman', or 'pearson'"
            )

    def _validate_validation_strategy(self):
        """Validates validation parameter"""
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
        """Validates verbose parameter"""
        check_positive_integer(self.verbose, "verbose")

    def _validate_explain_level(self):
        """Validates explain_level parameter"""
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
        """Validates golden_features parameter"""
        if isinstance(self.golden_features, str) and self.golden_features == "auto":
            return
        if isinstance(self.golden_features, int):
            return
        check_bool(self.golden_features, "golden_features")

    def _validate_features_selection(self):
        """Validates features_selection parameter"""
        if (
            isinstance(self.features_selection, str)
            and self.features_selection == "auto"
        ):
            return
        check_bool(self.features_selection, "features_selection")

    def _validate_start_random_models(self):
        """Validates start_random_models parameter"""
        if (
            isinstance(self.start_random_models, str)
            and self.start_random_models == "auto"
        ):
            return
        check_greater_than_zero_integer(self.start_random_models, "start_random_models")

    def _validate_hill_climbing_steps(self):
        """Validates hill_climbing_steps parameter"""
        if (
            isinstance(self.hill_climbing_steps, str)
            and self.hill_climbing_steps == "auto"
        ):
            return
        check_positive_integer(self.hill_climbing_steps, "hill_climbing_steps")

    def _validate_top_models_to_improve(self):
        """Validates top_models_to_improve parameter"""
        if (
            isinstance(self.top_models_to_improve, str)
            and self.top_models_to_improve == "auto"
        ):
            return
        check_positive_integer(self.top_models_to_improve, "top_models_to_improve")

    def _validate_boost_on_errors(self):
        """Validates boost_on_errors parameter"""
        if isinstance(self.boost_on_errors, str) and self.boost_on_errors == "auto":
            return
        check_bool(self.boost_on_errors, "boost_on_errors")

    def _validate_kmeans_features(self):
        """Validates kmeans_features parameter"""
        if isinstance(self.kmeans_features, str) and self.kmeans_features == "auto":
            return
        check_bool(self.kmeans_features, "kmeans_features")

    def _validate_mix_encoding(self):
        """Validates mix_encoding parameter"""
        if isinstance(self.mix_encoding, str) and self.mix_encoding == "auto":
            return
        check_bool(self.mix_encoding, "mix_encoding")

    def _validate_max_single_prediction_time(self):
        """Validates max_single_prediction_time parameter"""
        if self.max_single_prediction_time is None:
            return
        check_greater_than_zero_integer_or_float(
            self.max_single_prediction_time, "max_single_prediction_time"
        )

    def _validate_optuna_time_budget(self):
        """Validates optuna_time_budget parameter"""
        if self.optuna_time_budget is None:
            return
        check_greater_than_zero_integer(self.optuna_time_budget, "optuna_time_budget")

    def _validate_optuna_init_params(self):
        """Validates optuna_init_params parameter"""
        if self.optuna_init_params is None:
            return
        if type(self.optuna_init_params) is not dict:
            raise ValueError(
                f"Expected 'optuna_init_params' to be a dict, got '{type(self.optuna_init_params)}'"
            )

    def _validate_optuna_verbose(self):
        """Validates optuna_verbose parameter"""
        if self.optuna_verbose is None:
            return
        check_bool(self.optuna_verbose, "optuna_verbose")

    def _validate_n_jobs(self):
        """Validates mix_encoding parameter"""
        check_integer(self.n_jobs, "n_jobs")

    def _validate_random_state(self):
        """Validates random_state parameter"""
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

    report_style = """
.styled-table {
    border-collapse: collapse;
    font-size: 0.9em;
    font-family:Courier New;
}

.styled-table td, .styled-table th {
    border: 1px solid #ddd;
    padding: 8px;
}

.styled-table tr:nth-child(even){background-color: #f2f2f2;}

.styled-table tr:hover {background-color: #e0ecf5;}

.styled-table thead {
    padding-top: 6px;
    padding-bottom: 6px;
    text-align: left;
    background-color: #0099cc;
    color: white;
}

body {
    font-family: Arial;
    font-size: 1.0em;
    background-color: rgba(236, 243, 249, 0.15);
}

h1 {
    color: #004666;
    border-bottom: 1px solid rgba(0,70,102,0.3)
}
h2 {
    color: #004666;
    padding-bottom: 5px;
    margin-bottom: 0px;
}

ul {
    margin-top: 0px;
}

p {
    margin-top: 5px;
}

h3 {
    color: #004666;
    padding-bottom: 5px;
    margin-bottom: 0px;
}
a {
    font-weight: bold;
    color: #004666;
}

a:hover {
    cursor: pointer;
    color: #0099CC;
}


"""

    def _md_to_html(self, md_fname, page_type, dir_path, me=None):
        import markdown
        import base64

        if not os.path.exists(md_fname):
            return None
        content = ""
        with open(md_fname) as fin:
            content = fin.read()

        content = content.replace("README.md", "README.html")
        content_html = markdown.markdown(
            content, extensions=["markdown.extensions.tables"]
        )
        content_html = content_html.replace("<img ", '<img style="width:750px" ')
        content_html = content_html.replace("<table>", '<table class="styled-table">')
        content_html = content_html.replace("<tr>", '<tr style="text-align: right;">')

        # replace png figures to base64
        for f in os.listdir(dir_path):
            if ".png" in f:
                encoded_string = ""
                with open(os.path.join(dir_path, f), "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                    encoded_string = encoded_string.decode("utf-8")
                encoded_figure = f"data:image/png;base64, {encoded_string}"
                content_html = content_html.replace(f, encoded_figure)

        # insert svg figures
        for f in os.listdir(dir_path):
            if ".svg" in f:
                with open(os.path.join(dir_path, f), "rb") as image_file:
                    svg_plot = image_file.read()
                    svg_plot = svg_plot.decode("utf-8")

                arr = content_html.split("\n")
                new_content = []
                for i in arr:
                    if f in i:
                        new_content += [f"<p>{svg_plot}</p>"]
                    else:
                        new_content += [i]
                content_html = "\n".join(new_content)

        # change links
        if page_type == "main":
            for f in os.listdir(dir_path):
                if os.path.exists(os.path.join(dir_path, f, "README.md")):
                    old = f'href="{f}/README.html"'
                    new = f"onclick=\"toggleShow('{f}');toggleShow('main')\" "
                    content_html = content_html.replace(old, new)

        # other links
        if me is not None:
            old = 'href="../README.html"'
            new = f"onclick=\"toggleShow('{me}');toggleShow('main')\" "
            content_html = content_html.replace(old, new)

        beginning = ""

        if page_type == "main":
            beginning += """<img src="https://raw.githubusercontent.com/mljar/visual-identity/main/media/mljar_AutomatedML.png" style="height:128px; margin-left: auto;
margin-right: auto;display: block;"/>\n\n"""
            if os.path.exists(os.path.join(self._results_path, "EDA")):
                beginning += "<a onclick=\"toggleShow('EDA');toggleShow('main')\" >Automatic Exploratory Data Analysis Report</a>"
            if os.path.exists(os.path.join(self._results_path, "optuna/README.md")):
                beginning += "<h2><a onclick=\"toggleShow('optuna');toggleShow('main')\" >&#187; Optuna Params Tuning Report</a></h2>"

        content_html = beginning + content_html

        return content_html

    def _show_report(self, main_readme_html, width=900, height=1200):
        from IPython.display import HTML, IFrame

        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is None:
            with open(main_readme_html) as fin:
                return HTML(fin.read())
        else:
            return IFrame(main_readme_html, width=width, height=height)

    def _report(self, width=900, height=1200):

        self._results_path = self._get_results_path()
        main_readme_html = os.path.join(self._results_path, "README.html")

        if os.path.exists(main_readme_html):
            return self._show_report(main_readme_html, width, height)

        body = ""
        fname = os.path.join(self._results_path, "README.md")
        body += (
            '<div id="main">\n'
            + self._md_to_html(fname, "main", self._results_path)
            + "\n\n</div>\n\n"
        )

        for f in os.listdir(self._results_path):
            fname = os.path.join(self._results_path, f, "README.md")
            if os.path.exists(fname):
                body += (
                    f'<div id="{f}" style="display: none">\n'
                    + self._md_to_html(
                        fname, "sub", os.path.join(self._results_path, f), f
                    )
                    + "\n\n</div>\n\n"
                )

        body += """
    <script>
        function toggleShow(elementId) {
            var x = document.getElementById(elementId);
            if (x.style.display === "none") {
                x.style.display = "block";
            } else {
                x.style.display = "none";
            }
        }
    </script>
        """

        report_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
    {self.report_style}
    </style>
</head>
<body>
    {body}
</body>
</html>
"""
        with open(main_readme_html, "w") as fout:
            fout.write(report_content)

        return self._show_report(main_readme_html, width, height)

    def _need_retrain(self, X, y, sample_weight, decrease):

        metric = self._best_model.get_metric()

        X, y, sample_weight = ExcludeRowsMissingTarget.transform(
            X, y, sample_weight, warn=True
        )

        if self._ml_task == BINARY_CLASSIFICATION:
            prediction = self._predict_proba(X)[:, 1]
        if self._ml_task == MULTICLASS_CLASSIFICATION:
            prediction = self._predict_proba(X)
        else:
            prediction = self._predict(X)

        sign = -1.0 if Metric.optimize_negative(metric.name) else 1.0

        new_score = metric(y, prediction, sample_weight)
        old_score = self._best_model.get_final_loss()

        change = np.abs((old_score - new_score) / old_score)

        # always minimize the score
        if new_score > old_score:
            self.verbose_print(
                f"Model performance decreased by {np.round(change*100.0,2)}%"
            )
            return change > decrease
        else:
            self.verbose_print(
                f"Model performance increased by {np.round(change*100.0,2)}%"
            )
            return False
