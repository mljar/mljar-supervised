import os
import sys
import json
import copy
import time
import numpy as np
import pandas as pd
import logging
from tabulate import tabulate
from copy import deepcopy

from varname import nameof

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_x

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
from supervised.utils.decorators import (
    check_positive_integer,
    check_greater_than_zero_integer,
    check_bool,
)


logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.ERROR
)
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class AutoML(BaseEstimator):
    """
    Automated Machine Learning for supervised tasks (binary classification, multiclass classification, regression).

    Provides scikit-learn API compatibily

    Parameters
    ----------

    mode : str {"Explain", "Perform", "Compete"}, optional, default="Explain"
        Defines the goal and how intensive the AutoML search will be.
        There are three available modes:
        - `Explain` : To to be used when the user wants to explain and understand the data.
            - Uses 75%/25% train/test split.
            - Uses the following models: `Baseline`, `Linear`, `Decision Tree`, `Random Forest`, `XGBoost`, `Artificial Neural Network`, and `Ensemble`.
            - Has full explanations in reports: learning curves, importance plots, and SHAP plots.
        - `Perform` : To be used when the user wants to train a model that will be used in real-life use cases.
            - Uses 5-fold CV (Cross-Validation).
            - Uses the following models: `Linear`, `Random Forest`, `LightGBM`, `XGBoost`, `CatBoost`, `Artificial Neural Network`, and `Ensemble`.
            - Has learning curves and importance plots in reports.
        - `Compete` : To be used for machine learning competitions (maximum performance).
            - Uses 10-fold CV (Cross-Validation).
            - Uses the following models: `Linear`, `DecisionTree`, `Random Forest`, `Extra Trees`, `XGBoost`, `CatBoost`, `Artificial Neural Network`,
                `Artificial Neural Network`, `Nearest Neighbors`, `Ensemble`, and `Stacking`.
            - It has only learning curves in the reports.

    results_path : str, optional, default=None
        The path with results. If None, then the name of directory will be generated with the template: AutoML_{number},
        where the number can be from 1 to 1,000 - depends which direcory name will be available.
        If the `results_path` will point to directory with AutoML results (`params.json` must be present),
        then all models will be loaded.

    total_time_limit : int or None, optional, default=1800
        The time limit in seconds for AutoML training. If None, then
        `model_time_limit` is not used.

    model_time_limit : int, optional, default=None
        The time limit for training a single model, in seconds.
        If `model_time_limit` is set, the `total_time_limit` is not respected.
        The single model can contain several learners. The time limit for subsequent learners is computed based on `model_time_limit`.
        For example, in the case of 10-fold cross-validation, one model will have 10 learners.
        The `model_time_limit` is the time for all 10 learners.

    algorithms : list of str
        The list of algorithms that will be used in the training. The algorithms can be:
        [
            "Baseline",
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

    tuning_mode :  str {"Normal", "Sport", "Insane", "Perfect"}, default="Normal"
        The mode for tuning. The names are kept the same as `MLjar web application <https://mljar.com>`
        Each mode describe how many models will be checked:
        - `Normal` : about 5-10 models of each algorithm will be trained,
        - `Sport` : about 10-15 models of each algorithm will be trained,
        - `Insane` : about 15-20 models of each algorithm will be trained,
        - `Perfect` : about 25-35 models of each algorithm will be trained.
        You can also set how many models will be trained with `set_advanced` method.

    train_ensemble : bool, default=True
        Whether an ensemble gets created at the end of the training.

    stack_models : bool, default=True
        Whether a models stack gets created at the end of the training. Stack level is 1.

    eval_metric : str, default=None
        The metric to be optimized.
        If None, then:
            - `logloss` is used for classifications taks.
            - `rmse` is used for regression taks.
        .. note:: Still not implemented, please left `None`

    validation : JSON, default=None
        The JSON with validation type. Right now only Cross-Validation is supported.
        Example::

        {"validation_type": "kfold", "k_folds": 5, "shuffle": True, "stratify": True, "random_seed": 123}

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.
        .. note:: Still not implemented, please left `None`

    ml_task : str {"binary_classification", "multiclass_classification", "regression"} or None, default=None
        If left `None` AutoML will try to guess the task based on target values.
        If there will be only 2 values in the target, then task will be set to `"binary_classification"`.
        If number of values in the target will be between 2 and 20 (included), then task will be set to `"multiclass_classification"`.
        In all other casses, the task is set to `"regression"`.

    explain_level : int {0,1,2} or None, default=None
        The level of explanations included to each model:
        - if `explain_level` is `0` no explanations are produced.
        - if `explain_level` is `1` the following explanations are produced: importance plot (with permutation method), for decision trees produce tree plots, for linear models save coefficients.
        - if `explain_level` is `2` the following explanations are produced: the same as `1` plus SHAP explanations.
        If left `None` AutoML will produce explanations based on the selected `mode`.

    random_state : int, default=None
        Controls the randomness of the MLjar Tuner

    #TODO:Complete attribute description
    Attributes
    ----------
    models : list
        The list of built models
    best_model : 'ModelFramework' or 'Ensemble'
        The best model found by AutoML
    fit_time : int
        Duration of `fit()` method
    models_train_time :

    max_metrics :

    confusion_matrix :

    threshold : float
        The threshold used in binary classification
    metrics_details :
        Details about metrics

    X_train_path :

    y_train_path :

    X_validation_path :

    y_validation_path :

    data_info :

    model_paths : list

    stacked_models :

    time_spend :

    time_start :

    all_params :

    _init_params :


    Examples
    --------
    Binary Classification Example:
    >>> import pandas as pd
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import roc_auc_score
    >>> from supervised import AutoML
    >>> df = pd.read_csv(
    ...        "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv",
    ...       skipinitialspace=True
    ...    )
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ... df[df.columns[:-1]], df["income"], test_size=0.25
    ... )
    >>> automl = AutoML()
    >>> automl.fit(X_train, y_train)
    >>> y_pred_prob = automl.predict_proba(X_test)
    >>> print(f"AUROC: {roc_auc_score(y_test, y_pred_prob):.2f}%")

    Multi-Class Classification Example:
    >>> import pandas as pd
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.model_selection import train_test_split
    >>> from supervised import AutoML
    >>> digits = load_digits()
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     pd.DataFrame(digits.data), digits.target, stratify=digits.target, test_size=0.25,
    ...     random_state=123
    ... )
    >>> automl = AutoML(mode="Perform")
    >>> automl.fit(X_train, y_train)
    >>> y_pred = automl.predict(X_test)
    >>> print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}%")

    Regression Example:
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import mean_squared_error
    >>> from supervised import AutoML
    >>> housing = load_boston()
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...       pd.DataFrame(housing.data, columns=housing.feature_names),
    ...       housing.target,
    ...       test_size=0.25,
    ...       random_state=123,
    ... )
    >>> automl = AutoML(mode="Compete")
    >>> automl.fit(X_train, y_train)
    >>> predictions = automl.predict(X_test)
    >>> print("Test MSE:", mean_squared_error(y_test, predictions))

    """

    @property
    def is_fitted(self):
        return self._fit_level == "finished"

    @property
    def tuner_params(self):
        return {
            "start_random_models": self.start_random_models,
            "hill_climbing_steps": self.hill_climbing_steps,
            "top_models_to_improve": self.top_models_to_improve,
        }

    def __init__(
        self,
        mode="Explain",
        ml_task="auto",
        tuning_mode="Normal",
        results_path=None,
        total_time_limit=30 * 60,
        model_time_limit=None,
        algorithms="auto",
        train_ensemble=True,
        stack_models="auto",
        eval_metric="auto",
        validation="auto",
        verbose=0,
        explain_level="auto",
        golden_features="auto",
        feature_selection="auto",
        start_random_models="auto",
        hill_climbing_steps="auto",
        top_models_to_improve="auto",
        random_state=None,
    ):
        self.mode = mode
        self.ml_task = ml_task
        self.tuning_mode = tuning_mode
        self.results_path = results_path
        self.total_time_limit = total_time_limit
        self.model_time_limit = model_time_limit
        self.algorithms = algorithms
        self.train_ensemble = train_ensemble
        self.stack_models = stack_models
        self.eval_metric = eval_metric
        self.validation = validation
        self.verbose = verbose
        self.explain_level = explain_level
        self.golden_features = golden_features
        self.feature_selection = feature_selection
        self.start_random_models = start_random_models
        self.hill_climbing_steps = hill_climbing_steps
        self.top_models_to_improve = top_models_to_improve
        self.random_state = random_state
        logger.debug("AutoML.__init__")

    def set_advanced(
        self, start_random_models=1, hill_climbing_steps=0, top_models_to_improve=0
    ):
        """
        Advanced set of tuning parameters.

        :param start_random_models: Number of not-so-random models to check for each algorithm.
        :param hill_climbing_steps: Number of hill climbing steps during tuning.
        :param top_models_to_improve: Number of top models (of each algorithm) which will be considered for improving in hill climbing steps.
        """
        self.start_random_models = start_random_models
        self.hill_climbing_steps = hill_climbing_steps
        self.top_models_to_improve = top_models_to_improve

    def load(self):
        logger.info("Loading AutoML models ...")
        try:
            params = json.load(open(os.path.join(self.results_path, "params.json")))

            self._model_paths = params["saved"]
            self.ml_task = params["ml_task"]
            self.eval_metric = params["eval_metric"]
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
            with open(os.path.join(self.results_path, "best_model.txt"), "r") as fin:
                best_model_name = fin.read()

            self._best_model = models_map[best_model_name]

            data_info_path = os.path.join(self.results_path, "data_info.json")
            self._data_info = json.load(open(data_info_path))
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
            ldb["metric_type"] += [self.eval_metric]
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
                self.eval_metric,
                np.round(model.get_final_loss(), 6),
                np.round(model.get_train_time(), 2),
            )
        )
        self.log_train_time(model.get_type(), model.get_train_time())

    def _get_learner_time_limit(self, model_type):

        logger.debug(
            f"Fit level: {self._fit_level}, model type: {model_type}. "
            + f"Time spend: {json.dumps(self._time_spend, indent=4)}"
        )

        if self.model_time_limit is not None:
            k = self.validation.get("k_folds", 1.0)
            return self.model_time_limit / k

        if self._fit_level == "simple_algorithms":
            return None
        if self._fit_level == "default_algorithms":
            return None

        tune_algorithms = [
            a
            for a in self.algorithms
            if a not in ["Baseline", "Linear", "Decision Tree", "Nearest Neighbors"]
        ]
        tune_algs_cnt = len(tune_algorithms)
        if tune_algs_cnt == 0:
            return None

        time_elapsed = time.time() - self._start_time
        time_left = self.total_time_limit - time_elapsed

        k_folds = self.validation.get("k_folds", 1.0)

        if self._fit_level == "not_so_random":
            tt = (
                self.total_time_limit
                - self._time_spend["simple_algorithms"]
                - self._time_spend["default_algorithms"]
            )
            if self.stack_models:
                tt *= 0.6  # leave some time for stacking (approx. 40% for stacking of time left)
            tt /= 2.0  # leave some time for hill-climbing
            tt /= tune_algs_cnt  # give time equally for each algorithm
            tt /= k_folds  # time is per learner (per fold)
            return tt

        if self._fit_level == "hill_climbing":
            tt = (
                self.total_time_limit
                - self._time_spend["simple_algorithms"]
                - self._time_spend["default_algorithms"]
                - self._time_spend["not_so_random"]
            )
            if self.stack_models:
                tt *= 0.4  # leave some time for stacking (approx. 60% for stacking of time left)
            tt /= tune_algs_cnt  # give time equally for each algorithm
            tt /= k_folds  # time is per learner (per fold)
            return tt

        if self.stack_models and self._fit_level == "stack":
            tt = time_left
            tt /= tune_algs_cnt  # give time equally for each algorithm
            tt /= k_folds  # time is per learner (per fold)
            return tt

    def create_dir(self, model_path):
        if not os.path.exists(model_path):
            try:
                os.mkdir(model_path)
            except Exception as e:
                raise AutoMLException(f"Cannot create directory {model_path}. {str(e)}")

    def train_model(self, params):

        # do we have enough time to train?
        # if not, skip
        if not self._enough_time_to_train(params["learner"]["model_type"]):
            logger.info(f"Cannot train {params['name']} because of the time constraint")
            return False

        # let's create directory to log all training artifacts
        model_path = os.path.join(self.results_path, params["name"])
        self.create_dir(model_path)

        # prepare callbacks
        early_stop = EarlyStopping(
            {"metric": {"name": self.eval_metric}, "log_to_dir": model_path}
        )

        learner_time_constraint = LearnerTimeConstraint(
            {
                "learner_time_limit": self._get_learner_time_limit(
                    params["learner"]["model_type"]
                ),
                "min_steps": params["additional"].get("min_steps"),
            }
        )

        total_time_constraint = TotalTimeConstraint(
            {
                "total_time_limit": self.total_time_limit
                if self.model_time_limit is None
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
        if self.verbose != 0:
            # self.progress_bar.write(msg)
            print(msg)

    def log_train_time(self, model_type, train_time):
        if model_type in self._models_train_time:
            self._models_train_time[model_type] += [train_time]
        else:
            self._models_train_time[model_type] = [train_time]

    def _enough_time_to_train(self, model_type):
        # if model_time_limit is set, train every model
        # do not apply total_time_limit
        if self.model_time_limit is not None:
            return True
        # no total time limit, just train, dont ask
        if self.total_time_limit is None:
            return True

        total_time_spend = time.time() - self._start_time
        # no time left, do not train more models, sorry ...
        time_left = self.total_time_limit - total_time_spend
        if time_left < 0:
            return False

        # there is still time and model_type was not tested yet
        # we should try it
        if time_left > 0 and model_type not in self._models_train_time:
            return True

        # check the fit level type
        # we dont want to spend too much time on one level

        if self._fit_level == "not_so_random":

            time_should_use = (
                self.total_time_limit
                - self._time_spend["simple_algorithms"]
                - self._time_spend["default_algorithms"]
            )
            if self.stack_models:
                time_should_use *= 0.6  # leave time for stacking
            if self.hill_climbing_steps > 0:
                time_should_use /= 2.0  # leave time for hill-climbing

            if (
                total_time_spend
                > time_should_use
                + self._time_spend["simple_algorithms"]
                + self._time_spend["default_algorithms"]
            ):
                return False

        ##################
        # hill climbing check

        if self._fit_level is not None and "hill_climbing" in self._fit_level:

            time_should_use = (
                self.total_time_limit
                - self._time_spend["simple_algorithms"]
                - self._time_spend["default_algorithms"]
                - self._time_spend["not_so_random"]
            )
            if self.stack_models:
                time_should_use *= 0.4  # leave time for stacking

            if (
                total_time_spend
                > time_should_use
                + self._time_spend["simple_algorithms"]
                + self._time_spend["default_algorithms"]
                + self._time_spend["not_so_random"]
            ):
                return False

        model_total_time_spend = (
            0
            if model_type not in self._models_train_time
            else np.sum(self._models_train_time[model_type])
        )
        model_mean_time_spend = (
            0
            if model_type not in self._models_train_time
            else np.mean(self._models_train_time[model_type])
        )

        algo_cnt = float(len(self.algorithms))
        for a in ["Baseline", "Decision Tree", "Linear", "Nearest Neighbors"]:
            if a in self.algorithms:
                algo_cnt -= 1.0
        if algo_cnt < 1.0:
            algo_cnt = 1.0

        model_time_left = time_left / algo_cnt
        if model_mean_time_spend <= model_time_left:
            return True

        return False

    def ensemble_step(self, is_stacked=False):
        if self.train_ensemble and len(self._models) > 1:

            ensemble_path = os.path.join(
                self.results_path, "Ensemble_Stacked" if is_stacked else "Ensemble"
            )
            self.create_dir(ensemble_path)

            self.ensemble = Ensemble(
                self.eval_metric, self.ml_task, is_stacked=is_stacked
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
                if self.ml_task == BINARY_CLASSIFICATION:
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

    def stack_models(self):

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
        if self.total_time_limit is not None:
            time_left = self.total_time_limit - (time.time() - self._start_time)
            # we need at least 60 seconds to do anything
            if time_left < 60:
                return

        self.stack_models()

        X_train_stacked_path = os.path.join(
            self.results_path, "X_train_stacked.parquet"
        )
        if os.path.exists(X_train_stacked_path):
            return

        X = pd.read_parquet(self._X_train_path)
        org_columns = X.columns.tolist()
        X_stacked = self.get_stacked_data(X)
        new_columns = X_stacked.columns.tolist()
        added_columns = [c for c in new_columns if c not in org_columns]

        # save stacked train data
        X_stacked.to_parquet(X_train_stacked_path, index=False)

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
                        params["preprocessing"]["columns_preprocessing"][col] = [scale]
            self.train_model(params)
        """

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

    def _build_dataframe(self, X, y=None, X_validation=None, y_validation=None):
        # If Inputs are not pandas dataframes use scikit-learn validation for X and y
        if not isinstance(X, pd.DataFrame):
            X_train, y_train = check_X_y(X_train, y_train)
            # Create Pandas dataframe from np.arrays, columns get names with the schema: feature_{index}
            X_train = pd.DataFrame(
                X_train,
                columns=["feature_" + str(i) for i in range(1, len(X_train[1]) + 1)],
            )

        # Enforce X_train columns to be string
        X_train.columns = X_train.columns.astype(str)

        X_train.reset_index(drop=True, inplace=True)

        if not isinstance(y_train, pd.DataFrame):
            _, y_train = check_X_y(
                X_train, y_train
            )  # use of discard '_' is important, otherwise it will invalidate previous operations, and lead AutoML to crash
            y_train = pd.DataFrame(y_train, columns=["target"])
        else:
            # Check if target is only 1 column
            if y_train.columns != 1:
                raise AutoMLException(
                    f"Expected y_train to have 1 column, got {y_train.columns}."
                )

        X_train, y_train = ExcludeRowsMissingTarget.transform(
            X_train, y_train, warn=True
        )

        return X_train, y_train, X_validation, y_validation

    def _save_data(self, X_train, y_train, X_validation=None, y_validation=None):

        self.X_train_path = os.path.join(self.results_path, "X_train.parquet")
        self.y_train_path = os.path.join(self.results_path, "y_train.parquet")

        X_train.to_parquet(self.X_train_path, index=False)

        if self.ml_task == MULTICLASS_CLASSIFICATION:
            y_train = y_train.astype(str)

        y_train.to_parquet(self.y_train_path, index=False)

        self.validation["X_train_path"] = self.X_train_path
        self.validation["y_train_path"] = self.y_train_path
        self.validation["results_path"] = self.results_path

        columns_and_target_info = DataInfo.compute(X_train, y_train, self.ml_task)

        self.data_info = {
            "columns": X_train.columns.tolist(),
            "rows": X_train.shape[0],
            "cols": X_train.shape[1],
            "target_is_numeric": pd.api.types.is_numeric_dtype(y_train),
            "columns_info": columns_and_target_info["columns_info"],
            "target_info": columns_and_target_info["target_info"],
        }
        if columns_and_target_info.get("num_class") is not None:
            self.data_info["num_class"] = columns_and_target_info["num_class"]
        data_info_path = os.path.join(self.results_path, "data_info.json")
        with open(data_info_path, "w") as fout:
            fout.write(json.dumps(self.data_info, indent=4))

        self._drop_data_variables(X_train)

    def _drop_data_variables(self, X_train):

        X_train.drop(X_train.columns, axis=1, inplace=True)

    def _load_data_variables(self, X_train):
        if X_train.shape[1] == 0:
            X = pd.read_parquet(self.X_train_path)
            for c in X.columns:
                X_train.insert(loc=X_train.shape[1], column=c, value=X[c])

        os.remove(self.X_train_path)
        os.remove(self.y_train_path)

    def save_progress(self, step=None, generated_params=None):

        if step is not None and generated_params is not None:
            self._all_params[step] = generated_params

        state = {}

        state["fit_level"] = self._fit_level
        state["time_spend"] = self._time_spend
        state["all_params"] = self._all_params

        fname = os.path.join(self.results_path, "progress.json")
        with open(fname, "w") as fout:
            fout.write(json.dumps(state, indent=4))

    def load_progress(self):
        state = {}
        fname = os.path.join(self.results_path, "progress.json")
        if not os.path.exists(fname):
            return
        state = json.load(open(fname, "r"))
        self.fit_level = state.get("fit_level", self.fit_level)
        self.time_spend = state.get("time_spend", self.time_spend)
        self.all_params = state.get("all_params", self.all_params)

    def _validate_model(self):
        # All the necessary validation to params provided in __init__

        # Validate `mode`
        self._set_mode()

        # Validate `ml_task` -> this needs to be on top, because some of the methods below depend on task type
        self._set_ml_task()

        # Validate `tuning_mode`
        self._set_tuning_mode()

        # Validate `results_path`
        self._set_results_path()

        # Validate `total_time_limit`
        self._set_total_time_limit()

        # Validate `model_time_limit`
        self._set_model_time_limit()

        # Validate `algorithms`
        self._set_algorithms()

        # Validate `train_ensemble`
        self._set_train_ensemble()

        # Validate `stack_models`
        self._set_stack_models()

        # Validate metrics
        self._set_eval_metric()

        # Validate `validation`
        self._set_validation()

        # Validate `verbose`
        self._set_verbose()

        # Validate `explain_level`
        self._set_explain_level()

        # Validate `golden_features`
        self._set_golden_features()

        # Validate `feature_selection`
        self._set_feature_selection()

        # Validate `start_random_models`
        self._set_start_random_models()

        # Validate `hill_climbing_steps`
        self._set_hill_climbing_steps()

        # Validate `top_models_to_improve`
        self._set_top_models_to_improve()

        # Validate `random_state`
        self._set_random_state()

    def fit(self, X_train, y_train, X_validation=None, y_validation=None):
        """
        Fit the AutoML model.
        Parameters
        ----------
        X_train : list or numpy.ndarray or pandas.DataFrame
            Training data
        y_train : list or numpy.ndarray or pandas.DataFrame
            Training targets
        X_validation : list or numpy.ndarray or pandas.DataFrame
            Validation data
        y_validation : list or numpy.ndarray or pandas.DataFrame
            Targets for validation data
        """
        # Validate input and build dataframes
        X_train, y_train, X_validation, y_validation = self._build_train_dataframe(
            X_train, y_train, X_validation, y_validation
        )
        # Set the classes_ attribute
        self.classes_, y = np.unique(np.array(y_train), return_inverse=True)

        # Validate model
        self._validate_model()

        # Create needed attributes
        self._best_model = None
        self._fit_time = None
        self._models_train_time = {}
        self._threshold = None
        self._metrics_details = None
        self._max_metrics = None
        self._confusion_matrix = None
        self._models = []
        self._X_train_path, self._y_train_path = None, None
        self._X_validation_path, self._y_validation_path = None, None
        self._data_info = None
        self._model_paths = []
        self._stacked_models = None
        self._fit_level = None
        self._time_spend = {}
        self._time_start = {}
        self._start_time = time.time()  # it will be updated in `fit` method
        self._all_params = {}
        ##########################
        try:

            self.load_progress()

            self._start_time = time.time() - np.sum(list(self._time_spend.values()))

            ## EDA
            if self.explain_level == 2:

                os.mkdir(os.path.join(self.results_path, "EDA"))
                eda_path = os.path.join(self.results_path, "EDA")

                EDA.compute(X_train, y_train, eda_path)

            self._set_ml_task()

            # Useless lines?
            # if X_train is not None:
            #     X_train = X_train.copy()

            self._save_data(X_train, y_train, X_validation, y_validation)

            # if self.ml_task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:
            #     self._check_imbalanced(y_train)

            tuner = MljarTuner(
                self.tuner_params,
                self.algorithms,
                self.ml_task,
                self.validation,
                self.explain_level,
                self.data_info,
                self.golden_features,
                self.feature_selection,
                self.train_ensemble,
                self.stack_models,
                self.random_state,
            )
            self.tuner = tuner

            steps = tuner.steps()

            for step in steps:
                self._fit_level = step
                start = time.time()
                self._time_start[step] = start

                if step == "stack":
                    self.prepare_for_stacking()

                generated_params = []
                if step in self._all_params:
                    generated_params = self._all_params[step]
                else:
                    generated_params = tuner.generate_params(
                        step, self._models, self.results_path, self._stacked_models
                    )
                if generated_params is None:
                    continue
                if generated_params:
                    print("-" * 72)
                    print(f"{step} with {len(generated_params)} models to train ...")

                for params in generated_params:
                    if params.get("status", "") == "trained":
                        print(f"Skipping {params['name']}, already trained.")
                        continue
                    if params.get("status", "") == "skipped":
                        print(f"Skipped {params['name']}.")
                        continue

                    trained = False
                    if "ensemble" in step:
                        trained = self.ensemble_step(is_stacked=params["is_stacked"])
                    else:
                        trained = self.train_model(params)

                    params["status"] = "trained" if trained else "skipped"
                    params["final_loss"] = self._models[-1].get_final_loss()
                    params["train_time"] = self._models[-1].get_train_time()
                    self.save_progress(step, generated_params)

                self._time_spend[step] = np.round(
                    time.time() - start, 2
                ) + self._time_spend.get(step, 0)

            self._fit_time = time.time() - self._start_time

            self._fit_level = "finished"
            self.save_progress()

            logger.info(f"AutoML fit time: {self._fit_time}")

        except Exception as e:
            raise e
        finally:
            if self._X_train_path is not None:
                self._load_data_variables(X_train)

        return self

    def select_and_save_best(self):
        max_loss = 10e14
        for i, m in enumerate(self._models):
            if m.get_final_loss() < max_loss:
                self.best_model = m
                max_loss = m.get_final_loss()

        with open(os.path.join(self.results_path, "best_model.txt"), "w") as fout:
            fout.write(f"{self.best_model.get_name()}")

        with open(os.path.join(self.results_path, "params.json"), "w") as fout:
            params = {
                "ml_task": self.ml_task,
                "eval_metric": self.eval_metric,
                "saved": self._model_paths,
            }
            if self._stacked_models is not None:
                params["stacked"] = [m.get_name() for m in self.stacked_models]
            fout.write(json.dumps(params, indent=4))

        ldb = self.get_leaderboard()
        ldb.to_csv(os.path.join(self.results_path, "leaderboard.csv"), index=False)

        # save report
        ldb["Link"] = [f"[Results link]({m}/README.md)" for m in ldb["name"].values]
        ldb.insert(loc=0, column="Best model", value="")
        ldb.loc[ldb.name == self.best_model.get_name(), "Best model"] = "**the best**"

        with open(os.path.join(self.results_path, "README.md"), "w") as fout:
            fout.write(f"# AutoML Leaderboard\n\n")
            fout.write(tabulate(ldb.values, ldb.columns, tablefmt="pipe"))
            LeaderboardPlots.compute(ldb, self.results_path, fout)

    def _base_predict(self, X):

        if not isinstance(X.columns[0], str):
            X.columns = [str(c) for c in X.columns]

        input_columns = X.columns.tolist()
        for column in self.data_info["columns"]:
            if column not in input_columns:
                raise AutoMLException(
                    f"Missing column: {column} in input data. Cannot predict"
                )
        X = X[self.data_info["columns"]]

        # is stacked model
        if self.best_model._is_stacked:
            self.stack_models()
            X_stacked = self.get_stacked_data(X, mode="predict")

            if self.best_model.get_type() == "Ensemble":
                # Ensemble is using both original and stacked data
                predictions = self.best_model.predict(X, X_stacked)
            else:
                predictions = self.best_model.predict(X_stacked)
        else:
            predictions = self.best_model.predict(X)

        if self.ml_task == BINARY_CLASSIFICATION:
            # need to predict the label based on predictions and threshold
            neg_label, pos_label = (
                predictions.columns[0][11:],
                predictions.columns[1][11:],
            )

            if neg_label == "0" and pos_label == "1":
                neg_label, pos_label = 0, 1
            target_is_numeric = self.data_info.get("target_is_numeric", False)
            if target_is_numeric:
                neg_label = int(neg_label)
                pos_label = int(pos_label)
            # assume that it is binary classification
            predictions["label"] = (
                predictions.iloc[:, 1] > self.best_model._threshold
            ).astype(np.int32)
            return predictions
        elif self.ml_task == MULTICLASS_CLASSIFICATION:
            target_is_numeric = self.data_info.get("target_is_numeric", False)
            if target_is_numeric:
                predictions["label"] = predictions["label"].astype(np.int32)
            return predictions
        # Regression
        else:
            return predictions

    # @fit_required
    def predict(self, X):
        """
        Computes predictions from AutoML best model.
        Parameters
        ----------
        X : pandas.DataFrame
            The Pandas DataFrame with input data. The input data should have the same columns as data used for training, otherwise the `AutoMLException` will be raised.
        Returns
        -------
        predictions : numpy.ndarray
            One-dimensional array of class label for each object.
        Raises
        ------
        AutoMLException
            The input data doesn't have the same columns as data used for training.
            Model has not yet been fitted.
        """

        predictions = self._base_predict(X)

        # Return predictions
        # If classification task the result is in column 'label'
        # If regression task the result is in column 'prediction'
        return (
            predictions["label"].to_numpy()
            if self.ml_task != REGRESSION
            else predictions["prediction"].to_numpy()
        )

    # @fit_required
    def predict_proba(self, X):
        """
        Computes class probabilities from AutoML best model. This method can only be used for classification ML task.

        Parameters
        ----------
        X : pandas.DataFrame
            The Pandas DataFrame with input data. The input data should have the same columns as data used for training, otherwise the `AutoMLException` will be raised.

        Returns
        -------
        predictions : numpy.ndarray of shape (n_samples, n_classes)
            Matrix of containing class probabilities of the input samples

        Raises
        ------
        AutoMLException
            The input data doesn't have the same columns as data used for training.
            Model has not yet been fitted.

        """
        # Check is task type is correct
        if self.ml_task == REGRESSION:
            raise AutoMLException(
                f"Method `predict_proba()` can only be used when in classification tasks. Current task: '{self.ml_task}'."
            )

        # Make and return predictions
        # If classification task the result is in column 'label'
        # If regression task the result is in column 'prediction'
        # Need to drop `label` column because in case of multilabel classification,
        # the pandas dataframe returned by `predict()` already contains the predicted label.
        # Must pass `errors="ignore"` to pandas `drop()` method because in case of binary
        # classification the label column does not exist when `predict()` is called. This
        # parameter simulates a drop if column exists behavior.
        return self._base_predict(X).drop(["label"], axis=1, errors="ignore").to_numpy()

    # def get_params(self, deep=True):
    #     """
    #     Get parameters from AutoML model.

    #     Parameters
    #     ----------
    #     deep : bool, default=True
    #         If True, will return the parameters for this estimator and
    #         contained subobjects that are estimators.
    #     Returns
    #     -------
    #     params : dict
    #         Dictionary of {param_key: param_value}
    #     """
    #     params = self._init_params.copy()
    #     if deep:
    #         return deepcopy(params)
    #     else:
    #         return params

    # @no_fit_required
    # def set_params(self, **params):
    #     """
    #     Get parameters for this estimator.

    #     Parameters
    #     ----------
    #     deep : bool, default=True
    #         If True, will return the parameters for this estimator and
    #         contained subobjects that are estimators.
    #     Returns
    #     -------
    #     params : dict
    #         Dictionary of {param_key: param_value}
    #     """
    #     for key, value in params.items():
    #         self._init_params[key] = value

    #     return self

    def to_json(self):
        if self.best_model is None:
            return None

        return {
            "best_model": self.best_model.to_json(),
            "threshold": self.threshold,
            "ml_task": self.ml_task,
        }

    def from_json(self, json_data):

        if json_data["best_model"]["algorithm_short_name"] == "Ensemble":
            self.best_model = Ensemble()
            self.best_model.from_json(json_data["best_model"])
        else:
            self.best_model = ModelFramework(json_data["best_model"].get("params"))
            self.best_model.from_json(json_data["best_model"])
        self.threshold = json_data.get("threshold")

        self.ml_task = json_data.get("ml_task")

    def _set_mode(self):
        valid_modes = ["Explain", "Perform", "Compete"]
        if self.mode not in valid_modes:
            raise ValueError(
                f"Expected `{nameof(self.mode)}` to be {' or '.join(valid_modes)}, got '{self.mode}'"
            )

    def _set_ml_task(self):
        # if not set, guess
        if self.ml_task == "auto":
            classes_number = len(self.classes_)
            if classes_number == 2:
                self.ml_task = BINARY_CLASSIFICATION
            elif classes_number <= 20:
                self.ml_task = MULTICLASS_CLASSIFICATION
            else:
                self.ml_task = REGRESSION
        # validation
        if self.ml_task not in AlgorithmsRegistry.get_supported_ml_tasks():
            raise Exception(
                "Unknow Machine Learning task '{}'."
                " Supported tasks are: {}".format(
                    self.ml_task, AlgorithmsRegistry.get_supported_ml_tasks()
                )
            )
        if self.ml_task == REGRESSION:
            if "stratify" in self.validation:
                del self.validation["stratify"]
        logger.info("AutoML task to be solved: {}".format(self.ml_task))
        print(f"AutoML task to be solved: { self.ml_task}")

        # Set '_estimator_type' for sklearn api
        if (
            self.ml_task == BINARY_CLASSIFICATION
            or self.ml_task == MULTICLASS_CLASSIFICATION
        ):
            self._estimator_type = "classifier"
        else:
            self._estimator_type = "regressor"

    def _set_tuning_mode(self):
        # Validate tuning modes
        valid_tuning_modes = ["Normal", "Sport", "Insane", "Perfect"]
        if self.tuning_mode not in valid_tuning_modes:
            raise ValueError(
                f"Expected `{nameof(self.tuning_mode)}` to be {'or'.join(valid_tuning_modes)}, got {self.tuning_mode}"
            )

    def _set_results_path(self):
        if self.results_path is None:
            found = False
            for i in range(1, 10001):
                self.results_path = f"AutoML_{i}"
                if not os.path.exists(self.results_path):
                    found = True
                    break
            if not found:
                raise AutoMLException("Cannot create directory for AutoML results")

        if os.path.exists(self.results_path) and os.path.exists(
            os.path.join(self.results_path, "params.json")
        ):
            print(f"Directory {self.results_path} already exists")
            self.load()
        elif self.results_path is not None:

            if not os.path.exists(self.results_path):
                print(f"Create directory {self.results_path}")
                try:
                    os.mkdir(self.results_path)
                except Exception as e:
                    raise AutoMLException(
                        f"Cannot create directory {self.results_path}"
                    )
            elif os.path.exists(self.results_path) and len(
                os.listdir(self.results_path)
            ):
                raise AutoMLException(
                    f"Cannot set directory for AutoML. Directory {self.results_path} is not empty."
                )
        else:
            raise AutoMLException("Cannot set directory for AutoML results")

    def _set_total_time_limit(self):
        check_greater_than_zero_integer(
            self.total_time_limit, nameof(self.total_time_limit)
        )

    def _set_model_time_limit(self):
        if self.model_time_limit is not None:
            check_greater_than_zero_integer(
                self.model_time_limit, nameof(self.model_time_limit)
            )

    def _set_algorithms(self):
        if self.algorithms == "auto":
            if self.mode == "Explain":
                self.algorithms = [
                    "Baseline",
                    "Linear",
                    "Decision Tree",
                    "Random Forest",
                    "Xgboost",
                    "Neural Network",
                ]
            if self.mode == "Perform":
                self.algorithms = [
                    "Linear",
                    "Random Forest",
                    "LightGBM",
                    "Xgboost",
                    "CatBoost",
                    "Neural Network",
                ]
            if self.mode == "Compete":
                self.algorithms = [
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
        print(self.ml_task)
        for a in self.algorithms:
            if a not in list(AlgorithmsRegistry.registry[self.ml_task].keys()):
                raise AutoMLException(
                    f"The algorithm {a} is not allowed to use for ML task: {self.ml_task}. Allowed algorithms: {list(AlgorithmsRegistry.registry[self.ml_task].keys())}"
                )
        logger.info(f"AutoML will use algorithms: {self.algorithms}")
        print(f"AutoML will use algorithms: {self.algorithms}")

    def _set_train_ensemble(self):
        # `train_ensemble` defaults to True, no further checking required
        check_bool(self.train_ensemble, nameof(self.train_ensemble))

    def _set_stack_models(self):
        # `stack_models` defaults to auto, which only enables it on `Compete` mode
        if self.stack_models == "auto":
            self.stack_models = True if self.mode == "Compete" else False
        else:  # if not auto, check if input is valid bool
            check_bool(self.stack_models, nameof(self.stack_models))

    def _set_eval_metric(self):
        """ Set and validate the metric to be optimized. """
        if self.ml_task == BINARY_CLASSIFICATION:
            if self.eval_metric == "auto":
                self.eval_metric = "logloss"
            elif self.eval_metric not in ["logloss", "auc"]:
                raise AutoMLException(
                    "Metric {} is not allowed in ML task: {}".format(
                        self.eval_metric, self.ml_task
                    )
                )

        elif self.ml_task == MULTICLASS_CLASSIFICATION:
            if self.eval_metric == "auto":
                self.eval_metric = "logloss"
            elif self.eval_metric not in ["logloss"]:
                raise AutoMLException(
                    "Metric {} is not allowed in ML task: {}".format(
                        self.eval_metric, self.ml_task
                    )
                )
        elif self.ml_task == REGRESSION:
            if self.eval_metric == "auto":
                self.eval_metric = "rmse"
            elif self.eval_metric not in ["rmse"]:
                raise AutoMLException(
                    "Metric {} is not allowed in ML task: {}".format(
                        self.eval_metric, self.ml_task
                    )
                )

        msg = f"AutoML will optimize for metric: {self.eval_metric}"
        logger.info(msg)
        print(msg)

    def _set_validation(self):
        if self.validation == "auto":
            if self.mode == "Explain":
                self.validation = {
                    "validation_type": "split",
                    "train_ratio": 0.75,
                    "shuffle": True,
                    "stratify": True,
                }
        elif self.mode == "Perform":
            self.validation = {
                "validation_type": "kfold",
                "k_folds": 5,
                "shuffle": True,
                "stratify": True,
            }
        elif self.mode == "Compete":
            self.validation = {
                "validation_type": "kfold",
                "k_folds": 10,
                "shuffle": True,
                "stratify": True,
            }

        if self.validation["validation_type"] != "kfold" and self.stack_models:
            raise AutoMLException(
                "AutoML cannot stack models. Please set validation to 'kfold' to stack models."
            )
        # TODO: Implement further validations of validation

    def _set_verbose(self):
        check_positive_integer(self.verbose, nameof(self.verbose))

    def _set_explain_level(self):
        valid_explain_levels = [0, 1, 2]
        if self.explain_level == "auto":
            if self.mode == "Explain":
                self.explain_level = 2
            if self.mode == "Perform":
                self.explain_level = 1
            if self.mode == "Compete":
                self.explain_level = 0
        else:
            # Check if explain level is 0 or greater integer
            if not (
                isinstance(self.explain_level, int)
                and self.explain_level in valid_explain_levels
            ):
                raise ValueError(
                    f"Expected `{nameof(self.explain_level)}` to be {' or '.join([str(x) for x in valid_explain_levels])}, got '{self.explain_level}'"
                )

    def _set_golden_features(self):
        if self.golden_features == "auto":
            if self.mode == "Explain":
                self.golden_features = False
            if self.mode == "Perform":
                self.golden_features = True
            if self.mode == "Compete":
                self.golden_features = True
        else:
            check_bool(self.golden_features, nameof(self.golden_features))

    def _set_feature_selection(self):
        if self.feature_selection == "auto":
            if self.mode == "Explain":
                self.feature_selection = False
            if self.mode == "Perform":
                self.feature_selection = True
            if self.mode == "Compete":
                self.feature_selection = True
        else:
            check_bool(self.feature_selection, nameof(self.feature_selection))

    def _set_start_random_models(self):
        if self.start_random_models == "auto":
            if self.mode == "Explain":
                self.start_random_models = 1
            if self.mode == "Perform":
                self.start_random_models = 5
            if self.mode == "Compete":
                self.start_random_models = 10
        else:
            check_greater_than_zero_integer(
                self.start_random_models, nameof(self.start_random_models)
            )

    def _set_hill_climbing_steps(self):
        if self.hill_climbing_steps == "auto":
            if self.mode == "Explain":
                self.hill_climbing_steps = 0
            if self.mode == "Perform":
                self.hill_climbing_steps = 2
            if self.mode == "Compete":
                self.hill_climbing_steps = 2
        else:
            check_greater_than_zero_integer(
                self.hill_climbing_steps, nameof(self.hill_climbing_steps)
            )

    def _set_top_models_to_improve(self):
        if self.top_models_to_improve == "auto":
            if self.mode == "Explain":
                self.top_models_to_improve = 0
            if self.mode == "Perform":
                self.top_models_to_improve = 2
            if self.mode == "Compete":
                self.top_models_to_improve = 3
        else:
            check_greater_than_zero_integer(
                self.top_models_to_improve, nameof(self.top_models_to_improve)
            )

    def _set_random_state(self):
        if self.random_state is not None:
            check_positive_integer(self.random_state, nameof(self.random_state))