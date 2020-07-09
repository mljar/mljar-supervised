import os
import sys
import json
import copy
import time
import numpy as np
import pandas as pd
import logging
from tabulate import tabulate

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

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.ERROR
)
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class AutoML:
    """
    Automated Machine Learning for supervised tasks (binary classification, multiclass classification, regression).
    """

    def __init__(
        self, results_path=None, total_time_limit=30 * 60, mode="Explain", **kwargs
    ):

        """
        Initialize the AutoML object. 
        
        :param results_path: The path with results. 
        If left `None` then the name of directory will be generated, with template: AutoML_{number},
        where the number can be from 1 to 1,000 - depends which direcory name will be available.

        If the `results_path` will point to directory with AutoML results (`params.json` must be present), 
        then all models will be loaded.
        
        :param total_time_limit: The time limit in seconds for AutoML training. 
        It is not used when `model_time_limit` is not `None`.
        
        :params mode

        ## Additional (optional parameters)

        :param model_time_limit: The time limit for training a single model, in seconds. 
        If `model_time_limit` is set, the `total_time_limit` is not respected. 
        The single model can contain several learners.
        The time limit for single learner is computed based on `model_time_limit`.

        For example, in the case of 10-fold cross-validation, one model will have 10 learners. 
        The `model_time_limit` is time for all 10 learners.

        :param algorithms: The list of algorithms that will be used in the training. The algorithms can be:
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
        
        :param tuning_mode: The mode for tuning. It can be: `Normal`, `Sport`, `Insane`, `Perfect`. The names are kept the same as in https://mljar.com application.
        
        Each mode describe how many models will be checked:
        
        - `Normal` - about 5-10 models of each algorithm will be trained,
        - `Sport` - about 10-15 models of each algorithm will be trained,
        - `Insane` - about 15-20 models of each algorithm will be trained,
        - `Perfect` - about 25-35 models of each algorithm will be trained.
        
        You can also set how many models will be trained with `set_advanced` method.
        
        :param train_ensemble: If true then at the end of models training the ensemble will be created. (Default is `True`)

        :param stack_models: If true then stacked models will be created. Stack level is 1. (Default is `True`)
        
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
        
        :param explain_level: The level of explanations included to each model.
        `explain_level = 0` means no explanations
        `explain_level = 1` means produce importance plot (with permutation method), for decision trees produce tree plots, for linear models save coefficients
        `explain_level = 2` the same as for `1` plus SHAP explanations
        
        :param seed: The seed for random generator.

        """
        logger.debug("AutoML.__init__")

        self._results_path = results_path
        """
        total_time_limit is the time for computing for all models
        model_time_limit is the time for computing a single model
        if model_time_limit is None then its value is computed from total_time_limit
        if total_time_limit is set and model_time_limit is set, then total_time_limit constraint will be omitted
        """
        self._total_time_limit = total_time_limit

        if mode not in ["Explain", "Perform", "Compete"]:
            print("mode should be: Explain, Perform or Compete")
            print("Setting mode=Explain")
            mode = "Explain"

        self._mode = mode

        if self._mode == "Explain":
            self._train_ensemble = True
            self._stack_models = False
            self._validation = {
                "validation_type": "split",
                "train_ratio": 0.75,
                "shuffle": True,
                "stratify": True,
            }
            self._algorithms = [
                "Baseline",
                "Linear",
                "Decision Tree",
                "Random Forest",
                "Xgboost",
                "Neural Network",
            ]
            self._explain_level = 2
            self._start_random_models = 1
            self._hill_climbing_steps = 0
            self._top_models_to_improve = 0
        elif self._mode == "Perform":
            self._train_ensemble = True
            self._stack_models = False
            self._validation = {
                "validation_type": "kfold",
                "k_folds": 5,
                "shuffle": True,
                "stratify": True,
            }
            self._algorithms = [
                "Linear",
                "Random Forest",
                "LightGBM",
                "Xgboost",
                "CatBoost",
                "Neural Network",
            ]
            self._explain_level = 1
            self._start_random_models = 5
            self._hill_climbing_steps = 2
            self._top_models_to_improve = 2
        elif self._mode == "Compete":
            self._train_ensemble = True
            self._stack_models = True
            self._validation = {
                "validation_type": "kfold",
                "k_folds": 10,
                "shuffle": True,
                "stratify": True,
            }
            self._algorithms = [
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
            self._explain_level = 0
            self._start_random_models = 10
            self._hill_climbing_steps = 2
            self._top_models_to_improve = 3

        self._model_time_limit = None
        if "model_time_limit" in kwargs:
            self._model_time_limit = kwargs["model_time_limit"]

        # algorithms are checked during the fit method call,
        # we need to know the data to tell if allgorithm is proper
        if "algorithms" in kwargs:
            self._algorithms = kwargs["algorithms"]

        if "validation" in kwargs:
            self._validation = kwargs["validation"]

        if "train_ensemble" in kwargs:
            self._train_ensemble = kwargs["train_ensemble"]

        if "stack_models" in kwargs:
            self._stack_models = kwargs["stack_models"]

        if "explain_level" in kwargs:
            self._explain_level = kwargs["explain_level"]

        self._ml_task = None
        if "ml_task" in kwargs:
            self._ml_task = kwargs["ml_task"]

        self._user_set_optimize_metric = None
        if "optimize_metric" in kwargs:
            self._user_set_optimize_metric = kwargs["optimize_metric"]

        if "tuning_mode" in kwargs:
            self.set_tuning_mode(kwargs["tuning_mode"])

        self._verbose = True
        if "verbose" is kwargs:
            self._verbose = kwargs["verbose"]

        self._seed = 1234
        if "seed" in kwargs:
            self._seed = kwargs["seed"]

        self._tuner_params = {
            "start_random_models": self._start_random_models,
            "hill_climbing_steps": self._hill_climbing_steps,
            "top_models_to_improve": self._top_models_to_improve,
        }
        self._models = []  # instances of iterative learner framework or ensemble

        # it is instance of model framework or ensemble
        self._best_model = None
        self._verbose = True

        self._fit_time = None
        self._models_train_time = {}
        self._threshold = None
        self._metrics_details = None
        self._max_metrics = None
        self._confusion_matrix = None

        self._X_train_path, self._y_train_path = None, None
        self._X_validation_path, self._y_validation_path = None, None

        self._data_info = None
        self._model_paths = []
        self._stacked_models = None

        self._fit_level = None
        self._time_spend = {}
        self._start_time = time.time()  # it will be updated in `fit` method

        if self._validation["validation_type"] != "kfold" and self._stack_models:
            print(
                "Models cannot be stacked. Please set validation to k-fold to stack models."
            )
            # stacking only available of k-fold validation
            self._stack_models = False

        # this should be last in the constrcutor
        # in case there is a dir, it might load models
        self._set_results_dir()

    def set_tuning_mode(self, mode="Normal"):
        if mode == "Sport":
            self._start_random_models = 10
            self._hill_climbing_steps = 2
            self._top_models_to_improve = 3
        elif mode == "Insane":
            self._start_random_models = 15
            self._hill_climbing_steps = 3
            self._top_models_to_improve = 4
        elif mode == "Perfect":
            self._start_random_models = 25
            self._hill_climbing_steps = 5
            self._top_models_to_improve = 5
        else:  # Normal
            self._start_random_models = 5
            self._hill_climbing_steps = 1
            self._top_models_to_improve = 2
        self._tuner_params = {
            "start_random_models": self._start_random_models,
            "hill_climbing_steps": self._hill_climbing_steps,
            "top_models_to_improve": self._top_models_to_improve,
        }

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
            for i in range(1, 10001):
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
            with open(os.path.join(self._results_path, "best_model.txt"), "r") as fin:
                best_model_name = fin.read()

            self._best_model = models_map[best_model_name]

            data_info_path = os.path.join(self._results_path, "data_info.json")
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
            ldb["metric_type"] += [self._optimize_metric]
            ldb["metric_value"] += [m.get_final_loss()]
            ldb["train_time"] += [np.round(m.get_train_time(), 2)]
        return pd.DataFrame(ldb)

    def keep_model(self, model):
        if model is None:
            return
        self._models += [model]
        self.verbose_print(
            "{} final {} {} time {} seconds".format(
                model.get_name(),
                self._optimize_metric,
                model.get_final_loss(),
                np.round(model.get_train_time(), 2),
            )
        )
        self.log_train_time(model.get_type(), model.get_train_time())

    def _get_learner_time_limit(self, model_type):

        logger.debug(
            f"Fit level: {self._fit_level}, model type: {model_type}. "
            + f"Time spend: {json.dumps(self._time_spend, indent=4)}"
        )

        if self._model_time_limit is not None:
            k = self._validation.get("k_folds", 1.0)
            return self._model_time_limit / k

        if self._fit_level == "simple_algorithms":
            return None
        if self._fit_level == "default_algorithms":
            return None

        tune_algorithms = [
            a
            for a in self._algorithms
            if a not in ["Baseline", "Linear", "Decision Tree", "Nearest Neighbors"]
        ]
        tune_algs_cnt = len(tune_algorithms)
        if tune_algs_cnt == 0:
            return None

        time_elapsed = time.time() - self._start_time
        time_left = self._total_time_limit - time_elapsed

        k_folds = self._validation.get("k_folds", 1.0)

        if self._fit_level == "not_so_random":
            tt = (
                self._total_time_limit
                - self._time_spend["simple_algorithms"]
                - self._time_spend["default_algorithms"]
            )
            if self._stack_models:
                tt *= (
                    0.6
                )  # leave some time for stacking (approx. 40% for stacking of time left)
            tt /= 2.0  # leave some time for hill-climbing
            tt /= tune_algs_cnt  # give time equally for each algorithm
            tt /= k_folds  # time is per learner (per fold)
            return tt

        if self._fit_level == "hill_climbing":
            tt = (
                self._total_time_limit
                - self._time_spend["simple_algorithms"]
                - self._time_spend["default_algorithms"]
                - self._time_spend["not_so_random"]
            )
            if self._stack_models:
                tt *= (
                    0.4
                )  # leave some time for stacking (approx. 60% for stacking of time left)
            tt /= tune_algs_cnt  # give time equally for each algorithm
            tt /= k_folds  # time is per learner (per fold)
            return tt

        if self._stack_models and self._fit_level == "stack":
            tt = time_left
            tt /= tune_algs_cnt  # give time equally for each algorithm
            tt /= k_folds  # time is per learner (per fold)
            return tt

    def train_model(self, params):

        model_path = os.path.join(self._results_path, params["name"])
        early_stop = EarlyStopping(
            {"metric": {"name": self._optimize_metric}, "log_to_dir": model_path}
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
                "total_time_limit": self._total_time_limit
                if self._model_time_limit is None
                else None,
                "total_time_start": self._start_time,
            }
        )

        mf = ModelFramework(
            params,
            callbacks=[early_stop, learner_time_constraint, total_time_constraint],
        )

        if self._enough_time_to_train(mf.get_type()):

            # self.verbose_print(params["name"] + " training start ...")
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

            # save the best one in the case the training will be interrupted
            self.select_and_save_best()
        else:
            logger.info(f"Cannot train {mf.get_type()} because of time constraint")
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

        total_time_spend = time.time() - self._start_time
        # no time left, do not train more models, sorry ...
        time_left = self._total_time_limit - total_time_spend
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
                self._total_time_limit
                - self._time_spend["simple_algorithms"]
                - self._time_spend["default_algorithms"]
            )
            if self._stack_models:
                time_should_use *= 0.6  # leave time for stacking
            if self._hill_climbing_steps > 0:
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

        if self._fit_level == "hill_climbing":

            time_should_use = (
                self._total_time_limit
                - self._time_spend["simple_algorithms"]
                - self._time_spend["default_algorithms"]
                - self._time_spend["not_so_random"]
            )
            if self._stack_models:
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

        algo_cnt = float(len(self._algorithms))
        for a in ["Baseline", "Decision Tree", "Linear", "Nearest Neighbors"]:
            if a in self._algorithms:
                algo_cnt -= 1.0
        if algo_cnt < 1.0:
            algo_cnt = 1.0

        model_time_left = time_left / algo_cnt
        if model_mean_time_spend <= model_time_left:
            return True

        return False

    def ensemble_step(self, is_stacked=False):
        if self._train_ensemble and len(self._models) > 1:
            self.ensemble = Ensemble(
                self._optimize_metric, self._ml_task, is_stacked=is_stacked
            )
            oofs, target = self.ensemble.get_oof_matrix(self._models)
            self.ensemble.fit(oofs, target)
            self.keep_model(self.ensemble)

            ensemble_path = os.path.join(
                self._results_path, "Ensemble_Stacked" if is_stacked else "Ensemble"
            )
            try:
                os.mkdir(ensemble_path)
            except Exception as e:
                raise AutoMLException(f"Cannot create directory {ensemble_path}")
            self.ensemble.save(ensemble_path)
            self._model_paths += [ensemble_path]
            # save the best one in the case the training will be interrupted
            self.select_and_save_best()

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

    def stacked_ensemble_step(self):
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

        # read X directly from parquet
        X = pd.read_parquet(self._X_train_path)

        self.stack_models()

        org_columns = X.columns.tolist()
        X_stacked = self.get_stacked_data(X)
        new_columns = X_stacked.columns.tolist()
        added_columns = [c for c in new_columns if c not in org_columns]

        # save stacked data
        X_train_stacked_path = os.path.join(
            self._results_path, "X_train_stacked.parquet"
        )
        X_stacked.to_parquet(X_train_stacked_path, index=False)

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
        if self._ml_task == REGRESSION:
            if "stratify" in self._validation:
                del self._validation["stratify"]
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
                self._optimize_metric = "rmse"
            elif self._user_set_optimize_metric not in ["rmse"]:
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

        if isinstance(y_train, pd.DataFrame):
            if "target" not in y_train.columns:
                raise AutoMLException(
                    "y_train should be Numpy array, Pandas Series or DataFrame with column 'target' "
                )
            else:
                y_train = y_train["target"]
        y_train = pd.Series(np.array(y_train), name="target")

        X_train, y_train = ExcludeRowsMissingTarget.transform(
            X_train, y_train, warn=True
        )

        return X_train, y_train, X_validation, y_validation

    def _save_data(self, X_train, y_train, X_validation=None, y_validation=None):

        self._X_train_path = os.path.join(self._results_path, "X_train.parquet")
        self._y_train_path = os.path.join(self._results_path, "y_train.parquet")

        X_train.to_parquet(self._X_train_path, index=False)

        if self._ml_task == MULTICLASS_CLASSIFICATION:
            y_train = y_train.astype(str)

        pd.DataFrame({"target": y_train}).to_parquet(self._y_train_path, index=False)

        self._validation["X_train_path"] = self._X_train_path
        self._validation["y_train_path"] = self._y_train_path
        self._validation["results_path"] = self._results_path

        columns_and_target_info = DataInfo.compute(X_train, y_train, self._ml_task)

        self._data_info = {
            "columns": X_train.columns.tolist(),
            "rows": X_train.shape[0],
            "cols": X_train.shape[1],
            "target_is_numeric": pd.api.types.is_numeric_dtype(y_train),
            "columns_info": columns_and_target_info["columns_info"],
            "target_info": columns_and_target_info["target_info"],
        }
        if columns_and_target_info.get("num_class") is not None:
            self._data_info["num_class"] = columns_and_target_info["num_class"]
        data_info_path = os.path.join(self._results_path, "data_info.json")
        with open(data_info_path, "w") as fout:
            fout.write(json.dumps(self._data_info, indent=4))

        self._drop_data_variables(X_train)

    def _drop_data_variables(self, X_train):

        X_train.drop(X_train.columns, axis=1, inplace=True)

    def _load_data_variables(self, X_train):
        if X_train.shape[1] == 0:
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

            self._start_time = time.time()

            if not isinstance(X_train, pd.DataFrame):
                raise AutoMLException(
                    "AutoML needs X_train matrix to be a Pandas DataFrame"
                )

            self._set_ml_task(y_train)

            if X_train is not None:
                X_train = X_train.copy(deep=False)

            X_train, y_train, X_validation, y_validation = self._initial_prep(
                X_train, y_train, X_validation, y_validation
            )
            self._save_data(X_train, y_train, X_validation, y_validation)

            self._set_algorithms()
            self._set_metric()
            # self._estimate_training_times()

            if self._ml_task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:
                self._check_imbalanced(y_train)

            tuner = MljarTuner(
                self._tuner_params,
                self._algorithms,
                self._ml_task,
                self._validation,
                self._explain_level,
                self._data_info,
                self._seed,
            )
            self.tuner = tuner
            self._time_spend = {}
            self._time_start = {}

            # 1. Check simple algorithms
            self._fit_level = "simple_algorithms"
            start = time.time()
            self._time_start[self._fit_level] = start
            for params in tuner.simple_algorithms_params():
                self.train_model(params)
            self._time_spend["simple_algorithms"] = np.round(time.time() - start, 2)

            # 2. Default parameters
            self._fit_level = "default_algorithms"
            start = time.time()
            self._time_start[self._fit_level] = start
            for params in tuner.default_params(len(self._models)):
                self.train_model(params)
            self._time_spend["default_algorithms"] = np.round(time.time() - start, 2)

            # 3. The not-so-random step
            self._fit_level = "not_so_random"
            start = time.time()
            self._time_start[self._fit_level] = start
            generated_params = tuner.get_not_so_random_params(len(self._models))
            for params in generated_params:
                self.train_model(params)
            self._time_spend["not_so_random"] = np.round(time.time() - start, 2)

            # 4. The hill-climbing step
            self._fit_level = "hill_climbing"
            start = time.time()
            self._time_start[self._fit_level] = start
            for params in tuner.get_hill_climbing_params(self._models):
                self.train_model(params)
            self._time_spend["hill_climbing"] = np.round(time.time() - start, 2)

            # 5. Ensemble unstacked models
            self._fit_level = "ensemble_unstacked"
            start = time.time()
            self._time_start[self._fit_level] = start
            self.ensemble_step()
            self._time_spend["ensemble_unstacked"] = np.round(time.time() - start, 2)

            if self._stack_models:
                # 6. Stack best models
                self._fit_level = "stack"
                start = time.time()
                self._time_start[self._fit_level] = start
                self.stacked_ensemble_step()
                self._time_spend["stack"] = np.round(time.time() - start, 2)

                # 7. Ensemble all models (original and stacked)
                any_stacked = False
                for m in self._models:
                    if m._is_stacked:
                        any_stacked = True
                        break
                if any_stacked:
                    self._fit_level = "ensemble_all"
                    start = time.time()
                    self.ensemble_step(is_stacked=True)
                    self._time_spend["ensemble_all"] = np.round(time.time() - start, 2)

            self._fit_time = time.time() - self._start_time

            logger.info(f"AutoML fit time: {self._fit_time}")

        except Exception as e:
            raise e
        finally:
            if self._X_train_path is not None:
                self._load_data_variables(X_train)

    def select_and_save_best(self):
        max_loss = 10e14
        for i, m in enumerate(self._models):
            if m.get_final_loss() < max_loss:
                self._best_model = m
                max_loss = m.get_final_loss()

        with open(os.path.join(self._results_path, "best_model.txt"), "w") as fout:
            fout.write(f"{self._best_model.get_name()}")

        with open(os.path.join(self._results_path, "params.json"), "w") as fout:
            params = {
                "ml_task": self._ml_task,
                "optimize_metric": self._optimize_metric,
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

        # is stacked model
        if self._best_model._is_stacked:
            self.stack_models()
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
