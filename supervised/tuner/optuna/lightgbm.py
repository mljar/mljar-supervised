import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

from supervised.utils.metric import Metric
from supervised.utils.metric import (
    lightgbm_eval_metric_r2,
    lightgbm_eval_metric_spearman,
    lightgbm_eval_metric_pearson,
    lightgbm_eval_metric_f1,
    lightgbm_eval_metric_average_precision,
    lightgbm_eval_metric_accuracy,
    lightgbm_eval_metric_user_defined,
)
from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.registry import REGRESSION

from supervised.algorithms.lightgbm import (
    lightgbm_objective,
    lightgbm_eval_metric,
)


EPS = 1e-8


class LightgbmObjective:
    def __init__(
        self,
        ml_task,
        X_train,
        y_train,
        sample_weight,
        X_validation,
        y_validation,
        sample_weight_validation,
        eval_metric,
        cat_features_indices,
        n_jobs,
        random_state,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.sample_weight = sample_weight
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.sample_weight_validation = sample_weight_validation
        self.dtrain = lgb.Dataset(
            self.X_train.to_numpy()
            if isinstance(self.X_train, pd.DataFrame)
            else self.X_train,
            label=self.y_train,
            weight=self.sample_weight,
        )
        self.dvalid = lgb.Dataset(
            self.X_validation.to_numpy()
            if isinstance(self.X_validation, pd.DataFrame)
            else self.X_validation,
            label=self.y_validation,
            weight=self.sample_weight_validation,
        )

        self.cat_features_indices = cat_features_indices
        self.eval_metric = eval_metric
        self.learning_rate = 0.025
        self.rounds = 1000
        self.early_stopping_rounds = 50
        self.seed = random_state

        self.n_jobs = n_jobs
        if n_jobs == -1:
            self.n_jobs = 0

        self.objective = ""
        self.eval_metric_name = ""

        self.eval_metric_name, self.custom_eval_metric_name = lightgbm_eval_metric(
            ml_task, eval_metric.name
        )

        self.custom_eval_metric = None
        if self.eval_metric.name == "r2":
            self.custom_eval_metric = lightgbm_eval_metric_r2
        elif self.eval_metric.name == "spearman":
            self.custom_eval_metric = lightgbm_eval_metric_spearman
        elif self.eval_metric.name == "pearson":
            self.custom_eval_metric = lightgbm_eval_metric_pearson
        elif self.eval_metric.name == "f1":
            self.custom_eval_metric = lightgbm_eval_metric_f1
        elif self.eval_metric.name == "average_precision":
            self.custom_eval_metric = lightgbm_eval_metric_average_precision
        elif self.eval_metric.name == "accuracy":
            self.custom_eval_metric = lightgbm_eval_metric_accuracy
        elif self.eval_metric.name == "user_defined_metric":
            self.custom_eval_metric = lightgbm_eval_metric_user_defined

        self.num_class = (
            len(np.unique(y_train)) if ml_task == MULTICLASS_CLASSIFICATION else None
        )
        self.objective = lightgbm_objective(ml_task, eval_metric.name)

    def __call__(self, trial):
        param = {
            "objective": self.objective,
            "metric": self.eval_metric_name,
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [0.0125, 0.025, 0.05, 0.1]
            ),
            "num_leaves": trial.suggest_int("num_leaves", 2, 2048),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": min(
                trial.suggest_float("feature_fraction", 0.3, 1.0 + EPS), 1.0
            ),
            "bagging_fraction": min(
                trial.suggest_float("bagging_fraction", 0.3, 1.0 + EPS), 1.0
            ),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "feature_pre_filter": False,
            "seed": self.seed,
            "num_threads": self.n_jobs,
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        }

        if self.cat_features_indices:
            param["cat_feature"] = self.cat_features_indices
            param["cat_l2"] = trial.suggest_float("cat_l2", EPS, 100.0)
            param["cat_smooth"] = trial.suggest_float("cat_smooth", EPS, 100.0)

        if self.num_class is not None:
            param["num_class"] = self.num_class

        try:

            metric_name = self.eval_metric_name
            if metric_name == "custom":
                metric_name = self.custom_eval_metric_name
            pruning_callback = optuna.integration.LightGBMPruningCallback(
                trial, metric_name, "validation"
            )

            gbm = lgb.train(
                param,
                self.dtrain,
                valid_sets=[self.dvalid],
                valid_names=["validation"],
                verbose_eval=False,
                callbacks=[pruning_callback],
                num_boost_round=self.rounds,
                early_stopping_rounds=self.early_stopping_rounds,
                feval=self.custom_eval_metric,
            )

            preds = gbm.predict(self.X_validation)
            score = self.eval_metric(self.y_validation, preds)
            if Metric.optimize_negative(self.eval_metric.name):
                score *= -1.0
        except optuna.exceptions.TrialPruned as e:
            raise e
        except Exception as e:
            print("Exception in LightgbmObjective", str(e))
            return None

        return score
