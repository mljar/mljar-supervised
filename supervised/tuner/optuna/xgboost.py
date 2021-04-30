import numpy as np
import xgboost as xgb
import optuna

from supervised.utils.metric import Metric
from supervised.utils.metric import (
    xgboost_eval_metric_r2,
    xgboost_eval_metric_spearman,
    xgboost_eval_metric_pearson,
    xgboost_eval_metric_f1,
    xgboost_eval_metric_average_precision,
    xgboost_eval_metric_accuracy,
    xgboost_eval_metric_mse,
    xgboost_eval_metric_user_defined,
)
from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.registry import REGRESSION

from supervised.algorithms.xgboost import xgboost_objective, xgboost_eval_metric

EPS = 1e-8


class XgboostObjective:
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
        n_jobs,
        random_state,
    ):
        self.dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
        self.dvalidation = xgb.DMatrix(
            X_validation, label=y_validation, weight=sample_weight_validation
        )
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.eval_metric = eval_metric
        self.n_jobs = n_jobs

        self.learning_rate = 0.0125
        self.rounds = 1000
        self.early_stopping_rounds = 50
        self.seed = random_state

        self.objective = ""
        self.eval_metric_name = ""
        self.num_class = (
            len(np.unique(y_train)) if ml_task == MULTICLASS_CLASSIFICATION else None
        )

        self.objective = xgboost_objective(ml_task, eval_metric.name)
        self.eval_metric_name = xgboost_eval_metric(ml_task, eval_metric.name)

        self.custom_eval_metric = None
        if self.eval_metric_name == "r2":
            self.custom_eval_metric = xgboost_eval_metric_r2
        elif self.eval_metric_name == "spearman":
            self.custom_eval_metric = xgboost_eval_metric_spearman
        elif self.eval_metric_name == "pearson":
            self.custom_eval_metric = xgboost_eval_metric_pearson
        elif self.eval_metric_name == "f1":
            self.custom_eval_metric = xgboost_eval_metric_f1
        elif self.eval_metric_name == "average_precision":
            self.custom_eval_metric = xgboost_eval_metric_average_precision
        elif self.eval_metric_name == "accuracy":
            self.custom_eval_metric = xgboost_eval_metric_accuracy
        elif self.eval_metric_name == "mse":
            self.custom_eval_metric = xgboost_eval_metric_mse
        elif self.eval_metric_name == "user_defined_metric":
            self.custom_eval_metric = xgboost_eval_metric_user_defined

    def __call__(self, trial):
        param = {
            "objective": self.objective,
            "eval_metric": self.eval_metric_name,
            "tree_method": "hist",
            "booster": "gbtree",
            "eta": trial.suggest_categorical("eta", [0.0125, 0.025, 0.05, 0.1]),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "lambda": trial.suggest_float("lambda", EPS, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", EPS, 10.0, log=True),
            "colsample_bytree": min(
                trial.suggest_float("colsample_bytree", 0.3, 1.0 + EPS), 1.0
            ),
            "subsample": min(trial.suggest_float("subsample", 0.3, 1.0 + EPS), 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            "n_jobs": self.n_jobs,
            "seed": self.seed,
            "verbosity": 0,
        }
        if self.custom_eval_metric is not None:
            del param["eval_metric"]

        if self.num_class is not None:
            param["num_class"] = self.num_class
        try:
            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, f"validation-{self.eval_metric_name}"
            )
            bst = xgb.train(
                param,
                self.dtrain,
                self.rounds,
                evals=[(self.dvalidation, "validation")],
                early_stopping_rounds=self.early_stopping_rounds,
                callbacks=[pruning_callback],
                verbose_eval=False,
                feval=self.custom_eval_metric,
            )
            preds = bst.predict(self.dvalidation, ntree_limit=bst.best_ntree_limit)
            score = self.eval_metric(self.y_validation, preds)
            if Metric.optimize_negative(self.eval_metric.name):
                score *= -1.0
        except optuna.exceptions.TrialPruned as e:
            raise e
        except Exception as e:
            print("Exception in XgboostObjective", str(e))
            return None

        return score
