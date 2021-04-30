from catboost import CatBoostClassifier, CatBoostRegressor, CatBoost, Pool
import catboost
import optuna

from supervised.utils.metric import (
    Metric,
    CatBoostEvalMetricSpearman,
    CatBoostEvalMetricPearson,
    CatBoostEvalMetricAveragePrecision,
    CatBoostEvalMetricMSE,
    CatBoostEvalMetricUserDefined,
)

from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.registry import REGRESSION

from supervised.algorithms.catboost import catboost_objective, catboost_eval_metric

EPS = 1e-8


class CatBoostObjective:
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
        self.ml_task = ml_task
        self.X_train = X_train
        self.y_train = y_train
        self.sample_weight = sample_weight
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.eval_metric = eval_metric
        self.cat_features = cat_features_indices
        self.eval_set = Pool(
            data=X_validation,
            label=y_validation,
            cat_features=self.cat_features,
            weight=sample_weight_validation,
        )
        self.n_jobs = n_jobs
        self.rounds = 1000
        self.learning_rate = 0.0125
        self.early_stopping_rounds = 50
        self.seed = random_state

        self.objective = catboost_objective(ml_task, self.eval_metric.name)
        self.eval_metric_name = catboost_eval_metric(ml_task, self.eval_metric.name)
        self.custom_eval_metric = None
        if self.eval_metric_name == "spearman":
            self.custom_eval_metric = CatBoostEvalMetricSpearman()
        elif self.eval_metric_name == "pearson":
            self.custom_eval_metric = CatBoostEvalMetricPearson()
        elif self.eval_metric_name == "average_precision":
            self.custom_eval_metric = CatBoostEvalMetricAveragePrecision()
        elif self.eval_metric_name == "mse":
            self.custom_eval_metric = CatBoostEvalMetricMSE()
        elif self.eval_metric_name == "user_defined_metric":
            self.custom_eval_metric = CatBoostEvalMetricUserDefined()

    def __call__(self, trial):
        try:
            params = {
                "iterations": self.rounds,
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", [0.05, 0.1, 0.2]
                ),
                "depth": trial.suggest_int("depth", 2, 9),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", 0.0001, 10.0, log=False
                ),
                "random_strength": trial.suggest_float(
                    "random_strength", EPS, 10.0, log=False
                ),
                "rsm": trial.suggest_float("rsm", 0.1, 1),  # colsample_bylevel=rsm
                "loss_function": self.objective,
                "eval_metric": self.eval_metric_name,
                "verbose": False,
                "allow_writing_files": False,
                "thread_count": self.n_jobs,
                "random_seed": self.seed,
                # "border_count": trial.suggest_int("border_count", 16, 2048),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
                # "bootstrap_type": "Bernoulli"
                # trial.suggest_categorical(
                #    "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
                # ),
            }
            # if params["bootstrap_type"] == "Bayesian":
            #    params["bagging_temperature"] = trial.suggest_float(
            #        "bagging_temperature", 0, 10
            #    )
            # elif params["bootstrap_type"] in ["Bernoulli", "MVS"]:
            # params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

            Algorithm = (
                CatBoostRegressor if self.ml_task == REGRESSION else CatBoostClassifier
            )
            if self.custom_eval_metric is not None:
                params["eval_metric"] = self.custom_eval_metric
            model = Algorithm(**params)

            model.fit(
                self.X_train,
                self.y_train,
                sample_weight=self.sample_weight,
                early_stopping_rounds=self.early_stopping_rounds,
                eval_set=self.eval_set,
                verbose_eval=False,
                cat_features=self.cat_features,
            )

            if self.ml_task == BINARY_CLASSIFICATION:
                preds = model.predict_proba(
                    self.X_validation, ntree_end=model.best_iteration_ + 1
                )[:, 1]
            elif self.ml_task == MULTICLASS_CLASSIFICATION:
                preds = model.predict_proba(
                    self.X_validation, ntree_end=model.best_iteration_ + 1
                )
            else:  # REGRESSION
                preds = model.predict(
                    self.X_validation, ntree_end=model.best_iteration_ + 1
                )

            score = self.eval_metric(self.y_validation, preds)
            if Metric.optimize_negative(self.eval_metric.name):
                score *= -1.0

        except optuna.exceptions.TrialPruned as e:
            raise e
        except Exception as e:
            print("Exception in CatBoostObjective", str(e))
            # import traceback
            # print(traceback.format_exc())
            return None

        return score
