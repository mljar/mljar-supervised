from supervised.algorithms.random_forest import (
    RandomForestAlgorithm,
    RandomForestRegressorAlgorithm,
)

import optuna

from supervised.utils.metric import Metric
from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.registry import REGRESSION


class RandomForestObjective:
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
        self.ml_task = ml_task
        self.X_train = X_train
        self.y_train = y_train
        self.sample_weight = sample_weight
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.eval_metric = eval_metric
        self.n_jobs = n_jobs
        self.objective = "mse" if ml_task == REGRESSION else "gini"
        self.max_steps = 10  # RF is trained in steps 100 trees each
        self.seed = random_state

    def __call__(self, trial):
        try:
            Algorithm = (
                RandomForestRegressorAlgorithm
                if self.ml_task == REGRESSION
                else RandomForestAlgorithm
            )
            self.objective = (
                "mse"
                if self.ml_task == REGRESSION
                else trial.suggest_categorical("criterion", ["gini", "entropy"])
            )
            params = {
                "max_steps": self.max_steps,
                "criterion": self.objective,
                "max_depth": trial.suggest_int("max_depth", 2, 32),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100),
                "max_features": trial.suggest_float("max_features", 0.01, 1),
                "n_jobs": self.n_jobs,
                "seed": self.seed,
                "ml_task": self.ml_task,
            }
            model = Algorithm(params)
            model.fit(self.X_train, self.y_train, sample_weight=self.sample_weight)

            preds = model.predict(self.X_validation)

            score = self.eval_metric(self.y_validation, preds)
            if Metric.optimize_negative(self.eval_metric.name):
                score *= -1.0

        except optuna.exceptions.TrialPruned as e:
            raise e
        except Exception as e:
            print("Exception in RandomForestObjective", str(e))
            return None

        return score
