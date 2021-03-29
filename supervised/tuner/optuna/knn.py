import numpy as np
from supervised.algorithms.knn import KNeighborsAlgorithm, KNeighborsRegressorAlgorithm
import optuna

from supervised.utils.metric import Metric
from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.registry import REGRESSION


class KNNObjective:
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
        self.seed = random_state

    def __call__(self, trial):
        try:
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 128),
                "weights": trial.suggest_categorical(
                    "weights", ["uniform", "distance"]
                ),
                "n_jobs": self.n_jobs,
                "rows_limit": 100000,
                "ml_task": self.ml_task,
            }
            Algorithm = (
                KNeighborsRegressorAlgorithm
                if self.ml_task == REGRESSION
                else KNeighborsAlgorithm
            )
            model = Algorithm(params)
            model.fit(self.X_train, self.y_train, sample_weight=self.sample_weight)
            preds = model.predict(self.X_validation)

            score = self.eval_metric(self.y_validation, preds)
            if Metric.optimize_negative(self.eval_metric.name):
                score *= -1.0

        except optuna.exceptions.TrialPruned as e:
            raise e
        except Exception as e:
            print("Exception in KNNObjective", str(e))
            return None

        return score
