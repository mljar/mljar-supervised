from supervised.algorithms.nn import (
    MLPAlgorithm,
    MLPRegressorAlgorithm,
)
import optuna

from supervised.utils.metric import Metric
from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.registry import REGRESSION


class NeuralNetworkObjective:
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
        self.seed = random_state

    def __call__(self, trial):
        try:
            Algorithm = (
                MLPRegressorAlgorithm if self.ml_task == REGRESSION else MLPAlgorithm
            )
            params = {
                "dense_1_size": trial.suggest_int("dense_1_size", 4, 100),
                "dense_2_size": trial.suggest_int("dense_2_size", 2, 100),
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", [0.005, 0.01, 0.05, 0.1, 0.2]
                ),
                "learning_rate_type": trial.suggest_categorical(
                    "learning_rate_type", ["constant", "adaptive"]
                ),
                "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
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
            print("Exception in NeuralNetworkObjective", str(e))
            return None

        return score
