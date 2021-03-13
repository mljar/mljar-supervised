from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
import catboost
import optuna

from supervised.utils.metric import Metric
from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.registry import REGRESSION

EPS = 1e-8


class ExtraTreesObjective:
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
        self.max_steps = 1  # ET is trained in steps 100 trees each
        self.seed = 123

    def __call__(self, trial):
        try:
            Algorithm = (
                ExtraTreesRegressor
                if self.ml_task == REGRESSION
                else ExtraTreesClassifier
            )
            model = Algorithm(
                n_estimators=self.max_steps * 100,
                criterion=self.objective,
                max_depth=trial.suggest_int("max_depth", 2, 16),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 100),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 100),
                max_features=trial.suggest_float("max_features", 1e-8, 1),
                n_jobs=self.n_jobs,
                random_state=self.seed,
            )
            model.fit(self.X_train, self.y_train, sample_weight=self.sample_weight)

            if self.ml_task == BINARY_CLASSIFICATION:
                preds = model.predict_proba(self.X_validation)[:, 1]
            elif self.ml_task == MULTICLASS_CLASSIFICATION:
                preds = model.predict_proba(self.X_validation)
            else:  # REGRESSION
                preds = model.predict(self.X_validation)

            score = self.eval_metric(self.y_validation, preds)
            if Metric.optimize_negative(self.eval_metric.name):
                score *= -1.0

        except optuna.exceptions.TrialPruned as e:
            raise e
        except Exception as e:
            print("Exception in ExtraTreesObjective", str(e))
            return None

        return score
