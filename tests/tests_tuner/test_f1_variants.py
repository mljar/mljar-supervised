import numpy as np
import optuna
import pytest
from sklearn.datasets import make_classification

from supervised.tuner.optuna.catboost import CatBoostObjective
from supervised.tuner.optuna.lightgbm import LightgbmObjective
from supervised.tuner.optuna.xgboost import XgboostObjective
from supervised.utils.metric import F1_AVERAGES, Metric


@pytest.mark.parametrize("name", F1_AVERAGES)
@pytest.mark.parametrize(
    "objective_class", [LightgbmObjective, XgboostObjective, CatBoostObjective]
)
def test_optuna_f1_objective_and_pruning_maximize_selected_variant(
    name, objective_class
):
    X, y = make_classification(
        n_samples=90,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=123,
    )
    arguments = dict(
        ml_task="multiclass_classification",
        X_train=X[:60],
        y_train=y[:60],
        sample_weight=None,
        X_validation=X[60:],
        y_validation=y[60:],
        sample_weight_validation=None,
        eval_metric=Metric({"name": name}),
        n_jobs=1,
        random_state=123,
    )
    if objective_class != XgboostObjective:
        arguments["cat_features_indices"] = []
    objective = objective_class(**arguments)
    objective.rounds = 3
    objective.early_stopping_rounds = 2
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=123)
    )
    study.optimize(objective, n_trials=1)
    trial = study.trials[0]
    assert trial.state == optuna.trial.TrialState.COMPLETE
    assert np.isfinite(trial.value) and 0 <= trial.value <= 1
    if objective_class != CatBoostObjective:
        assert trial.intermediate_values
        assert all(0 <= value <= 1 for value in trial.intermediate_values.values())
        assert objective.custom_eval_metric.keywords["metric_name"] == name
    else:
        assert (
            objective.eval_metric_name
            == f"TotalF1:average={F1_AVERAGES[name].capitalize()}"
        )
