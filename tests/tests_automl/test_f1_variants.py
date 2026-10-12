import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from supervised import AutoML
from supervised.algorithms import catboost, lightgbm, xgboost
from supervised.utils.metric import F1_AVERAGES, Metric


@pytest.mark.parametrize("name", F1_AVERAGES)
@pytest.mark.parametrize("classes", [2, 3])
def test_f1_training_ensemble_reports_and_reload(tmp_path, monkeypatch, name, classes):
    for module in [catboost, lightgbm, xgboost]:
        monkeypatch.setitem(module.additional, "max_rounds", 8)
    X, y = make_classification(
        n_samples=150,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=classes,
        n_clusters_per_class=1,
        weights=[0.65, 0.35] if classes == 2 else [0.6, 0.2, 0.2],
        random_state=123,
    )
    X = pd.DataFrame(X, columns=list("abcde"))
    # String labels also check that predict() keeps the original label mapping.
    y = np.array([f"class_{value}" for value in y])
    root = tmp_path / name
    model = AutoML(
        results_path=str(root),
        ml_task=(
            "binary_classification" if classes == 2 else "multiclass_classification"
        ),
        eval_metric=name,
        algorithms=["Baseline", "Decision Tree", "LightGBM", "Xgboost", "CatBoost"],
        train_ensemble=True,
        stack_models=False,
        explain_level=0,
        validation_strategy={
            "validation_type": "kfold",
            "k_folds": 2,
            "shuffle": True,
            "stratify": True,
            "random_seed": 123,
        },
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        boost_on_errors=False,
        mix_encoding=False,
        start_random_models=1,
        hill_climbing_steps=0,
        total_time_limit=120,
        n_jobs=1,
        verbose=0,
    )
    model.fit(X.copy(), y)
    types = {m.get_type() for m in model._models}
    assert {
        "Baseline",
        "Decision Tree",
        "LightGBM",
        "Xgboost",
        "CatBoost",
        "Ensemble",
    }.issubset(types)
    for member in model._models:
        oof = member.get_out_of_folds()
        probabilities = oof[
            [c for c in oof.columns if c.startswith("prediction")]
        ].to_numpy()
        assert member.get_final_loss() == pytest.approx(
            Metric({"name": name})(oof.target.to_numpy(), probabilities)
        )
        assert member.get_metric_name() == name
        if member.get_type() != "Ensemble":
            assert (
                f"{name} ({F1_AVERAGES[name]} averaging)"
                in (root / member.get_name() / "README.md").read_text()
            )
            desc = json.loads((root / member.get_name() / "framework.json").read_text())
            assert desc["metric_name"] == name
    assert set(model.get_leaderboard().metric_type) == {name}
    labels = model.predict(X.copy())
    assert model.score(X.copy(), y) == accuracy_score(y, labels)
    expected = model.predict_proba(X.copy())
    reloaded = AutoML(results_path=str(root))
    np.testing.assert_array_equal(reloaded.predict_proba(X.copy()), expected)
    np.testing.assert_array_equal(reloaded.predict(X.copy()), labels)
    assert reloaded._eval_metric == name
