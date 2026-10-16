import json
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_friedman1

from supervised import AutoML
from supervised.model_framework import ModelFramework
from supervised.tuner.time_controller import TimeController


@pytest.mark.parametrize(
    "task, default_budget",
    [("binary", False), ("multiclass", False), ("regression", False), ("regression", True)],
)
def test_discovery_training_reports_and_reload(
    tmp_path, monkeypatch, task, default_budget
):
    pytest.importorskip("features_goldmine")
    if task == "regression":
        X, y = make_friedman1(n_samples=600, n_features=6, random_state=42)
    else:
        X, y = make_classification(
            n_samples=600,
            n_features=6,
            n_informative=4,
            n_classes=3 if task == "multiclass" else 2,
            random_state=42,
        )
        y = np.array([f"class_{v}" for v in y])
    X = pd.DataFrame(X, columns=list("abcdef"))
    X["category"] = pd.cut(X.a, 4).astype(str)
    # Also exercise the real allocation without extending the discovery step.
    # Small datasets finish well before these generous total budgets.
    if not default_budget:
        original = TimeController.time_should_use
        monkeypatch.setattr(
            TimeController,
            "time_should_use",
            lambda self, step: 60 if step == "feature_discovery" else original(self, step),
        )
    root = tmp_path / "automl"
    model = AutoML(
        results_path=str(root),
        total_time_limit=600 if default_budget else 120,
        mode="Compete",
        algorithms=["Decision Tree", "Linear"],
        n_jobs=1,
        explain_level=0,
        validation_strategy={
            "validation_type": "kfold",
            "k_folds": 2,
            "shuffle": True,
            "stratify": task != "regression",
        },
        golden_features=True,
        feature_discovery=True,
        kmeans_features=False,
        features_selection=False,
        start_random_models=1,
        hill_climbing_steps=0,
        boost_on_errors=False,
        mix_encoding=False,
        train_ensemble=True,
        stack_models=False,
    )
    model.fit(X, y)
    discovered = [m for m in model._models if "FeatureDiscovery" in m.get_name()]
    assert discovered
    assert len(discovered) <= 9
    assert not (root / "errors.md").exists()
    assert any("GoldenFeatures_FeatureDiscovery" in m.get_name() for m in discovered)
    for m in discovered:
        assert "## Feature discovery" in (root / m.get_name() / "README.md").read_text()
        expected = m.predict(X.iloc[:20])
        reloaded = ModelFramework.load(root, m.get_name())
        actual = reloaded.predict(X.iloc[:20])
        np.testing.assert_allclose(
            expected.filter(like="prediction"),
            actual.filter(like="prediction"),
            rtol=1e-10,
        )
    # Inference needs only frozen transformations, never the OOF training cache.
    for path in root.glob("feature_discovery/*/training.joblib"):
        path.unlink()
    expected = model.predict_all(X.iloc[:20])
    moved = tmp_path / "moved"
    shutil.move(root, moved)
    reloaded = AutoML(results_path=str(moved))
    actual = reloaded.predict_all(X.iloc[:20])
    np.testing.assert_allclose(
        expected.filter(like="prediction"), actual.filter(like="prediction"), rtol=1e-10
    )
    state = json.loads((moved / "progress.json").read_text())
    assert state["fit_level"] == "finished"
    assert any(
        s["model_type"] == "preprocessing" for s in state["time_controller"]["spend"]
    )


def test_discovery_survives_hill_climbing_and_stacking(tmp_path, monkeypatch):
    pytest.importorskip("features_goldmine")
    X, y = make_friedman1(n_samples=600, n_features=6, random_state=42)
    X = pd.DataFrame(X)
    original = TimeController.time_should_use
    monkeypatch.setattr(TimeController, "time_should_use", lambda self, step: 60 if step == "feature_discovery" else original(self, step))
    root = tmp_path / "automl"
    model = AutoML(
        results_path=str(root), mode="Compete", total_time_limit=180,
        algorithms=["LightGBM", "Decision Tree"], n_jobs=1, explain_level=0,
        validation_strategy={"validation_type": "kfold", "k_folds": 2, "shuffle": True},
        golden_features=False, feature_discovery=True, kmeans_features=False,
        features_selection=False, start_random_models=1, hill_climbing_steps=1,
        top_models_to_improve=1, boost_on_errors=False, mix_encoding=False,
        train_ensemble=True, stack_models=True,
    )
    model.fit(X, y)
    assert not (root / "errors.md").exists()
    assert any(m._is_stacked and "FeatureDiscovery" in m.get_name() for m in model._models)
    assert not list(root.glob("feature_discovery/*/training.joblib"))
    expected = model.predict(X.iloc[:20])
    reloaded = AutoML(results_path=str(root))
    np.testing.assert_allclose(expected, reloaded.predict(X.iloc[:20]), rtol=1e-10)


def test_discovery_failure_preserves_trained_models(tmp_path, monkeypatch):
    from supervised.preprocessing.feature_discovery_transformer import FeatureDiscovery

    monkeypatch.setattr("supervised.base_automl.check_discovery_dependencies", lambda: None)

    def fail(*args):
        raise RuntimeError("Discovery failure diagnostic")

    monkeypatch.setattr(FeatureDiscovery, "prepare", fail)
    X, y = make_friedman1(n_samples=100, n_features=6, random_state=42)
    model = AutoML(
        results_path=str(tmp_path), algorithms=["Decision Tree"], total_time_limit=60,
        n_jobs=1, explain_level=0, golden_features=False, feature_discovery=True,
        features_selection=False, start_random_models=1, hill_climbing_steps=0,
        train_ensemble=False, stack_models=False,
    )
    model.fit(X, y)
    assert len(model._models) == 1
    assert np.isfinite(model.predict(X[:5])).all()
    assert "Discovery failure diagnostic" in (tmp_path / "errors.md").read_text()
