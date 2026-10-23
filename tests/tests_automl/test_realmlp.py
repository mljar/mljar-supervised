import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_classification

from supervised import AutoML


def test_explicit_weights_rejected(tmp_path):
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    m = AutoML(results_path=str(tmp_path / "weighted"), mode="Compete", algorithms=["RealMLP"])
    with pytest.raises(ValueError, match="sample weights"):
        m.fit(X, y, sample_weight=np.ones(len(y)))


@pytest.mark.skipif(
    importlib.util.find_spec("pytabkit") is None, reason="optional dependency"
)
def test_automl_training_explanations_and_reload(tmp_path):
    X, y = make_classification(n_samples=300, n_features=8, random_state=42)
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(8)])
    X["cat"] = pd.Series(np.where(X.x0 > 0, "high", "low"), dtype="category")
    X.loc[::9, "x1"] = np.nan
    path = str(tmp_path / "automl")
    m = AutoML(
        results_path=path,
        mode="Compete", algorithms=["RealMLP"],
        total_time_limit=120,
        train_ensemble=False,
        stack_models=False,
        golden_features=False,
        feature_discovery=False,
        boost_on_errors=False,
        kmeans_features=False,
        mix_encoding=False,
        features_selection=False,
        explain_level=2,
        start_random_models=2,
        hill_climbing_steps=0,
        n_jobs=2,
        validation_strategy={
            "validation_type": "split",
            "train_ratio": 0.8,
            "shuffle": True,
            "stratify": True,
        },
        custom_params={"realmlp": {"device": "cpu", "n_epochs": 5}},
    )
    m.fit(X, y)
    p = m.predict_proba(X)
    assert len(m._models) == 2
    assert list(tmp_path.rglob("*_importance.csv"))
    assert not list(tmp_path.rglob("*shap*.png"))
    assert list(tmp_path.rglob("learning_curves.png"))
    assert list(tmp_path.rglob("*training.log.json"))
    loaded = AutoML(results_path=path)
    assert_allclose(p, loaded.predict_proba(X), atol=1e-6)
    assert loaded.custom_params == {"realmlp": {"device": "cpu", "n_epochs": 5}}
    # Earlier development artifacts still reload after the public API rename.
    settings_path = tmp_path / "automl" / "params.json"
    settings = json.loads(settings_path.read_text())
    settings["realmlp_params"] = settings.pop("custom_params")["realmlp"]
    settings_path.write_text(json.dumps(settings))
    legacy = AutoML(results_path=path)
    assert_allclose(p, legacy.predict_proba(X), atol=1e-6)
    assert legacy.custom_params == loaded.custom_params


def test_custom_params_existing_algorithm(tmp_path):
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    options = {"random_forest": {"max_depth": 3}}
    m = AutoML(
        results_path=str(tmp_path / "forest"),
        algorithms=["Random Forest"],
        custom_params=options,
        train_ensemble=False,
        explain_level=0,
        start_random_models=1,
        hill_climbing_steps=0,
        n_jobs=2,
    )
    m.fit(X, y)
    assert m._models[0].learners[0].params["max_depth"] == 3
    assert options == {"random_forest": {"max_depth": 3}}
    loaded = AutoML(results_path=str(tmp_path / "forest"))
    assert_allclose(m.predict_proba(X), loaded.predict_proba(X))
    assert loaded.custom_params == options


@pytest.mark.parametrize("mode", ["Explain", "Perform", "Optuna"])
def test_realmlp_requires_compete(mode):
    with pytest.raises(ValueError, match="only in mode"):
        model = AutoML(mode=mode, algorithms=["RealMLP"])
        model._ml_task = "binary_classification"
        model._get_algorithms()


@pytest.mark.parametrize("available", [True, False])
@pytest.mark.parametrize("mode", ["Explain", "Perform", "Compete", "Optuna"])
def test_automatic_realmlp_selection(monkeypatch, available, mode):
    monkeypatch.setattr(
        "supervised.algorithms.realmlp.dependencies_available", lambda: available
    )
    assert ("RealMLP" in AutoML(mode=mode)._get_algorithms()) == (
        mode == "Compete" and available
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pytabkit") is None, reason="optional dependency"
)
def test_automatic_weighted_training_skips_realmlp(tmp_path, monkeypatch):
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = AutoML(
        mode="Compete", results_path=str(tmp_path / "weighted_auto"),
        train_ensemble=False, stack_models=False, golden_features=False,
        features_selection=False, boost_on_errors=False, kmeans_features=False,
        mix_encoding=False, start_random_models=1, hill_climbing_steps=0,
        explain_level=0, n_jobs=2,
        validation_strategy={"validation_type": "split", "train_ratio": 0.8,
                             "shuffle": True, "stratify": True},
    )
    monkeypatch.setattr(model, "_get_algorithms", lambda: ["Decision Tree", "RealMLP"])
    with pytest.warns(UserWarning, match="RealMLP skipped"):
        model.fit(X, y, sample_weight=np.ones(len(y)))
    assert model._algorithms == ["Decision Tree"]
    assert len(model.predict(X)) == len(y)
