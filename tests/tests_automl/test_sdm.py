import os
import numpy as np
import pytest
from sklearn.datasets import make_classification
from supervised import AutoML


@pytest.mark.parametrize("algorithm", ["Kumo", "TabICLv2"])
@pytest.mark.parametrize("mode", ["Explain", "Perform", "Optuna"])
def test_compete_only(algorithm, mode):
    model = AutoML(mode=mode, algorithms=[algorithm])
    model._ml_task = "binary_classification"
    with pytest.raises(ValueError, match="only in mode"):
        model._get_algorithms()


@pytest.mark.parametrize("available", [True, False])
def test_automatic_selection(monkeypatch, available):
    monkeypatch.setattr(
        "supervised.algorithms.sdm.automatic_available", lambda: available
    )
    algorithms = AutoML(mode="Compete")._get_algorithms()
    assert ("Kumo" in algorithms) == available
    assert ("TabICLv2" in algorithms) == available
    assert "Kumo" not in AutoML(mode="Explain")._get_algorithms()


@pytest.mark.parametrize("algorithm", ["Kumo", "TabICLv2"])
def test_weights_rejected(algorithm, tmp_path):
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = AutoML(
        mode="Compete", algorithms=[algorithm], results_path=str(tmp_path / "weighted")
    )
    with pytest.raises(ValueError, match="sample weights"):
        model.fit(X, y, sample_weight=np.ones(len(y)))


@pytest.mark.skipif(
    os.environ.get("MLJAR_TEST_SDM") != "1", reason="opt-in checkpoint integration"
)
def test_compete_both_algorithms_and_reload(tmp_path):
    X, y = make_classification(n_samples=300, n_features=8, random_state=42)
    path = str(tmp_path / "automl")
    model = AutoML(
        results_path=path,
        mode="Compete",
        algorithms=["Kumo", "TabICLv2"],
        total_time_limit=180,
        train_ensemble=True,
        stack_models=False,
        golden_features=False,
        features_selection=False,
        boost_on_errors=False,
        kmeans_features=False,
        mix_encoding=False,
        explain_level=2,
        start_random_models=1,
        hill_climbing_steps=0,
        n_jobs=4,
        validation_strategy={
            "validation_type": "split",
            "train_ratio": 0.8,
            "shuffle": True,
            "stratify": True,
        },
        custom_params={
            name: {
                "device": os.environ.get("MLJAR_TEST_SDM_DEVICE", "cpu"),
                "max_context_size": 256,
                "num_estimators": 2,
                "local_files_only": True,
            }
            for name in ["kumo", "tabiclv2"]
        },
    )
    model.fit(X, y)
    types = {m.get_type() for m in model._models}
    assert {"Kumo", "TabICLv2"} <= types
    p = model.predict_proba(X[:30])
    reloaded = AutoML(results_path=path)
    np.testing.assert_allclose(p, reloaded.predict_proba(X[:30]), atol=1e-6)
    assert list(tmp_path.rglob("*training.log.json"))
    assert list(tmp_path.rglob("learning_curves.png"))
    assert list(tmp_path.rglob("*_importance.csv"))
    assert not list(tmp_path.rglob("*shap*.png"))
    assert (tmp_path / "automl" / "sdm_weights").is_dir()


@pytest.mark.skipif(
    os.environ.get("MLJAR_TEST_SDM") != "1", reason="checkpoint integration"
)
def test_automatic_weights_skip_both(tmp_path, monkeypatch):
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    m = AutoML(
        mode="Compete",
        results_path=str(tmp_path / "weighted_auto"),
        train_ensemble=False,
        stack_models=False,
        golden_features=False,
        features_selection=False,
        boost_on_errors=False,
        kmeans_features=False,
        mix_encoding=False,
        start_random_models=1,
        hill_climbing_steps=0,
        explain_level=0,
        n_jobs=2,
    )
    monkeypatch.setattr(
        m, "_get_algorithms", lambda: ["Decision Tree", "Kumo", "TabICLv2"]
    )
    with pytest.warns(UserWarning, match="skipped because sample weights"):
        m.fit(X, y, sample_weight=np.ones(len(y)))
    assert m._algorithms == ["Decision Tree"]
