import os
import weakref
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from supervised import AutoML


@pytest.mark.skipif(
    os.environ.get("MLJAR_TEST_SDM") != "1", reason="opt-in checkpoint integration"
)
@pytest.mark.parametrize("algorithm", ["Kumo", "TabICLv2"])
def test_saved_sdm_folds_release_networks_after_prediction(
    tmp_path, monkeypatch, algorithm
):
    from supervised.algorithms.factory import AlgorithmFactory
    from supervised.algorithms.sdm import SDMAlgorithm
    from supervised.model_framework import ModelFramework
    from supervised.preprocessing.preprocessing import Preprocessing

    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    X = pd.DataFrame(X, columns=list("abcd"))
    params = {
        "model_type": algorithm,
        "ml_task": "binary_classification",
        "device": os.environ.get("MLJAR_TEST_SDM_DEVICE", "cpu"),
        "max_context_size": 64,
        "num_estimators": 1,
        "local_files_only": True,
        "n_jobs": 2,
    }
    learner = AlgorithmFactory.get_algorithm(params)
    learner.fit(X[:80], y[:80], X_validation=X[80:], y_validation=y[80:])
    expected = learner.predict(X[80:84])
    path = str(tmp_path / learner.get_fname())
    learner.save(path)
    learner.unload()

    framework = ModelFramework({"learner": params, "validation_strategy": {}})
    # Ten independently loaded wrappers exercise checkpoint residency without
    # spending the integration-test budget on ten equivalent training fits.
    framework.learners = [
        AlgorithmFactory.load(learner.get_params(), path, lazy_load=True)
        for _ in range(10)
    ]
    framework.preprocessings = [
        Preprocessing({"ml_task": params["ml_task"]}) for _ in range(10)
    ]
    networks = []
    load = SDMAlgorithm._load_network

    def tracked_load(self):
        assert not any(network() is not None for network in networks)
        load(self)
        networks.append(weakref.ref(self.model))

    monkeypatch.setattr(SDMAlgorithm, "_load_network", tracked_load)
    for _ in range(2):
        prediction = framework.predict(X[80:84])
        np.testing.assert_allclose(
            prediction.prediction_1, expected, rtol=1e-5, atol=1e-7
        )
        assert all(fold.model is None for fold in framework.learners)
        assert not any(network() is not None for network in networks)
    assert len(networks) == 20


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
def test_compete_precision_trials_and_saved_reports(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(
        "supervised.tuner.mljar_tuner.HillClimbing.get", lambda *args: [None, None]
    )
    X, y = make_classification(n_samples=120, n_features=5, random_state=42)
    device = os.environ.get("MLJAR_TEST_SDM_DEVICE", "cpu")
    model = AutoML(
        results_path=str(tmp_path / "automl"),
        mode="Compete",
        algorithms=["Kumo", "TabICLv2"],
        model_time_limit=60,
        train_ensemble=False,
        stack_models=False,
        golden_features=False,
        feature_discovery=False,
        features_selection=False,
        boost_on_errors=False,
        kmeans_features=False,
        mix_encoding=False,
        explain_level=0,
        start_random_models=1,
        hill_climbing_steps=1,
        top_models_to_improve=1,
        n_jobs=2,
        validation_strategy={
            "validation_type": "split",
            "train_ratio": 0.8,
            "shuffle": True,
            "stratify": True,
        },
        custom_params={
            name: {
                "device": device,
                "max_context_size": 64,
                "num_estimators": 1,
                "local_files_only": True,
            }
            for name in ["kumo", "tabiclv2"]
        },
    )
    model.fit(X, y)
    precisions = ["fp16", "fp32"] if device.startswith("cuda") else ["fp16"]
    assert {
        (m.get_type(), m.params["learner"]["precision"]) for m in model._models
    } == {(name, p) for name in ["Kumo", "TabICLv2"] for p in precisions}
    for m in model._models:
        requested = m.params["learner"]["precision"]
        effective = requested if device.startswith("cuda") else "fp32"
        for learner in m.learners:
            assert learner.training_info["precision"] == effective
        path = tmp_path / "automl" / m.get_name()
        assert effective in (path / "README.md").read_text()
        for log in path.glob("*_training.log.json"):
            info = json.loads(log.read_text())
            assert info["requested_precision"] == requested
            assert info["precision"] == effective
        assert all(l.model is None for l in m.learners)
    p = model.predict_proba(X[:17])
    loaded = AutoML(results_path=str(tmp_path / "automl"))
    np.testing.assert_allclose(p, loaded.predict_proba(X[:17]), atol=1e-6)


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
