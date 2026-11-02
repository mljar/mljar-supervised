import json
import shutil

import numpy as np
import pandas as pd
import pytest

from supervised import AutoML
from supervised.algorithms import sdm
from supervised.algorithms.factory import AlgorithmFactory
from supervised.exceptions import ResourceAdmissionError
from supervised.preprocessing.preprocessing import Preprocessing


@pytest.fixture
def fake_network(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    checkpoint = tmp_path / "fake-weights.pt"
    checkpoint.write_bytes(b"pretrained-test-weights")
    notice = tmp_path / "notice.txt"
    notice.write_text("test")
    calls = []

    def download(self):
        self.checkpoint_path = str(checkpoint)
        self.notice_paths = [str(notice)]
        self.checkpoint_info = dict(revision="test", filename=checkpoint.name)

    def predict(self, X):
        if not hasattr(self, "context"):
            self._load_state(self.model_file_path)
        calls.append((self.name, len(self.context), len(X)))
        z = np.asarray(X.iloc[:, 0], dtype=float) + len(self.context) / 1000
        if self.ml_task == "regression":
            return z + float(np.mean(self.context_y))
        if len(self.classes_) == 2:
            return 1 / (1 + np.exp(-z))
        values = np.column_stack([np.exp(z * (i + 1) / 5) for i in range(len(self.classes_))])
        return values / values.sum(axis=1, keepdims=True)

    monkeypatch.setattr(sdm, "check_dependencies", lambda: None)
    monkeypatch.setattr(sdm, "version", lambda name: "0.2.1")
    monkeypatch.setattr(sdm.SDMAlgorithm, "_download", download)
    monkeypatch.setattr(sdm.SDMAlgorithm, "_load_network", lambda self: setattr(self, "model", object()))
    monkeypatch.setattr(sdm.SDMAlgorithm, "_check_resources",
                        lambda self, rows, cols, checkpoint_bytes=0: {"rows": rows, "features": cols})
    monkeypatch.setattr(sdm.SDMAlgorithm, "predict", predict)
    return calls


def options(root, algorithm, task, validation):
    return dict(
        results_path=str(root), mode="Compete", algorithms=[algorithm],
        ml_task=task, eval_metric="rmse" if task == "regression" else "logloss",
        total_time_limit=None, validation_strategy=validation,
        train_ensemble=False, stack_models=False, explain_level=0,
        golden_features=False, feature_discovery=False, features_selection=False,
        kmeans_features=False, mix_encoding=False, boost_on_errors=False,
        start_random_models=1, hill_climbing_steps=0, n_jobs=1, verbose=0,
        custom_params={algorithm.lower(): {"device": "cpu"}},
    )


@pytest.mark.parametrize("algorithm", ["Kumo", "TabICLv2"])
@pytest.mark.parametrize("task", ["binary_classification", "multiclass_classification", "regression"])
def test_cv_scores_keep_fold_contexts_and_inference_uses_all_rows(tmp_path, fake_network, algorithm, task):
    X = pd.DataFrame(np.random.default_rng(11).normal(size=(90, 3)), columns=list("abc"))
    y = (np.arange(90) / 10 if task == "regression" else
         np.array(["class-a", "class-b", "class-c"] if task.startswith("multi") else ["no", "yes"])[
             np.arange(90) % (3 if task.startswith("multi") else 2)])
    root = tmp_path / "run"
    validation = dict(validation_type="kfold", k_folds=3, shuffle=True,
                      stratify=task != "regression", repeats=2)
    model = AutoML(**options(root, algorithm, task, validation)).fit(X.copy(), y)
    assert model._models
    for fitted in model._models:
        assert len(fitted.learners) == len(fitted.preprocessings) == 1
        folder = root / fitted.get_name()
        desc = json.loads((folder / "framework.json").read_text())
        assert "validation_models" in (folder / "README.md").read_text()
        assert len(desc["validation_models"]["learners"]) == 6
        assert desc["tfm_full_context"]["context_rows"] == 90
        assert desc["learners"][0]["training_info"]["validation_score"] is None
        assert desc["learners"][0]["training_info"]["context_rows"] == 90
        oof = fitted.get_out_of_folds()
        values = []
        for k, (learner_desc, path, prep_desc) in enumerate(zip(
            desc["validation_models"]["learners"], desc["validation_models"]["saved"],
            desc["validation_models"]["preprocessing"],
        )):
            repeat, fold = divmod(k, 3)
            ix = np.load(root / f"folds/fold_{fold}_repeat_{repeat}_validation_indices.npy")
            learner = AlgorithmFactory.load(learner_desc, str(root / path), lazy_load=True)
            prep = Preprocessing(); prep.from_json(prep_desc, str(root))
            transformed, _, _ = prep.transform(X.iloc[ix].copy(), None)
            p = prep.inverse_scale_target(learner.predict(transformed))
            values.append(pd.DataFrame(np.asarray(p).reshape(len(ix), -1), index=ix))
            assert len(learner.context) == 60
        expected = pd.concat(values).groupby(level=0).mean().sort_index()
        columns = [c for c in oof if "prediction" in c]
        np.testing.assert_allclose(oof[columns], expected, rtol=1e-6, atol=1e-6)
        assert len(oof) == 90
    fake_network.clear()
    actual = model.predict_all(X.iloc[:7])
    assert fake_network == [("learner_full", 90, 7)]
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    loaded = AutoML(results_path=str(moved), verbose=0)
    fake_network.clear()
    pd.testing.assert_frame_equal(actual, loaded.predict_all(X.iloc[:7]))
    assert fake_network == [("learner_full", 90, 7)]


def test_holdout_keeps_its_training_context(tmp_path, fake_network):
    X = pd.DataFrame(np.random.default_rng(3).normal(size=(100, 3)))
    model = AutoML(**options(tmp_path / "split", "Kumo", "binary_classification",
                            dict(validation_type="split", train_ratio=0.8, stratify=True)))
    model.fit(X.copy(), np.arange(100) % 2)
    for fitted in model._models:
        assert fitted._tfm_full_context is None
        assert fitted.learners[0].training_info["context_rows"] == 80


def test_full_row_limit_rejects_before_cv(tmp_path, fake_network, monkeypatch):
    def check(self, rows):
        if rows > 70:
            raise ResourceAdmissionError("test full context too large", {"rows": rows, "reason": "training_rows"})
    monkeypatch.setattr(sdm.SDMAlgorithm, "_check_training_size", check)
    X = pd.DataFrame(np.random.default_rng(4).normal(size=(90, 3)))
    model = AutoML(**options(tmp_path / "limit", "Kumo", "binary_classification",
                            dict(validation_type="kfold", k_folds=3, stratify=True)))
    with pytest.raises(Exception, match="No models produced"):
        model.fit(X, np.arange(90) % 2)
    assert fake_network == []


def test_stacks_use_oof_for_training_and_full_context_for_prediction(tmp_path, fake_network, monkeypatch):
    from supervised.algorithms import lightgbm
    monkeypatch.setitem(lightgbm.additional, "max_rounds", 10)
    X = pd.DataFrame(np.random.default_rng(6).normal(size=(150, 3)))
    params = options(tmp_path / "stack", "Kumo", "binary_classification",
                     dict(validation_type="kfold", k_folds=3, stratify=True))
    params.update(algorithms=["Kumo", "TabICLv2", "LightGBM"], start_random_models=3,
                  stack_models=True, train_ensemble=True,
                  custom_params={"kumo": {"device": "cpu"}, "tabiclv2": {"device": "cpu"}})
    model = AutoML(**params).fit(X.copy(), np.arange(150) % 2)
    assert any(m._is_stacked and m.get_type() == "LightGBM" for m in model._models)
    assert not any(m._is_stacked and m.get_type() in ("Kumo", "TabICLv2") for m in model._models)
    assert {"Kumo", "TabICLv2"}.issubset({m.get_type() for m in model._stacked_models})
    training = model.get_stacked_data(X.copy())
    for m in model._stacked_models:
        if m.get_type() in ("Kumo", "TabICLv2"):
            np.testing.assert_allclose(training[m.get_name() + "_prediction"],
                                       m.get_out_of_folds()["prediction"])
    fake_network.clear()
    stacked = next(m for m in model._models if m._is_stacked and m.get_type() == "LightGBM")
    prediction = model._base_predict(X.iloc[:7], model=stacked)
    assert len(prediction) == 7
    assert fake_network and all(name == "learner_full" and rows == 150 for name, rows, _ in fake_network)


def test_feature_discovery_is_prepared_separately_on_full_data(tmp_path, fake_network, monkeypatch):
    pytest.importorskip("features_goldmine")
    from supervised.preprocessing.feature_discovery_transformer import FeatureDiscovery
    from supervised.tuner.time_controller import TimeController
    from sklearn.datasets import make_classification

    full_calls = []
    original = FeatureDiscovery.prepare
    def prepare(self, validation_params, time_limit, progress=None, **kwargs):
        result = original(self, validation_params, min(time_limit, 8), progress, **kwargs)
        if kwargs.get("full_data"):
            full_calls.append(result)
        return result
    monkeypatch.setattr(FeatureDiscovery, "prepare", prepare)
    old_time = TimeController.time_should_use
    monkeypatch.setattr(TimeController, "time_should_use",
                        lambda self, step: 16 if step == "feature_discovery" else old_time(self, step))
    X, y = make_classification(n_samples=150, n_features=4, n_informative=3,
                               n_redundant=0, random_state=13)
    X = pd.DataFrame(X, columns=list("abcd"))
    params = options(tmp_path / "discovery", "TabICLv2", "binary_classification",
                     dict(validation_type="kfold", k_folds=2, shuffle=True, stratify=True))
    params.update(feature_discovery=True, total_time_limit=180)
    model = AutoML(**params).fit(X.copy(), y)
    discovered = [m for m in model._models if "FeatureDiscovery" in m.get_name()]
    assert discovered and full_calls
    for m in discovered:
        full = m.preprocessings[0]._feature_discovery
        fold = m._validation_models["preprocessing"][0]["feature_discovery"]
        assert full.info["directory"] != fold["info"]["directory"]
        assert full.info["generator_id"] != fold["info"]["generator_id"]
        assert full.info["identity_schema_version"] == fold["info"]["identity_schema_version"] == 1
        assert m._tfm_full_context["context_rows"] == 150
        assert len(m.predict(X.iloc[:5])) == 5
        from supervised.model_framework import ModelFramework
        restored = ModelFramework.load(str(tmp_path / "discovery"), m.get_name())
        assert restored.preprocessings[0]._feature_discovery.to_json() == full.to_json()
        assert restored._validation_models["preprocessing"] == m._validation_models["preprocessing"]
        expected_columns = list(full.transform(X.iloc[:5]))
        assert list(restored.preprocessings[0]._feature_discovery.transform(X.iloc[:5])) == expected_columns
        np.testing.assert_allclose(restored.predict(X.iloc[:5]), m.predict(X.iloc[:5]))


def test_full_context_refuses_generated_drop_before_discovery(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from supervised.tfm_full_context import TFMFullContext
    from supervised.exceptions import AutoMLException
    from supervised.preprocessing.feature_discovery_transformer import FeatureDiscovery

    obj = TFMFullContext.__new__(TFMFullContext)
    obj.framework = SimpleNamespace(
        learner_params={}, params={}, _explain_level=0, _data_store=None,
        validation_params={"X_path": "X.data"},
        preprocessing_params={"drop_features": ["feature_discovery_3"]},
    )
    obj.remaining = lambda: 100
    monkeypatch.setattr("supervised.tfm_full_context.data_row_count", lambda *a, **k: 20)
    monkeypatch.setattr(AlgorithmFactory, "get_algorithm", lambda p: SimpleNamespace(_check_training_size=lambda n: None))
    monkeypatch.setattr(FeatureDiscovery, "prepare", lambda *a, **k: pytest.fail("must reject before preparing another generator"))
    with pytest.raises(AutoMLException, match="generated features"):
        obj.prepare()
