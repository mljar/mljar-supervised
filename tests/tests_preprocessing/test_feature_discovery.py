import json
import shutil
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.datasets import make_friedman1

from supervised.exceptions import AutoMLException
from supervised.preprocessing.feature_discovery_transformer import (
    FeatureDiscovery,
    FeatureDiscoveryTransformer,
    check_dependencies,
)
from supervised.preprocessing.preprocessing import Preprocessing
from supervised.preprocessing.scale import Scale
from supervised.utils.utils import Store, dump_data
from supervised.validation.validation_step import ValidationStep


def discovery_data(tmp_path):
    X, y = make_friedman1(n_samples=600, n_features=6, random_state=42)
    X, y = pd.DataFrame(X, columns=list("abcdef")), pd.Series(y, name="target")
    X["category"] = pd.cut(X.a, 4).astype(str)
    store = Store()
    x_path, y_path = str(tmp_path / "X.data"), str(tmp_path / "y.data")
    dump_data(x_path, X, store=store)
    dump_data(y_path, y.to_frame(), store=store)
    params = {
        "validation_type": "kfold",
        "k_folds": 2,
        "shuffle": True,
        "random_seed": 42,
        "results_path": str(tmp_path),
        "X_path": x_path,
        "y_path": y_path,
    }
    return X, y, store, params


@pytest.fixture
def prepared(tmp_path):
    pytest.importorskip("features_goldmine")
    X, y, store, params = discovery_data(tmp_path)
    original = check_dependencies()
    seen = []

    fit_transform = original.fit_transform

    def observed(self, X, y, **kwargs):
        seen.append((X.copy(), y.copy()))
        return fit_transform(self, X, y, **kwargs)

    discovery = FeatureDiscovery(tmp_path, "regression", 42, 1, store)
    with patch.object(original, "fit_transform", observed):
        config = discovery.prepare(params, 30)
    assert config and 1 <= len(config["profiles"]) <= 3
    return X, y, store, params, config, seen, discovery


def test_fold_isolation_cache_and_frozen_replay(tmp_path, prepared):
    X, y, store, params, config, seen, discovery = prepared
    validation = ValidationStep(params, store)
    assert len(seen) == 2
    for fold, (fit_X, fit_y) in enumerate(seen):
        train, valid = validation.get_split(fold)
        assert fit_X.index.equals(train["X"].index)
        assert not set(fit_X.index) & set(valid["X"].index)
        np.testing.assert_array_equal(fit_y, train["y"])
        info = config["folds"][f"0:{fold}"]
        adapter = FeatureDiscoveryTransformer(info, "default", tmp_path)
        training = adapter.fit_transform(train["X"], train["y"])
        assert training.index.equals(train["X"].index)
        stacked = train["X"].assign(stacking_prediction=np.arange(len(train["X"])))
        assert_frame_equal(training, adapter.fit_transform(stacked, train["y"]))
        expected = adapter.transform(valid["X"])
        restored = FeatureDiscoveryTransformer.from_json(adapter.to_json(), tmp_path)
        assert_frame_equal(expected, restored.transform(valid["X"]))
        unseen = valid["X"].copy()
        unseen["category"] = "unseen category"
        unseen.iloc[0, 0] = np.nan
        assert np.isfinite(restored.transform(unseen).to_numpy()).all()
        with pytest.raises(AutoMLException, match="changed"):
            adapter.fit_transform(train["X"], train["y"] + 1)
        with pytest.raises(AutoMLException, match="changed"):
            adapter.fit_transform(train["X"].iloc[::-1], train["y"].iloc[::-1])
    # Discovery is shared, including across a new manager (resume).
    with patch.object(
        check_dependencies(), "fit_transform", side_effect=AssertionError("refit")
    ):
        assert discovery.prepare(params, 30) == config
        assert (
            FeatureDiscovery(tmp_path, "regression", 42, 1, store).prepare(params, 30)
            == config
        )
    # Changed targets require different artifacts, never a cache hit.
    dump_data(params["y_path"], (y + 100).to_frame(), store=store)
    changed = discovery.prepare(params, 30)
    assert changed["folds"]["0:0"]["directory"] != config["folds"]["0:0"]["directory"]


def test_preprocessing_scaled_target_and_portable_artifacts(tmp_path, prepared):
    _, _, store, params, config, _, _ = prepared
    train, valid = ValidationStep(params, store).get_split(0)
    options = {
        "target_preprocessing": [Scale.SCALE_NORMAL],
        "columns_preprocessing": {},
        "feature_discovery": {**config, "feature_set": "default", "scale": True},
    }
    preprocessing = Preprocessing(options, "model", 0, 0, results_path=tmp_path)
    augmented, target, _ = preprocessing.fit_and_transform(train["X"], train["y"])
    assert abs(target.mean()) < 1e-10
    added = preprocessing._feature_discovery._new_columns
    np.testing.assert_allclose(augmented[added].mean(), 0, atol=1e-10)
    expected, _, _ = preprocessing.transform(valid["X"], valid["y"])
    moved = tmp_path / "moved"
    shutil.copytree(tmp_path / "feature_discovery", moved / "feature_discovery")
    for path in moved.rglob("training.joblib"):
        path.unlink()
    restored = Preprocessing()
    restored.from_json(json.loads(json.dumps(preprocessing.to_json())), moved)
    actual, _, _ = restored.transform(valid["X"], valid["y"])
    assert_frame_equal(expected, actual)
    assert "generator.md" in preprocessing._feature_discovery.markdown()


def test_training_encoding_cache_is_not_replaced_by_inference(tmp_path):
    # A deliberately different training matrix proves the adapter loads training
    # outputs instead of reconstructing them with transform(training_rows).
    pytest.importorskip("features_goldmine")
    _, _, store, params = discovery_data(tmp_path)
    discovery = FeatureDiscovery(tmp_path, "regression", 42, 1, store)
    config = discovery.prepare(params, 30)
    train, _ = ValidationStep(params, store).get_split(0)
    info = config["folds"]["0:0"]
    path = tmp_path / info["directory"] / "training.joblib"
    matrix = joblib.load(path)
    matrix.iloc[:, :] = -123.0
    joblib.dump(matrix, path)
    adapter = FeatureDiscoveryTransformer(info, "default", tmp_path)
    assert (adapter.fit_transform(train["X"], train["y"]) == -123).all().all()
    assert not (adapter.transform(train["X"]) == -123).all().all()


def test_unavailable_profiles_and_expired_budget(tmp_path, monkeypatch):
    _, _, store, params = discovery_data(tmp_path)

    class FakeGoldmine:
        calls = 0

        def __init__(self, **kwargs):
            assert kwargs["max_feature_sets"] == 3

        def fit_transform(self, X, y, **kwargs):
            FakeGoldmine.calls += 1
            sets = {"default": pd.DataFrame({"a": X.a}, index=X.index)}
            if self.calls == 1:
                sets["compact"] = pd.DataFrame({"b": X.b}, index=X.index)
            return sets

        def save(self, path, report):
            path.touch()

    monkeypatch.setattr(
        "supervised.preprocessing.feature_discovery_transformer.check_dependencies",
        lambda: FakeGoldmine,
    )
    monkeypatch.setattr("supervised.preprocessing.feature_discovery_transformer.importlib.metadata.version", lambda name: "1.1.0")
    discovery = FeatureDiscovery(tmp_path, "regression", 42, 1, store)
    assert discovery.prepare(params, 0) is None
    assert FakeGoldmine.calls == 0
    assert discovery.prepare(params, 10)["profiles"] == ["default"]
