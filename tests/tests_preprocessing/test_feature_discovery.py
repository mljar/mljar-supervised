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
    discovery_input,
    fingerprint,
)
from supervised.preprocessing.preprocessing import Preprocessing
from supervised.preprocessing.scale import Scale
from supervised.utils.utils import Store, dump_data
from supervised.validation.validation_step import ValidationStep


def mixed_frame():
    return pd.DataFrame(
        {
            "number": np.arange(6, dtype=np.int64),
            "category": pd.Categorical(["a", "b", "c", "a", "b", "c"]),
            "small": np.arange(6, dtype=np.uint8),
            "value": np.arange(6, dtype=float),
            "flag": pd.Categorical(["yes", "no"] * 3),
        },
        index=[11, 13, 17, 19, 23, 29],
    )


def test_fingerprint_ignores_dataframe_storage_layout(tmp_path):
    X = mixed_frame()
    y = pd.Series([0, 1] * 3, index=X.index, name="target")
    selected = X[list(X.columns)]
    assert_frame_equal(X, selected)
    expected = fingerprint(X, y)
    assert fingerprint(selected, y.copy(deep=False)) == expected
    assert fingerprint(pd.concat([X[[c]] for c in X], axis=1), y) == expected
    path = tmp_path / "data.joblib"
    joblib.dump((selected, y), path)
    assert fingerprint(*joblib.load(path)) == expected


@pytest.mark.parametrize(
    "change", ["value", "target", "rows", "index", "dtype", "categories", "columns"]
)
def test_fingerprint_rejects_changed_data_and_schema(change):
    X = mixed_frame()
    y = pd.Series([0, 1] * 3, index=X.index, name="target")
    expected = fingerprint(X, y)
    if change == "value":
        X.iloc[0, 0] += 1
    elif change == "target":
        y.iloc[0] = 1
    elif change == "rows":
        X, y = X.iloc[::-1], y.iloc[::-1]
    elif change == "index":
        X.index = range(len(X))
    elif change == "dtype":
        X["small"] = X.small.astype(np.int64)
    elif change == "categories":
        X["category"] = X.category.cat.reorder_categories(["c", "b", "a"])
    else:
        X = X[list(reversed(X.columns))]
    assert fingerprint(X, y) != expected


def test_cached_training_features_accept_equivalent_mixed_frame(tmp_path):
    X = mixed_frame()
    y = pd.Series([0, 1] * 3, index=X.index, name="target")
    values = pd.DataFrame({"encoded": np.arange(len(X))}, index=X.index)
    joblib.dump(values, tmp_path / "training.joblib")
    info = dict(
        fingerprint=fingerprint(X, y),
        fingerprint_version=2,
        input_columns=list(X),
        directory=".",
        sets={"default": ["encoded"]},
        rename={"encoded": "new_feature"},
    )
    adapter = FeatureDiscoveryTransformer(info, "default", tmp_path)
    actual = adapter.fit_transform(X[list(X)], y.copy(deep=False))
    assert_frame_equal(actual, values.rename(columns=info["rename"]))
    with pytest.raises(AutoMLException, match="changed"):
        adapter.fit_transform(X, y + 1)
    # RC3 manifests omit the version. Keep their original hash interpretation.
    legacy = {**info, "fingerprint": fingerprint(discovery_input(X), y, version=1)}
    legacy.pop("fingerprint_version")
    assert_frame_equal(
        actual,
        FeatureDiscoveryTransformer(legacy, "default", tmp_path).fit_transform(X, y),
    )


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
    monkeypatch.setattr(
        "supervised.preprocessing.feature_discovery_transformer.importlib.metadata.version",
        lambda name: "1.1.0",
    )
    discovery = FeatureDiscovery(tmp_path, "regression", 42, 1, store)
    assert discovery.prepare(params, 0) is None
    assert FakeGoldmine.calls == 0
    assert discovery.prepare(params, 10)["profiles"] == ["default"]
