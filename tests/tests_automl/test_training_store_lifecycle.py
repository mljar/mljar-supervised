import numpy as np
import pandas as pd
import pytest

from supervised import AutoML
from supervised.apps.metadata import _build_feature_schema, collect_app_metadata
from supervised.base_automl import BaseAutoML
from supervised.utils.utils import dump_data


def data(rows=100):
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(rows, 3)), columns=["a", "b", "c"])
    return X, X.a.to_numpy() + rng.normal(size=rows) * 0.1


def automl(path, **kwargs):
    params = dict(
        results_path=str(path),
        ml_task="regression",
        algorithms=["Baseline"],
        explain_level=0,
        train_ensemble=False,
        stack_models=False,
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        boost_on_errors=False,
        mix_encoding=False,
        start_random_models=1,
        hill_climbing_steps=0,
        n_jobs=2,
        verbose=0,
        total_time_limit=120,
        validation_strategy={"validation_type": "split", "train_ratio": 0.8},
    )
    params.update(kwargs)
    return AutoML(**params)


def assert_released(model):
    assert model._training_store is None
    if model.tuner is not None:
        assert len(model.tuner._data_store) == 0
    for member in model._models:
        if hasattr(member, "_data_store") and member._data_store is not None:
            assert len(member._data_store) == 0


def test_completed_fit_reload_and_app_metadata(tmp_path):
    X, y = data()
    X["category"] = ["a", "b"] * (len(X) // 2)
    model = automl(tmp_path / "run")
    model.fit(X.copy(), y)
    assert_released(model)
    expected = _build_feature_schema(model, X, model._data_info["columns_values"])
    assert collect_app_metadata(model)["feature_schema"] == expected
    reloaded = AutoML(results_path=str(tmp_path / "run"))
    np.testing.assert_allclose(model.predict(X), reloaded.predict(X))
    assert collect_app_metadata(reloaded)["feature_schema"] == expected
    # Runtime dependencies must never enter serialized training parameters.
    assert "data_store" not in (tmp_path / "run" / "params.json").read_text()
    assert (
        "data_store"
        not in (tmp_path / "run" / "1_Baseline" / "framework.json").read_text()
    )


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_preparation_failure_cleans_store(tmp_path, monkeypatch, exception):
    X, y = data()
    original = BaseAutoML._save_data
    seen = []

    def fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        seen.append(self._training_store)
        assert len(seen[-1]) == 2
        raise exception("injected")

    monkeypatch.setattr(BaseAutoML, "_save_data", fail)
    model = automl(tmp_path / "failure")
    with pytest.raises(exception, match="injected"):
        model.fit(X, y)
    assert_released(model)
    assert len(seen[0]) == 0


def test_cleanup_does_not_touch_another_active_fit(tmp_path, monkeypatch):
    X, y = data()
    original = BaseAutoML._save_data
    outer = automl(tmp_path / "outer")
    inner = automl(tmp_path / "inner")

    def nested(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self is outer:
            outer_store = self._training_store
            inner.fit(X.copy(), y.copy())
            assert_released(inner)
            assert len(outer_store) == 2
            assert outer_store.get(self._X_path).shape == X.shape

    monkeypatch.setattr(BaseAutoML, "_save_data", nested)
    outer.fit(X, y)
    assert_released(outer)
    np.testing.assert_allclose(outer.predict(X), inner.predict(X))


def test_partial_training_releases_store_and_keeps_completed_model(
    tmp_path, monkeypatch
):
    X, y = data()
    original = BaseAutoML.train_model

    def fail_one(self, params):
        if params["learner"]["model_type"] == "Decision Tree":
            raise RuntimeError("injected learner failure")
        return original(self, params)

    monkeypatch.setattr(BaseAutoML, "train_model", fail_one)
    model = automl(tmp_path / "partial", algorithms=["Baseline", "Decision Tree"])
    model.fit(X, y)
    assert_released(model)
    assert (tmp_path / "partial" / "errors.md").exists()
    assert np.isfinite(model.predict(X)).all()


def test_resume_recovers_checkpointed_generated_weights(tmp_path, monkeypatch):
    X, y = data()
    original = BaseAutoML.train_model
    path = tmp_path / "resume"

    def interrupt(self, params):
        weight_path = str(path / "generated_sample_weight.data")
        dump_data(
            weight_path,
            pd.DataFrame({"sample_weight": np.linspace(0.9, 1.1, len(y))}),
            store=self._training_store,
            checkpoint=True,
        )
        params["validation_strategy"]["sample_weight_path"] = weight_path
        self.save_progress(self._fit_level, [params])
        raise KeyboardInterrupt("checkpointed")

    monkeypatch.setattr(BaseAutoML, "train_model", interrupt)
    first = automl(path)
    with pytest.raises(KeyboardInterrupt):
        first.fit(X.copy(), y)
    assert_released(first)
    monkeypatch.setattr(BaseAutoML, "train_model", original)
    resumed = automl(path)
    resumed.fit(X.copy(), y)
    assert_released(resumed)
    control = automl(tmp_path / "control")
    control.fit(X.copy(), y, sample_weight=np.linspace(0.9, 1.1, len(y)))
    np.testing.assert_allclose(resumed.predict(X), control.predict(X))


@pytest.mark.parametrize("same_instance", [False, True])
def test_interrupted_stacking_rebuilds_training_data_on_resume(
    tmp_path, monkeypatch, same_instance
):
    X, y = data(120)
    options = dict(
        algorithms=[
            "Baseline",
            "Linear",
            "Decision Tree",
            "Random Forest",
            "Extra Trees",
        ],
        stack_models=True,
        total_time_limit=600,
        validation_strategy={"validation_type": "kfold", "k_folds": 2, "shuffle": True},
    )
    original = BaseAutoML.train_model
    seen = []

    def interrupt(self, params):
        if params.get("is_stacked"):
            assert self._training_store.contains(
                str(tmp_path / "stack" / "X_stacked.data")
            )
            seen.append(True)
            raise KeyboardInterrupt("stacking")
        return original(self, params)

    monkeypatch.setattr(BaseAutoML, "train_model", interrupt)
    first = automl(tmp_path / "stack", **options)
    with pytest.raises(KeyboardInterrupt, match="stacking"):
        first.fit(X.copy(), y)
    assert seen
    assert_released(first)
    monkeypatch.setattr(BaseAutoML, "train_model", original)
    # A stale disk artifact must not be mistaken for an in-memory frame.
    (tmp_path / "stack" / "X_stacked.data").write_bytes(b"stale")
    resumed = first if same_instance else automl(tmp_path / "stack", **options)
    resumed.fit(X.copy(), y)
    names = [m.get_name() for m in resumed._models]
    assert len(names) == len(set(names))
    assert any(m._is_stacked for m in resumed._models)
    assert_released(resumed)
    reloaded = AutoML(results_path=str(tmp_path / "stack"))
    for m in resumed._models:
        if m._is_stacked:
            assert np.isfinite(
                resumed._base_predict(X.copy(), model=m)["prediction"]
            ).all()
    np.testing.assert_allclose(resumed.predict(X.copy()), reloaded.predict(X.copy()))


def test_custom_validation_with_weights_and_sensitive_features(tmp_path):
    from supervised.utils.utils import Store, dump_data
    from supervised.validation.validation_step import ValidationStep
    import joblib

    X, y = data()
    store = Store()
    values = {
        "X": X,
        "y": pd.DataFrame({"target": y}),
        "sample_weight": pd.DataFrame({"sample_weight": np.arange(100) + 1}),
        "sensitive_features": pd.DataFrame({"group": ["a", "b"] * 50}),
    }
    params = {"validation_type": "custom", "results_path": str(tmp_path)}
    for name, frame in values.items():
        path = str(tmp_path / f"{name}.data")
        dump_data(path, frame, store=store)
        params[f"{name}_path"] = path
    cv_path = str(tmp_path / "cv.data")
    train, valid = np.arange(80), np.arange(80, 100)
    joblib.dump([(train, valid)], cv_path)
    params["cv_path"] = cv_path
    a, b = ValidationStep(params, data_store=store).get_split(0)
    pd.testing.assert_frame_equal(a["X"], X.iloc[train])
    pd.testing.assert_frame_equal(
        b["sensitive_features"], values["sensitive_features"].iloc[valid]
    )
    np.testing.assert_array_equal(b["sample_weight"], np.arange(81, 101))
    a["X"].iloc[0, 0] = 999
    pd.testing.assert_frame_equal(store.get(params["X_path"]), X)


def test_optional_app_statistics_do_not_break_training(tmp_path):
    X, y = data()
    X.iloc[0, 0] = np.inf
    model = automl(tmp_path / "infinite")
    model.fit(X, y)
    assert_released(model)
    assert np.isfinite(model.predict(X)).all()


@pytest.mark.parametrize(
    "values",
    [
        [1, 2, 3],
        [1.0, 2.0],
        [0.12, 0.01],
        [0.12, 0.00001, 3.2],
        [0.12345, 0.123456789],
        [np.nan, 1.2345],
        [np.inf, 0.12345],
    ],
)
def test_app_numeric_step_preserves_original_result(values):
    import math
    from supervised.apps.metadata import _numeric_step, _detect_numeric_dtype

    series = pd.Series(values)
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if _detect_numeric_dtype(clean) == "int":
        expected = 1
    else:
        places = 0
        for value in clean.tolist():
            if math.isfinite(value):
                text = f"{float(value):.10f}".rstrip("0").rstrip(".")
                if "." in text:
                    places = max(places, len(text.split(".")[1]))
        expected = 1 if places == 0 else float(10 ** (-min(places, 4)))
    assert _numeric_step(series) == expected
