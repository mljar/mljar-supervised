"""Schema compatibility against the implementation before the performance fix."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from supervised.apps import metadata
from tests.tests_automl import _app_metadata_reference as reference


CASES = [
    pd.Series(np.random.default_rng(7).normal(size=1000)),
    pd.Series(np.random.default_rng(7).normal(size=1000), dtype="float32"),
    pd.Series([1, 2, 3, None], dtype="Int64"),
    pd.Series([1.5, None, 3.25], dtype="Float64"),
    pd.Series([True, False, None], dtype="boolean"),
    pd.Series(["yes", "no", None]),
    pd.Series(["yes", "unknown", None]),
    pd.Series(["a", "b", "a", None], dtype="category"),
    pd.Series(["a", "b", "c", None], dtype="string"),
    pd.Series(pd.date_range("2020-01-01", periods=4, tz="UTC")),
    pd.Series([None, None], dtype="object"),
    pd.Series([np.nan, np.nan]),
    pd.Series([], dtype="float64"),
    pd.Series([0.0, 1.0, np.nan]),
    pd.Series([0.0] * 50 + [1.0, 2.0]),
    pd.Series([1.0] * 50 + list(range(20))),
    pd.Series([2.0] * 100),
    pd.Series([1.0, np.inf, -np.inf]),
    pd.Series([1e12 + 0.1, 1e12 + 0.2, 1e12 + 0.3]),
    pd.Series([1.0] * 50 + [0.25]),
    pd.Series(["1", "2.5", "invalid"]),
    pd.Series([1 + 2j, 3 + 4j]),
]


@pytest.mark.parametrize("series", CASES)
@pytest.mark.parametrize(
    "info",
    [
        [],
        ["categorical"],
        ["text_transform"],
        ["datetime_transform"],
        ["missing_values"],
    ],
)
def test_feature_schema_matches_reference(series, info):
    X = pd.DataFrame({"feature": series})
    original = X.copy(deep=True)
    automl = SimpleNamespace(
        _data_info={"columns": ["feature", "absent"], "columns_info": {"feature": info}}
    )
    args = (automl, X, {"absent": ["fallback"]})
    assert metadata._build_feature_schema(
        *args, tolerate_errors=True
    ) == reference._build_feature_schema(*args, tolerate_errors=True)
    pd.testing.assert_frame_equal(X, original)


@pytest.mark.parametrize("order", ["C", "F"])
def test_wide_schema_matches_reference_and_does_not_mutate(order):
    values = np.array(np.random.default_rng(9).normal(size=(1000, 25)), order=order)
    values[::11, 0] = np.nan
    X = pd.DataFrame(values, columns=[str(i) for i in range(25)])
    original = X.copy(deep=True)
    automl = SimpleNamespace(_data_info={"columns": list(X), "columns_info": {}})
    assert metadata._build_feature_schema(automl, X) == reference._build_feature_schema(
        automl, X
    )
    pd.testing.assert_frame_equal(X, original)


def test_saved_schema_uses_resampled_data_after_cleanup_and_reload(
    tmp_path, monkeypatch
):
    from supervised import AutoML
    from tests.tests_automl.test_training_store_lifecycle import automl, assert_released

    X = pd.DataFrame({"feature": np.arange(100, dtype=float)})
    y = np.array([0] * 99 + [1])
    expected = []
    original = metadata._build_feature_schema

    def inspect_resampled(model, frame, *args, **kwargs):
        assert len(frame) == 119
        schema = reference._build_feature_schema(model, frame, *args, **kwargs)
        expected.append(schema)
        return original(model, frame, *args, **kwargs)

    monkeypatch.setattr(metadata, "_build_feature_schema", inspect_resampled)
    model = automl(tmp_path / "imbalanced", ml_task="binary_classification")
    model.fit(X, y)
    assert_released(model)
    assert len(expected) == 1
    assert model._data_info["app_feature_schema"] == expected[0]
    assert sum(expected[0][0]["distribution"]["counts"]) == 119
    loaded = AutoML(results_path=str(tmp_path / "imbalanced"))
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
    assert metadata.collect_app_metadata(loaded)["feature_schema"] == expected[0]
