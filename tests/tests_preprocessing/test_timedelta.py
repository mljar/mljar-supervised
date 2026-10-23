import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from supervised import AutoML
from supervised.preprocessing.preprocessing import Preprocessing
from supervised.preprocessing.preprocessing_missing import PreprocessingMissingValues
from supervised.preprocessing.scale import Scale
from supervised.tuner.data_info import DataInfo


@pytest.mark.parametrize("resolution", ["s", "ms", "us", "ns"])
def test_durations_become_seconds_in_fit_transform_and_reload(resolution):
    X = pd.DataFrame(
        {
            "elapsed": pd.to_timedelta([-2, 0.5, 3, None], unit="s").astype(
                f"timedelta64[{resolution}]"
            ),
            "number": [10.0, 20.0, 30.0, 40.0],
            "category": ["a", "b", "a", "b"],
        },
        index=[10, 20, 30, 40],
    )
    original = X.copy(deep=True)
    p = Preprocessing()
    fitted, _, _ = p.fit_and_transform(X, None)
    assert_allclose(fitted.elapsed, X.elapsed.dt.total_seconds(), equal_nan=True)
    pd.testing.assert_series_equal(fitted.number, X.number)
    pd.testing.assert_series_equal(fitted.category, X.category)
    pd.testing.assert_frame_equal(X, original)

    saved = json.loads(json.dumps(p.to_json()))
    assert saved["timedelta_columns"] == ["elapsed"]
    loaded = Preprocessing()
    loaded.from_json(saved, "")
    for transformer in [p, loaded]:
        transformed, _, _ = transformer.transform(X, None)
        pd.testing.assert_frame_equal(transformed, fitted)
    pd.testing.assert_frame_equal(X, original)


def test_duration_conversion_precedes_imputation_and_scaling():
    X = pd.DataFrame({"elapsed": pd.to_timedelta([-2, 0.5, 3, None], unit="s")})
    numeric = X.assign(elapsed=X.elapsed.dt.total_seconds())
    params = {
        "columns_preprocessing": {
            "elapsed": [PreprocessingMissingValues.FILL_NA_MEDIAN, Scale.SCALE_NORMAL]
        }
    }
    p = Preprocessing(params)
    fitted, _, _ = p.fit_and_transform(X, None)
    expected, _, _ = Preprocessing(params).fit_and_transform(numeric, None)
    pd.testing.assert_frame_equal(fitted, expected)
    loaded = Preprocessing()
    loaded.from_json(json.loads(json.dumps(p.to_json())), "")
    transformed, _, _ = loaded.transform(X, None)
    pd.testing.assert_frame_equal(transformed, expected)
    assert not transformed.isna().any().any()


def test_prediction_duration_resolution_can_differ_from_training():
    train = pd.DataFrame(
        {"elapsed": pd.to_timedelta([1000, 2000], unit="ms").astype("timedelta64[ms]")}
    )
    validation = pd.DataFrame(
        {
            "elapsed": pd.to_timedelta([-0.5, 1.25, None], unit="s").astype(
                "timedelta64[ns]"
            )
        }
    )
    p = Preprocessing()
    p.fit_and_transform(train, None)
    loaded = Preprocessing()
    loaded.from_json(json.loads(json.dumps(p.to_json())), "")
    transformed, _, _ = loaded.transform(validation, None)
    assert_allclose(transformed.elapsed, [-0.5, 1.25, np.nan], equal_nan=True)


def test_duration_data_info_matches_numeric_seconds():
    X = pd.DataFrame(
        {
            "elapsed": pd.to_timedelta([-2, 0.5, 3, None], unit="s"),
            "constant": pd.to_timedelta([1, 1, 1, 1], unit="s"),
            "empty": pd.Series([pd.NaT] * 4, dtype="timedelta64[ns]"),
        }
    )
    before = X.copy(deep=True)
    numeric = X.apply(lambda col: col.dt.total_seconds())
    y = pd.Series([0, 1, 0, 1])
    info = DataInfo.compute(X, y, "binary_classification")
    assert info == DataInfo.compute(numeric, y, "binary_classification")
    assert "categorical" not in info["columns_info"]["elapsed"]
    assert info["columns_info"]["constant"] == ["constant_column"]
    assert info["columns_info"]["empty"] == ["empty_column"]
    pd.testing.assert_frame_equal(X, before)


def test_legacy_preprocessing_metadata_without_durations():
    X = pd.DataFrame({"number": [1.0, 2.0]})
    p = Preprocessing()
    fitted, _, _ = p.fit_and_transform(X, None)
    assert "timedelta_columns" not in p.to_json()
    loaded = Preprocessing()
    loaded.from_json(p.to_json(), "")
    transformed, _, _ = loaded.transform(X, None)
    pd.testing.assert_frame_equal(transformed, fitted)


@pytest.mark.parametrize("task", ["regression", "binary_classification"])
def test_automl_duration_training_prediction_and_reload(tmp_path, task):
    seconds = np.linspace(-10, 10, 60)
    X = pd.DataFrame({"elapsed": pd.to_timedelta(seconds, unit="s")})
    X.loc[5, "elapsed"] = pd.NaT
    before = X.copy(deep=True)
    y = pd.Series(seconds if task == "regression" else (seconds > 0).astype(int))
    model = AutoML(
        results_path=str(tmp_path / "models"),
        algorithms=["Linear", "Decision Tree"],
        ml_task=task,
        mode="Explain",
        explain_level=0,
        train_ensemble=False,
        n_jobs=1,
        verbose=0,
    )
    model.fit(X, y)
    assert set(model.get_leaderboard().model_type) == {"Linear", "Decision Tree"}
    predictions = model.predict(X)
    loaded = AutoML(results_path=str(tmp_path / "models"))
    assert_allclose(loaded.predict(X), predictions)
    assert_allclose(
        model.predict(X.assign(elapsed=X.elapsed.dt.total_seconds())), predictions
    )
    pd.testing.assert_frame_equal(X, before)
