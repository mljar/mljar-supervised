import copy
import json

import numpy as np
import pandas as pd
import pytest

from supervised.preprocessing.preprocessing import Preprocessing
from supervised.preprocessing.preprocessing_categorical import PreprocessingCategorical
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.tuner.data_info import DataInfo
from supervised.tuner.preprocessing_tuner import PreprocessingTuner


@pytest.mark.parametrize("dtype", [object, "string", "category"])
@pytest.mark.parametrize(
    "strategy",
    [
        PreprocessingTuner.CATEGORICALS_ALL_INT,
        PreprocessingTuner.CATEGORICALS_MIX,
    ],
)
def test_selected_categorical_encoding_survives_fold_text_threshold(dtype, strategy):
    values = [f"token_{i:03}" for i in range(201)] + ["token_000"] * 399
    X = pd.DataFrame(
        {
            "category": pd.Series(values, dtype=dtype),
            "numeric": np.arange(600, dtype=float),
        }
    )
    y = pd.Series(np.arange(600) % 2)
    info = DataInfo.compute(X, y, "binary_classification")
    assert "categorical" in info["columns_info"]["category"]
    training = X.iloc[:240].copy()
    if dtype != "category":
        assert (
            PreprocessingUtils.get_type(training["category"]) == PreprocessingUtils.TEXT
        )
    params = PreprocessingTuner.get(
        ["convert_categorical", "scale"],
        info,
        "binary_classification",
        categorical_strategy=strategy,
    )
    pipeline = Preprocessing(copy.deepcopy(params))
    encoded, _, _ = pipeline.fit_and_transform(training.copy(), y.iloc[:240].copy())
    assert np.isfinite(encoded.to_numpy(dtype=float)).all()
    assert "category" in encoded
    assert encoded["category"].nunique() == 201
    validation, _, _ = pipeline.transform(X.iloc[240:].copy(), y.iloc[240:].copy())
    assert list(encoded.columns) == list(validation.columns)
    new = pd.DataFrame({"category": ["token_001", "unseen"], "numeric": [1.0, 2.0]})
    expected, _, _ = pipeline.transform(new.copy(), None)
    restored = Preprocessing()
    restored.from_json(json.loads(json.dumps(pipeline.to_json())), ".")
    actual, _, _ = restored.transform(new.copy(), None)
    pd.testing.assert_frame_equal(expected, actual)


@pytest.mark.parametrize("method", ["categorical_to_int", "categorical_to_onehot"])
def test_explicit_encoding_honors_selected_text_column_and_onehot_fallback(method):
    X = pd.DataFrame({"category": [f"value_{i}" for i in range(201)]})
    assert PreprocessingUtils.get_type(X["category"]) == PreprocessingUtils.TEXT
    encoder = PreprocessingCategorical(["category"], method)
    encoder.fit(X)
    actual = encoder.transform(X.copy())
    # Above 200 categories, requested one-hot encoding falls back to integers.
    assert list(actual.columns) == ["category"]
    assert actual["category"].nunique() == 201
    assert not any(isinstance(value, str) for value in actual["category"])


@pytest.mark.parametrize("dtype", ["int64", "float64", "Int64", "Float64"])
def test_already_numeric_columns_are_not_reencoded(dtype):
    X = pd.DataFrame({"number": pd.Series([10, 20, 30], dtype=dtype)})
    encoder = PreprocessingCategorical(["number"], "categorical_to_int")
    encoder.fit(X)
    assert encoder._convert_params == {}
    pd.testing.assert_frame_equal(X, encoder.transform(X.copy()))
