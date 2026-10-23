import json

import numpy as np
import pandas as pd
import pytest

from supervised.algorithms.registry import BINARY_CLASSIFICATION, AlgorithmsRegistry
from supervised.preprocessing.preprocessing import Preprocessing
from supervised.tuner.data_info import DataInfo
from supervised.tuner.preprocessing_tuner import PreprocessingTuner


@pytest.mark.parametrize("dtype", ["int64", "float64", "category"])
def test_binary_target_encoding_depends_on_dtype(dtype):
    X = pd.DataFrame({"feature": [0, 1, 2, 3]})
    y = pd.Series([0, 1, 0, 1], dtype=dtype)
    info = DataInfo.compute(X, y, BINARY_CLASSIFICATION)
    assert info["target_info"] == (["convert_0_1"] if dtype == "category" else [])


@pytest.mark.parametrize("categories", [[0, 1], [1, 0], [2, 1, 0, -1]])
@pytest.mark.parametrize("ordered", [False, True])
def test_categorical_binary_target_mapping_survives_reload(categories, ordered):
    X = pd.DataFrame({"feature": [10, 20, 30, 40, 50]})
    y = pd.Series(
        pd.Categorical([1, 0, None, 1, 0], categories=categories, ordered=ordered),
        name="target",
    )
    original = y.copy()
    info = DataInfo.compute(X, y, BINARY_CLASSIFICATION)
    required = AlgorithmsRegistry.registry[BINARY_CLASSIFICATION]["LightGBM"][
        "required_preprocessing"
    ]
    params = PreprocessingTuner.get(required, info, BINARY_CLASSIFICATION)
    preprocessing = Preprocessing(params)
    transformed_X, transformed_y, _ = preprocessing.fit_and_transform(X.copy(), y)
    assert pd.api.types.is_integer_dtype(transformed_y.dtype)
    np.testing.assert_array_equal(transformed_y, [1, 0, 1, 0])
    np.testing.assert_array_equal(transformed_X["feature"], [10, 20, 40, 50])
    assert preprocessing.get_target_class_names() == ["0", "1"]

    # Exercise the actual JSON round trip, not just an in-memory copy.
    serialized = json.loads(json.dumps(preprocessing.to_json()))
    restored = Preprocessing({})
    restored.from_json(serialized, results_path=None)
    for instance in (preprocessing, restored):
        validation_y = y.iloc[[4, 2, 3, 0, 1]]
        _, encoded_y, _ = instance.transform(X.iloc[[4, 2, 3, 0, 1]], validation_y)
        assert pd.api.types.is_integer_dtype(encoded_y.dtype)
        np.testing.assert_array_equal(encoded_y, [0, 1, 1, 0])
        assert instance.get_target_class_names() == ["0", "1"]
    pd.testing.assert_series_equal(y, original)
