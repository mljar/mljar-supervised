import json
import warnings

import numpy as np
import pandas as pd
import pytest

from supervised.preprocessing.label_encoder import LabelEncoder
from supervised.preprocessing.preprocessing import Preprocessing
from supervised.preprocessing.preprocessing_categorical import PreprocessingCategorical


@pytest.mark.parametrize("dtype", [bool, object, "boolean", "category"])
@pytest.mark.parametrize(
    "training", [[False, True, False], [True, True], [False, False]]
)
def test_boolean_labels_keep_codes_after_reload_and_unseen_values(dtype, training):
    encoder = LabelEncoder()
    values = pd.Series(training, dtype=dtype)
    encoder.fit(values)
    known = encoder.transform(values)
    restored = LabelEncoder()
    restored.from_json(json.loads(json.dumps(encoder.to_json())))
    np.testing.assert_array_equal(known, restored.transform(values))
    new = pd.Series([False, True, False], dtype=dtype)
    expected = encoder.transform(new)
    np.testing.assert_array_equal(expected, restored.transform(new))
    # Adding a previously unseen boolean must not shift any existing label code.
    np.testing.assert_array_equal(known, encoder.transform(values))
    np.testing.assert_array_equal(known, restored.transform(values))
    again = LabelEncoder()
    again.from_json(json.loads(json.dumps(restored.to_json(), sort_keys=True)))
    np.testing.assert_array_equal(expected, again.transform(new))
    np.testing.assert_array_equal(
        encoder.inverse_transform(expected).astype(str),
        again.inverse_transform(expected).astype(str),
    )


@pytest.mark.parametrize(
    "mapping,expected",
    [
        ({"False": 0, "True": 1}, [0, 1, 0]),
        ({"True": 0, "False": 1}, [1, 0, 1]),
        ({"True": 0}, [1, 0, 1]),
        ({"False": 0}, [0, 1, 0]),
        ({"0.0": 0, "1.0": 1}, [0, 1, 0]),
        ({"1.0": 0}, [1, 0, 1]),
        ({"0.0": 0}, [0, 1, 0]),
    ],
)
@pytest.mark.parametrize("dtype", [bool, object, "boolean", "category"])
def test_legacy_boolean_mappings(mapping, expected, dtype):
    encoder = LabelEncoder()
    encoder.from_json(mapping)
    np.testing.assert_array_equal(
        encoder.transform(pd.Series([False, True, False], dtype=dtype)), expected
    )


@pytest.mark.parametrize("dtype", [bool, object, "boolean", "category"])
def test_categorical_boolean_output_is_integer_and_preserves_other_columns(dtype):
    X = pd.DataFrame(
        {
            "flag": pd.Series([False, True, False], dtype=dtype),
            "value": [1.25, 2.5, 3.75],
        }
    )
    X.index = [9, 4, 7]
    original = X.copy()
    encoder = PreprocessingCategorical(["flag"], "categorical_to_int")
    encoder.fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = encoder.transform(X.copy())
    assert actual["flag"].tolist() == [0, 1, 0]
    assert pd.api.types.is_integer_dtype(actual["flag"])
    pd.testing.assert_series_equal(original["value"], actual["value"])
    pd.testing.assert_frame_equal(X, original)
    restored = PreprocessingCategorical()
    restored.from_json(json.loads(json.dumps(encoder.to_json())))
    pd.testing.assert_frame_equal(actual, restored.transform(X.copy()))


def test_nullable_boolean_missing_values_and_scaling_survive_reload():
    X = pd.DataFrame(
        {
            "flag": pd.Series([False, True, pd.NA, False], dtype="boolean"),
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    pipeline = Preprocessing(
        {
            "columns_preprocessing": {
                "flag": ["na_fill_median", "categorical_to_int", "scale_normal"],
                "value": ["scale_normal"],
            }
        }
    )
    actual, _, _ = pipeline.fit_and_transform(X.copy(), None)
    assert np.isfinite(actual.to_numpy()).all()
    np.testing.assert_allclose(actual.mean(), 0, atol=1e-12)
    restored = Preprocessing()
    restored.from_json(json.loads(json.dumps(pipeline.to_json())), ".")
    expected, _, _ = pipeline.transform(X.copy(), None)
    reloaded, _, _ = restored.transform(X.copy(), None)
    pd.testing.assert_frame_equal(expected, reloaded)


def test_boolean_looking_strings_and_numeric_label_order_are_preserved():
    encoder = LabelEncoder()
    values = pd.Series(["False", "True", "False"])
    encoder.fit(values)
    expected = encoder.transform(values)
    restored = LabelEncoder()
    restored.from_json(encoder.to_json())
    np.testing.assert_array_equal(expected, restored.transform(values))
    new_strings = pd.Series(["False", "True", "other"])
    np.testing.assert_array_equal(
        encoder.transform(new_strings), restored.transform(new_strings)
    )
    assert restored.transform(pd.Series([False, True], dtype="boolean")).tolist() == [
        0,
        1,
    ]
    numeric = LabelEncoder(try_to_fit_numeric=True)
    values = pd.Series(["10", "2", "1"])
    numeric.fit(values)
    assert numeric.transform(values).tolist() == [2, 1, 0]
    restored.from_json(json.loads(json.dumps(numeric.to_json(), sort_keys=True)))
    np.testing.assert_array_equal(numeric.transform(values), restored.transform(values))
