import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from supervised.preprocessing.preprocessing_categorical import PreprocessingCategorical


@pytest.mark.parametrize("categorical_dtype", [False, True])
def test_unseen_categories_have_frozen_batch_independent_codes(categorical_dtype):
    train = pd.DataFrame({"resource": ["known_a", "known_b", "known_a"]})
    query = pd.DataFrame(
        {"resource": ["unseen_z", "known_a", "unseen_a", "known_b"]},
        index=[2, 7, 11, 19],
    )
    if categorical_dtype:
        train["resource"] = train.resource.astype("category")
        query["resource"] = query.resource.astype("category")
    encoder = PreprocessingCategorical(["resource"])
    encoder.fit(train)
    saved = json.loads(json.dumps(encoder.to_json()))
    full = encoder.transform(query.copy())
    parts = pd.concat(
        [encoder.transform(query.iloc[[i]].copy()) for i in range(len(query))]
    )
    assert_frame_equal(full, parts)
    np.testing.assert_array_equal(full.resource, [2, 0, 2, 1])
    assert encoder.to_json() == saved
    restored = PreprocessingCategorical()
    restored.from_json(saved)
    assert_frame_equal(full, restored.transform(query.copy()))
    assert_frame_equal(full.iloc[::-1], restored.transform(query.iloc[::-1].copy()))


def test_frozen_encoding_preserves_known_boolean_and_numeric_categories():
    train = pd.DataFrame(
        {
            "flag": pd.Series([True, False, True], dtype="boolean"),
            "code": pd.Categorical([10, 20, 10]),
        }
    )
    encoder = PreprocessingCategorical(list(train))
    encoder.fit(train)
    expected = encoder.transform(train.copy())
    np.testing.assert_array_equal(expected.flag, [1, 0, 1])
    np.testing.assert_array_equal(expected.code, [0, 1, 0])
    restored = PreprocessingCategorical()
    restored.from_json(json.loads(json.dumps(encoder.to_json())))
    assert_frame_equal(expected, restored.transform(train.copy()))


def test_mixed_unseen_boolean_values_do_not_change_known_codes():
    encoder = PreprocessingCategorical(["flag"])
    encoder.fit(pd.DataFrame({"flag": [True, False, True]}))
    query = pd.DataFrame({"flag": [True, 1, "unknown", False, "True", None]})
    full = encoder.transform(query.copy())
    parts = pd.concat(
        [encoder.transform(query.iloc[[i]].copy()) for i in range(len(query))]
    )
    assert_frame_equal(full, parts)
    np.testing.assert_array_equal(full.flag, [1, 1, 2, 0, 1, 2])
