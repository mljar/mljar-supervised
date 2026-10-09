import json

import numpy as np
import pandas as pd
import pytest

from supervised.preprocessing.preprocessing import Preprocessing


@pytest.mark.parametrize("categories", [["a", "b"], ["a", "b", "c"]])
def test_onehot_scaling_uses_generated_columns_and_survives_reload(categories):
    X = pd.DataFrame(
        {
            "category": categories * 4,
            "numeric": np.arange(len(categories) * 4, dtype=float),
        }
    )
    params = {
        "columns_preprocessing": {
            "category": ["categorical_to_onehot", "scale_normal"],
            "numeric": ["scale_normal"],
        },
        "target_preprocessing": [],
    }
    pipeline = Preprocessing(params)
    actual, _, _ = pipeline.fit_and_transform(X.copy(), None)
    outputs = (
        ["category_b"]
        if len(categories) == 2
        else [f"category_{v}" for v in categories]
    )
    assert list(actual.columns) == ["numeric"] + outputs
    assert set(pipeline._scale[0].columns) == set(actual.columns)
    np.testing.assert_allclose(actual.mean(), 0, atol=1e-12)
    np.testing.assert_allclose(actual.var(ddof=0), 1, atol=1e-12)
    new = pd.DataFrame(
        {"category": [categories[0], "unseen"], "numeric": [100.0, 200.0]}
    )
    expected, _, _ = pipeline.transform(new.copy(), None)
    restored = Preprocessing()
    restored.from_json(json.loads(json.dumps(pipeline.to_json())), ".")
    reloaded, _, _ = restored.transform(new.copy(), None)
    pd.testing.assert_frame_equal(expected, reloaded)


def test_onehot_integer_fallback_scales_the_original_column():
    X = pd.DataFrame({"category": [f"value_{i}" for i in range(201)] * 4})
    pipeline = Preprocessing(
        {
            "columns_preprocessing": {
                "category": ["categorical_to_onehot", "scale_normal"]
            }
        }
    )
    actual, _, _ = pipeline.fit_and_transform(X.copy(), None)
    assert list(actual.columns) == ["category"]
    np.testing.assert_allclose(actual.mean(), 0, atol=1e-12)
    np.testing.assert_allclose(actual.var(ddof=0), 1, atol=1e-12)


def test_mixed_integer_onehot_and_missing_value_scaling():
    X = pd.DataFrame(
        {
            "onehot": ["a", "b", "c", None] * 4,
            "integer": ["x", "y", "z", "x"] * 4,
            "numeric": np.arange(16, dtype=float),
        }
    )
    pipeline = Preprocessing(
        {
            "columns_preprocessing": {
                "onehot": ["na_fill_median", "categorical_to_onehot", "scale_normal"],
                "integer": ["categorical_to_int", "scale_normal"],
                "numeric": ["scale_normal"],
            }
        }
    )
    actual, _, _ = pipeline.fit_and_transform(X.copy(), None)
    assert set(actual.columns) == {
        "onehot_a",
        "onehot_b",
        "onehot_c",
        "integer",
        "numeric",
    }
    np.testing.assert_allclose(actual.mean(), 0, atol=1e-12)
    assert not actual.isna().any().any()
