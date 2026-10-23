import json
import warnings

import numpy as np
import pandas as pd
import pytest

from supervised.preprocessing.label_binarizer import LabelBinarizer
from supervised.preprocessing.preprocessing_categorical import PreprocessingCategorical


def test_wide_onehot_preserves_values_order_and_reload_without_fragmentation():
    X = pd.DataFrame({f"c{i}": ["b", "a", "c", None] for i in range(60)},
                     index=[8, 3, 5, 5])
    train = X.iloc[:3]
    encoder = PreprocessingCategorical(list(X), PreprocessingCategorical.CONVERT_ONE_HOT)
    encoder.fit(train)
    restored = PreprocessingCategorical()
    restored.from_json(json.loads(json.dumps(encoder.to_json())))
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        actual = restored.transform(X)
        actual["extra"] = 1  # downstream insertion must also be safe
    expected = pd.DataFrame({f"c{i}_{value}": (X[f"c{i}"] == value).astype(int)
                             for i in range(60) for value in ["a", "b", "c"]},
                            index=X.index)
    expected["extra"] = 1
    pd.testing.assert_frame_equal(actual, expected)
    assert list(X) == [f"c{i}" for i in range(60)]


@pytest.mark.parametrize("values", [["a", "b"], [False, True], list(map(str, range(150)))])
def test_binarizer_matches_legacy_assignment_with_unknowns_and_collision(values):
    encoder = LabelBinarizer()
    encoder.fit(pd.DataFrame({"c": values}), "c")
    X = pd.DataFrame({"before": range(3), "c": [values[0], values[-1], None],
                      f"c_{values[-1]}": [7, 8, 9], "after": np.arange(3.)},
                     index=[7, 2, 1])
    expected = dict(X.items())
    selected = values[1:] if len(values) == 2 else np.unique(values)
    for value in selected:
        expected[f"c_{value}"] = (X.c == value).astype(int)
    del expected["c"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        actual = encoder.transform(X, "c")
    pd.testing.assert_frame_equal(actual, pd.DataFrame(expected, index=X.index))
