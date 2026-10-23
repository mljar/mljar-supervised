"""Preprocessing owns its output even with pandas copy-on-write disabled."""

from contextlib import nullcontext
import warnings

import numpy as np
import pandas as pd
import pytest

from supervised.preprocessing.preprocessing import Preprocessing
from supervised.utils.utils import Store


@pytest.mark.parametrize("copy_on_write", [False, True])
@pytest.mark.parametrize("selection", [slice(1, 7), [1, 2, 3, 4, 5, 6]])
def test_sliced_input_is_unchanged_and_warning_free(copy_on_write, selection):
    context = (
        pd.option_context("mode.copy_on_write", copy_on_write)
        if int(pd.__version__.split(".")[0]) < 3
        else nullcontext()
    )
    with context, warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        setting_warning = getattr(pd.errors, "SettingWithCopyWarning", None)
        if setting_warning is not None:
            warnings.simplefilter("error", setting_warning)
        X = pd.DataFrame(
            {
                "number": np.arange(8, dtype=float),
                "missing": [np.nan, 1, 2, np.nan, 4, 5, np.nan, 7],
                "category": ["a", "b"] * 4,
                "constant": [1] * 8,
            }
        )
        original = X.copy(deep=True)
        train = X.iloc[selection]
        valid = X.iloc[2:8]
        train_before, valid_before = train.copy(deep=True), valid.copy(deep=True)
        prep = Preprocessing(
            {
                "columns_preprocessing": {
                    "number": ["scale_normal"],
                    "missing": ["na_fill_median"],
                    "category": ["categorical_to_int"],
                    "constant": ["remove_column"],
                },
            }
        )
        transformed, _, _ = prep.fit_and_transform(train, None)
        predicted, _, _ = prep.transform(valid, None)
        assert not transformed.isna().any().any()
        assert not predicted.isna().any().any()
        assert "constant" not in transformed
        assert transformed["category"].dtype.kind in "iu"
        pd.testing.assert_frame_equal(train, train_before)
        pd.testing.assert_frame_equal(valid, valid_before)
        pd.testing.assert_frame_equal(X, original)


def test_selected_rows_do_not_copy_full_data_or_copy_subset_twice(monkeypatch):
    frame = pd.DataFrame({"x": np.arange(1000, dtype=float)})
    store = Store()
    store.set("X", frame)
    original_copy = pd.DataFrame.copy
    calls = []

    def copy(self, deep=True):
        calls.append((len(self), deep))
        return original_copy(self, deep=deep)

    monkeypatch.setattr(pd.DataFrame, "copy", copy)
    selected = store.get_rows("X", [1, 4, 7])
    assert calls == [(3, False)]
    with warnings.catch_warnings():
        warning = getattr(pd.errors, "SettingWithCopyWarning", None)
        if warning is not None:
            warnings.simplefilter("error", warning)
        selected.iloc[0, 0] = -1
    assert frame.iloc[1, 0] == 1
