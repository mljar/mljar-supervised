import numpy as np
import pandas as pd
import pytest

from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.tuner.data_info import DataInfo
from tests.tests_tuner._data_info_reference import DataInfo as ReferenceDataInfo
from tests.tests_tuner._data_info_reference import PreprocessingUtils as ReferenceUtils


def columns():
    n = 512
    rng = np.random.default_rng(1234)
    result = {
        "normal": pd.Series(rng.normal(size=n)),
        "scaled": pd.Series(rng.normal(4, 3, size=n)),
        "float32": pd.Series(rng.normal(size=n), dtype="float32"),
        "integer": pd.Series(np.arange(n)),
        "unsigned": pd.Series(np.arange(n), dtype="uint32"),
        "nullable_float": pd.Series([1.0, pd.NA, 2.0, 3.0] * 128, dtype="Float64"),
        "nullable_int": pd.Series([1, pd.NA, 2, 3] * 128, dtype="Int64"),
        "nullable_many": pd.Series(range(n), dtype="Int64"),
        "nullable_bool": pd.Series([True, pd.NA, False, True] * 128, dtype="boolean"),
        "bool": pd.Series([True, False] * 256),
        "constant": pd.Series([1.0] * n),
        "constant_missing": pd.Series([1.0, np.nan] * 256),
        "constant_late_change": pd.Series([1.0] * (n - 1) + [2.0]),
        "all_missing": pd.Series([np.nan] * n),
        "all_nullable_missing": pd.Series([pd.NA] * n, dtype="Float64"),
        "all_object_missing": pd.Series([None] * n, dtype=object),
        "infinity": pd.Series([np.inf, -np.inf, 1.0, np.nan] * 128),
        "constant_infinity": pd.Series([np.inf, np.nan] * 256),
        "signed_zero": pd.Series([0.0, -0.0] * 256),
        "few_categories": pd.Series(["a", "b", None, "c"] * 128, dtype=object),
        "twenty_categories": pd.Series([str(i % 20) for i in range(n)]),
        "many_categories": pd.Series([str(i % 21) for i in range(n)]),
        "text": pd.Series([str(i) for i in range(n)]),
        "text_with_missing": pd.Series([str(i) for i in range(256)] + [None] * 256),
        "category": pd.Series(pd.Categorical([str(i) for i in range(n)])),
        "category_with_unused": pd.Series(
            pd.Categorical(["a"] * n, categories=["a", "b"])
        ),
        "datetime": pd.Series(pd.date_range("2020-01-01", periods=n)),
        "datetime_tz": pd.Series(pd.date_range("2020-01-01", periods=n, tz="UTC")),
        "datetime_missing": pd.Series([pd.Timestamp("2020-01-01"), pd.NaT] * 256),
        "timedelta": pd.Series(pd.to_timedelta(np.arange(n), unit="s")),
    }
    for dtype in ("float32", "float64"):
        for value in (0.5, np.nextafter(0.5, 0.0), np.nextafter(0.5, 1.0)):
            result[f"mean_boundary_{dtype}_{value}"] = pd.Series(
                [value - 0.25, value + 0.25] * 256, dtype=dtype
            )
        for value in (1.5, np.nextafter(1.5, 0.0), np.nextafter(1.5, 2.0)):
            result[f"std_boundary_{dtype}_{value}"] = pd.Series(
                [-value, value] * 256, dtype=dtype
            )
    return result


@pytest.mark.parametrize(
    "task", ["regression", "binary_classification", "multiclass_classification"]
)
def test_metadata_matches_reference_across_dtypes(task):
    frame = pd.DataFrame(columns())
    rng = np.random.default_rng(42)
    y = (
        pd.Series(rng.normal(size=len(frame)))
        if task == "regression"
        else pd.Series(
            np.arange(len(frame)) % (2 if task == "binary_classification" else 10)
        )
    )
    assert DataInfo.compute(frame, y, task) == ReferenceDataInfo.compute(frame, y, task)


@pytest.mark.parametrize("name,series", list(columns().items()), ids=list(columns()))
def test_type_helpers_preserve_reference(name, series):
    for method in (
        "get_type",
        "is_categorical",
        "is_datetime",
        "is_text",
        "is_scale_needed",
    ):
        if method == "is_scale_needed" and not pd.api.types.is_numeric_dtype(
            series.dtype
        ):
            continue
        with np.errstate(all="ignore"):
            try:
                expected = getattr(ReferenceUtils, method)(series)
            except (TypeError, ValueError) as error:
                with pytest.raises(type(error), match=str(error)):
                    getattr(PreprocessingUtils, method)(series)
            else:
                assert getattr(PreprocessingUtils, method)(series) == expected


@pytest.mark.parametrize("unique_count", [20, 21, 200, 201, 256, 257])
def test_text_and_encoding_boundaries(unique_count):
    values = [str(i % unique_count) for i in range(512)]
    for missing in (0, 200):
        frame = pd.DataFrame({"column": values[: 512 - missing] + [None] * missing})
        y = pd.Series(np.arange(512) % 2)
        assert DataInfo.compute(
            frame, y, "binary_classification"
        ) == ReferenceDataInfo.compute(frame, y, "binary_classification")


def test_empty_frame_and_nonbinary_target_labels():
    for n in (0, 50):
        frame = pd.DataFrame({"column": np.arange(n, dtype=float)})
        y = pd.Series(["yes", "no"] * (n // 2), dtype=object)
        assert DataInfo.compute(
            frame, y, "binary_classification"
        ) == ReferenceDataInfo.compute(frame, y, "binary_classification")


def test_numeric_metadata_does_not_sort_unique_values(monkeypatch):
    frame = pd.DataFrame({"column": np.linspace(0, 1, 512)})
    y = pd.Series([0, 1] * 256)
    original = np.unique
    lengths = []

    def record(values, *args, **kwargs):
        lengths.append(np.asarray(values).shape)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np, "unique", record)
    DataInfo.compute(frame, y, "binary_classification")
    # Only the target needs unique labels; the feature no longer sorts its values.
    assert lengths == [(512,)]


def test_categorical_counts_are_computed_once(monkeypatch):
    frame = pd.DataFrame({"column": ["a", "b", None] * 100})
    y = pd.Series([0, 1] * 150)
    original = np.unique
    lengths = []

    def record(values, *args, **kwargs):
        lengths.append(len(values))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np, "unique", record)
    DataInfo.compute(frame, y, "binary_classification")
    assert lengths == [200, 300]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_strided_numeric_columns_match_reference(dtype, order):
    rng = np.random.default_rng(19)
    values = np.array(rng.normal(size=(512, 20)), dtype=dtype, order=order)
    values[:, 0] = np.tile([-1.0, 2.0], 256)  # mean 0.5, population std 1.5
    values[:, 1] = np.tile([-1.5, 1.5], 256)
    values[::7, 2] = np.nan
    frame = pd.DataFrame(values, copy=False)
    y = pd.Series(rng.normal(size=512))
    assert DataInfo.compute(frame, y, "regression") == ReferenceDataInfo.compute(
        frame, y, "regression"
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_multiple_blocks_preserve_scaling_boundaries_and_missing_values(
    monkeypatch, dtype
):
    from functools import partial
    from supervised.tuner import data_info
    from supervised.utils.column_scan import iter_columns

    values = np.array(np.random.default_rng(71).normal(size=(512, 24)), dtype=dtype)
    for i, mean in enumerate([0.5, np.nextafter(0.5, 0.0), np.nextafter(0.5, 1.0)]):
        values[:, i] = np.tile([mean - 0.25, mean + 0.25], 256)
    for i, std in enumerate([1.5, np.nextafter(1.5, 0.0), np.nextafter(1.5, 2.0)], 3):
        values[:, i] = np.tile([-std, std], 256)
    values[::7, 8] = np.nan
    values[:, 9] = np.nan
    values[:, 10] = np.inf
    values[-1, 11] = -np.inf
    values[:, 12] = 1
    values[-1, 12] = 2
    frame = pd.DataFrame(values, copy=False)
    original = frame.copy(deep=True)
    y = pd.Series(np.arange(512) % 2)
    monkeypatch.setattr(
        data_info, "iter_columns", partial(iter_columns, buffer_bytes=8192)
    )
    assert DataInfo.compute(
        frame, y, "binary_classification"
    ) == ReferenceDataInfo.compute(frame, y, "binary_classification")
    pd.testing.assert_frame_equal(frame, original)
