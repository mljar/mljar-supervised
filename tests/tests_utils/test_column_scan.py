import numpy as np
import pandas as pd
import pytest

from supervised.utils.column_scan import iter_columns


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "uint32"])
def test_strided_blocks_are_bounded_contiguous_and_read_only(dtype):
    values = np.arange(256 * 23).reshape(256, 23).astype(dtype)
    frame = pd.DataFrame(values, index=np.arange(256)[::-1], copy=False)
    original = frame.copy(deep=True)
    budget = 256 * values.dtype.itemsize * 4
    for column, series in iter_columns(frame, buffer_bytes=budget):
        pd.testing.assert_series_equal(series, frame[column])
        buffer = series.to_numpy(copy=False)
        assert buffer.flags.c_contiguous
        assert not buffer.flags.writeable
        assert not np.shares_memory(buffer, values)
        assert buffer.base.nbytes <= budget
    pd.testing.assert_frame_equal(frame, original)
    assert values.flags.writeable


def test_contiguous_columns_are_not_copied():
    values = np.asfortranarray(np.arange(512).reshape(64, 8))
    frame = pd.DataFrame(values, copy=False)
    for column, series in iter_columns(frame):
        assert np.shares_memory(series.to_numpy(copy=False), values)
    assert values.flags.writeable


def test_requested_order_absent_columns_and_extension_dtypes():
    frame = pd.DataFrame(np.arange(128).reshape(16, 8), columns=list("abcdefgh"))
    frame["nullable"] = pd.Series([1, pd.NA] * 8, dtype="Int64")
    frame["category"] = pd.Categorical(["a", "b"] * 8)
    frame["ip"] = ["192.0.2.1", "2001:db8::1"] * 8
    requested = ["f", "b", "c", "absent", "nullable", "category", "ip", "a"]
    actual = list(iter_columns(frame, requested, buffer_bytes=256))
    assert [name for name, _ in actual] == requested
    for name, series in actual:
        if name == "absent":
            assert series is None
        else:
            pd.testing.assert_series_equal(series, frame[name])


def test_empty_rows_and_single_column_larger_than_budget():
    for rows in (0, 512):
        frame = pd.DataFrame(np.arange(rows * 3).reshape(rows, 3))
        for column, series in iter_columns(frame, buffer_bytes=1):
            pd.testing.assert_series_equal(series, frame[column])
