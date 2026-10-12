"""Bounded contiguous buffers for repeated, read-only column inspection."""

import numpy as np
import pandas as pd


def iter_columns(frame, columns=None, buffer_bytes=32 * 1024 * 1024):
    """Yield columns in order, packing strided NumPy numeric runs in small blocks.

    Extension and non-numeric columns retain their pandas representation. Already
    contiguous numeric columns need no copy. Consumers must only read the series.
    A single column may exceed the budget; otherwise each buffer is at most the
    budget. The previous yielded series can keep one preceding buffer alive.
    """
    columns = list(frame.columns if columns is None else columns)
    dtypes = frame.dtypes
    position = 0
    while position < len(columns):
        column = columns[position]
        if column not in frame.columns:
            yield column, None
            position += 1
            continue
        series = frame[column]
        dtype = series.dtype
        if not isinstance(dtype, np.dtype) or dtype.kind not in "iuf":
            yield column, series
            position += 1
            continue
        values = series.to_numpy(copy=False)
        if values.flags.c_contiguous:
            yield column, series
            position += 1
            continue
        start = frame.columns.get_loc(column)
        width = max(1, buffer_bytes // max(1, len(frame) * dtype.itemsize))
        end = position + 1
        while end < min(len(columns), position + width):
            next_position = start + end - position
            if (
                next_position >= len(frame.columns)
                or frame.columns[next_position] != columns[end]
                or dtypes.iloc[next_position] != dtype
            ):
                break
            end += 1
        block = np.array(
            frame.iloc[:, start : start + end - position].to_numpy(copy=False),
            order="F",
            copy=True,
        )
        block.setflags(write=False)
        for offset, name in enumerate(columns[position:end]):
            yield name, pd.Series(
                block[:, offset], index=frame.index, name=name, copy=False
            )
        del block
        position = end
