import copy
import os
import tempfile

import joblib
import numpy as np


class Store:
    """Temporary training data owned by one fit, never by the process."""

    def __init__(self):
        self._data = {}

    def set(self, key, value):
        self._data[key] = value

    def get(self, key):
        return copy.deepcopy(self._data[key])

    def get_rows(self, key, rows, *, positional=True):
        """Return an isolated subset without copying the full stored frame.

        Array indexing materializes selected rows (or uses pandas copy-on-write).
        Never accept slices here: older pandas can return writable slice views.
        As with DataFrame.copy, nested mutable Python objects are not deep-copied.
        """
        rows = np.asarray(rows)
        if rows.ndim != 1:
            raise ValueError("Row selection must be a one-dimensional array")
        frame = self._data[key]
        subset = frame.iloc[rows] if positional else frame.loc[rows]
        # Array indexing already isolates the data. Give the subset its own
        # frame metadata too, without another copy of the selected arrays.
        return subset.copy(deep=False)

    def row_count(self, key):
        return len(self._data[key])

    def contains(self, key):
        return key in self._data

    def clear(self):
        self._data.clear()

    def __len__(self):
        return len(self._data)

    def logical_bytes(self):
        # Counts logical DataFrame bytes, not physical/shared resident memory.
        return sum(
            int(value.memory_usage(deep=False).sum()) for value in self._data.values()
        )


def dump_data(file_path, df, *, store, checkpoint=False):
    if store is None:
        raise RuntimeError("Training data requires an active data store")
    if checkpoint:
        # Generated weights must be recoverable before progress references them.
        directory = os.path.dirname(os.path.abspath(file_path))
        fd, temporary = tempfile.mkstemp(prefix=".training-data-", dir=directory)
        os.close(fd)
        try:
            joblib.dump(df, temporary)
            os.replace(temporary, file_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    store.set(file_path, df)


def _ensure_data(file_path, store):
    if store is None:
        raise RuntimeError("Training data requires an active data store")
    if not store.contains(file_path):
        if not os.path.isfile(file_path):
            raise KeyError(
                f"Training data is unavailable: {file_path}. "
                "Resume fit with the original inputs; generated weights require their checkpoint."
            )
        store.set(file_path, joblib.load(file_path))


def load_data(file_path, *, store, rows=None, positional=True):
    _ensure_data(file_path, store)
    if rows is not None:
        return store.get_rows(file_path, rows, positional=positional)
    return store.get(file_path)


def data_row_count(file_path, *, store):
    _ensure_data(file_path, store)
    return store.row_count(file_path)
