import gc
import weakref

import pandas as pd
import pytest

from supervised.utils.utils import Store, dump_data, load_data


def test_stores_are_isolated_and_reads_still_copy():
    a, b = Store(), Store()
    a.set("X", pd.DataFrame({"x": [1, 2]}))
    b.set("X", pd.DataFrame({"x": [3, 4]}))
    result = a.get("X")
    result.iloc[0, 0] = 99
    assert a.get("X").iloc[0, 0] == 1
    a.clear()
    assert len(a) == 0
    assert b.get("X").iloc[0, 0] == 3


def test_clear_releases_strong_references():
    store = Store()
    frame = pd.DataFrame({"x": [1, 2]})
    ref = weakref.ref(frame)
    store.set("X", frame)
    del frame
    assert ref() is not None
    store.clear()
    gc.collect()
    assert ref() is None


def test_generated_checkpoint_survives_store_cleanup(tmp_path):
    path = str(tmp_path / "generated_weight.data")
    expected = pd.DataFrame({"sample_weight": [0.9, 1.1]})
    original = Store()
    dump_data(path, expected, store=original, checkpoint=True)
    original.clear()
    resumed = Store()
    pd.testing.assert_frame_equal(load_data(path, store=resumed), expected)
    assert len(original) == 0 and len(resumed) == 1


def test_failed_checkpoint_preserves_previous_file(tmp_path, monkeypatch):
    import supervised.utils.utils as utils

    path = str(tmp_path / "weight.data")
    old = pd.DataFrame({"w": [1]})
    store = Store()
    dump_data(path, old, store=store, checkpoint=True)

    def broken_dump(value, path):
        with open(path, "wb") as f:
            f.write(b"incomplete")
        raise OSError("disk error")

    monkeypatch.setattr(utils.joblib, "dump", broken_dump)
    with pytest.raises(OSError, match="disk error"):
        dump_data(path, pd.DataFrame({"w": [2]}), store=store, checkpoint=True)
    pd.testing.assert_frame_equal(load_data(path, store=Store()), old)
    pd.testing.assert_frame_equal(store.get(path), old)
    assert not list(tmp_path.glob(".training-data-*"))


@pytest.mark.parametrize(
    "rows", [[0, 2], [0, 1, 2], [2, 1, 0], [1, 1], [True, False, True], []]
)
def test_selected_rows_are_isolated_without_full_read(rows, monkeypatch):
    import numpy as np

    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    store = Store()
    store.set("X", frame)
    original = frame.copy(deep=True)

    def forbidden(key):
        raise AssertionError("Full-frame copy is redundant")

    monkeypatch.setattr(store, "get", forbidden)
    selected = load_data("X", store=store, rows=rows)
    pd.testing.assert_frame_equal(selected, frame.iloc[rows])
    if len(selected):
        selected.iloc[0, 0] = 99
        # Exercise writable NumPy output on pre-CoW pandas too.
        values = selected.to_numpy()
        if values.flags.writeable:
            values[:] = -5
    selected.drop(columns=["b"], inplace=True)
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match="one-dimensional"):
        store.get_rows("X", slice(None))


def test_selected_checkpoint_load_and_row_count(tmp_path):
    from supervised.utils.utils import data_row_count

    path = str(tmp_path / "weights.data")
    frame = pd.DataFrame({"sample_weight": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
    store = Store()
    dump_data(path, frame, store=store, checkpoint=True)
    store.clear()
    assert data_row_count(path, store=store) == 3
    selected = load_data(path, store=store, rows=[30, 10], positional=False)
    pd.testing.assert_frame_equal(selected, frame.loc[[30, 10]])
    selected.iloc[0, 0] = 99
    pd.testing.assert_frame_equal(store.get(path), frame)
