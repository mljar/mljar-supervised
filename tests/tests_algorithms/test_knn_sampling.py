import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

from supervised.algorithms.knn import KNeighborsAlgorithm, KNeighborsRegressorAlgorithm


def features(rows):
    return np.column_stack((np.arange(rows), np.sin(np.arange(rows))))


@pytest.mark.parametrize(
    "counts", [(1199, 1), (600, 401), (1198, 1, 1), (200, 2), (49980, 20)]
)
def test_fallback_retains_classes_cap_and_unique_rows(counts, tmp_path):
    y = np.repeat(np.arange(len(counts)), counts)
    X = features(len(y))
    original = X.copy()
    limit = 10 if counts == (200, 2) else 1000
    task = "binary_classification" if len(counts) == 2 else "multiclass_classification"
    params = dict(ml_task=task, rows_limit=limit, n_neighbors=3, n_jobs=1)
    with pytest.warns(UserWarning, match="class-preserving subsampling"):
        first = KNeighborsAlgorithm(params)
        first.fit(X, y)
    with pytest.warns(UserWarning, match="class-preserving subsampling"):
        second = KNeighborsAlgorithm(params)
        second.fit(X, y)
    selected = first.model._fit_X[:, 0].astype(int)
    assert len(selected) == limit
    assert len(np.unique(selected)) == limit
    assert_array_equal(first.model.classes_, np.unique(y))
    assert_array_equal(np.unique(y[selected]), np.unique(y))
    assert_array_equal(first.model._fit_X, second.model._fit_X)
    assert_array_equal(X, original)
    expected = first.predict(X[:10])
    assert np.isfinite(expected).all()
    path = str(tmp_path / first.get_fname())
    first.save(path)
    restored = KNeighborsAlgorithm(params)
    restored.load(path)
    assert_array_equal(restored.predict(X[:10]), expected)


def test_healthy_classification_preserves_exact_legacy_selection():
    X = features(2400)
    y = np.arange(2400) % 3
    expected_X, _, expected_y, _ = train_test_split(
        X, y, train_size=1000, stratify=y, random_state=1234
    )
    learner = KNeighborsAlgorithm(dict(ml_task="multiclass_classification", n_jobs=1))
    learner.fit(X, y)
    assert_array_equal(learner.model._fit_X, expected_X)
    assert_array_equal(learner.model.classes_[learner.model._y], expected_y)


@pytest.mark.parametrize("container", ["numpy", "pandas", "sparse"])
def test_regression_sampling_is_unstratified_and_repeatable(container):
    X = features(1400)
    y = np.linspace(-0.25, 2.25, len(X))
    expected_X, _, expected_y, _ = train_test_split(
        X, y, train_size=1000, random_state=1234
    )
    if container == "pandas":
        index = np.arange(len(X))[::-1] + 500
        X = pd.DataFrame(X, index=index)
        y = pd.Series(y, index=index)
    elif container == "sparse":
        X = csr_matrix(X)
    learner = KNeighborsRegressorAlgorithm(dict(ml_task="regression", n_jobs=1))
    learner.fit(X, y)
    fitted = (
        learner.model._fit_X.toarray()
        if container == "sparse"
        else learner.model._fit_X
    )
    assert_array_equal(fitted, expected_X)
    assert_array_equal(learner.model._y, expected_y)
    assert np.isfinite(learner.predict(expected_X[:10])).all()


def test_classification_pandas_positions_and_string_labels():
    X = pd.DataFrame(features(1200), index=np.arange(1200)[::-1] + 500)
    y = pd.Series(["major"] * 1199 + ["minor"], index=X.index)
    original = X.copy()
    learner = KNeighborsAlgorithm(dict(ml_task="binary_classification", n_jobs=1))
    with pytest.warns(UserWarning, match="class-preserving subsampling"):
        learner.fit(X, y)
    assert_array_equal(learner.model.classes_, ["major", "minor"])
    assert 1199 in learner.model._fit_X[:, 0]
    pd.testing.assert_frame_equal(X, original)


def test_cap_equal_to_class_count_keeps_one_of_each():
    y = np.repeat(np.arange(10), [20] * 9 + [1])
    learner = KNeighborsAlgorithm(
        dict(ml_task="multiclass_classification", rows_limit=10, n_neighbors=1)
    )
    with pytest.warns(UserWarning, match="class-preserving subsampling"):
        learner.fit(features(len(y)), y)
    selected = learner.model._fit_X[:, 0].astype(int)
    assert_array_equal(np.bincount(y[selected]), np.ones(10))


@pytest.mark.parametrize("rows", [40, 1000])
def test_below_cap_uses_all_rows(rows):
    X = features(rows)
    learner = KNeighborsRegressorAlgorithm(dict(ml_task="regression"))
    learner.fit(X, np.arange(rows, dtype=float))
    assert_array_equal(learner.model._fit_X, X)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, None])
def test_invalid_rows_limit_is_actionable(limit):
    learner = KNeighborsAlgorithm(
        dict(ml_task="binary_classification", rows_limit=limit)
    )
    with pytest.raises(ValueError, match="rows_limit must be a positive integer"):
        learner.fit(features(20), np.arange(20) % 2)


def test_too_many_classes_for_cap_is_actionable():
    learner = KNeighborsAlgorithm(
        dict(ml_task="multiclass_classification", rows_limit=10)
    )
    with pytest.raises(ValueError, match="cannot retain all 11 training classes"):
        learner.fit(features(22), np.arange(22) % 11)


@pytest.mark.parametrize("rows,limit", [(2, 1000), (20, 2)])
def test_neighbors_cannot_exceed_available_sample(rows, limit):
    learner = KNeighborsRegressorAlgorithm(
        dict(ml_task="regression", n_neighbors=3, rows_limit=limit)
    )
    with pytest.raises(ValueError, match="Decrease n_neighbors"):
        learner.fit(features(rows), np.arange(rows, dtype=float))
