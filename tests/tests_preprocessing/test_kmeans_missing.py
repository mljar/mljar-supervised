import copy

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from supervised.preprocessing.kmeans_transformer import KMeansTransformer
from supervised.preprocessing.preprocessing import Preprocessing


def data():
    X, y = make_classification(n_samples=80, n_features=5, random_state=17)
    X = pd.DataFrame(X, columns=list("abcde"))
    X.loc[::3, "a"] = np.nan
    X["empty"] = np.nan
    return X, pd.Series(y)


def test_kmeans_frozen_imputation_reload_and_row_alignment(tmp_path):
    X, y = data()
    (tmp_path / "model").mkdir()
    model = KMeansTransformer(str(tmp_path), "model", 0)
    model.fit(X, y)
    assert model._fill_values["empty"] == 0
    assert model._fill_values["b"] == X.b.median()
    query = X.iloc[:4].copy()
    query.index = [8, 3, 8, 1]
    query.loc[:, "b"] = [np.nan, 10000, -10000, 1]
    original = query.copy()
    expected = model.transform(query.copy())
    assert np.isfinite(expected[model._new_features]).all().all()
    pd.testing.assert_frame_equal(expected[original.columns], original)
    parts = pd.concat([model.transform(query.iloc[[i]].copy()) for i in range(4)])
    pd.testing.assert_frame_equal(expected, parts)
    saved = model.to_json()
    restored = KMeansTransformer()
    restored.from_json(saved, str(tmp_path))
    pd.testing.assert_frame_equal(expected, restored.transform(query.copy()))
    assert restored._fill_values == model._fill_values


def test_legacy_kmeans_artifact_without_imputer(tmp_path):
    X, y = data()
    X = X.fillna(0)
    (tmp_path / "model").mkdir()
    model = KMeansTransformer(str(tmp_path), "model", 0)
    model.fit(X, y)
    saved = model.to_json()
    joblib.dump({"kmeans": model._kmeans, "scale": model._scale}, model._result_path)
    restored = KMeansTransformer()
    restored.from_json(saved, str(tmp_path))
    assert restored._fill_values is None
    pd.testing.assert_frame_equal(model.transform(X.copy()), restored.transform(X.copy()))


def test_pipeline_kmeans_features_use_training_values_before_legacy_fallback(tmp_path):
    X, y = data()
    X["kind"] = ["a", "b"] * 40
    params = {"columns_preprocessing": {"kind": ["categorical_to_int"]},
              "kmeans_features": {"results_path": str(tmp_path)},
              "target_preprocessing": []}
    (tmp_path / "model").mkdir()
    pre = Preprocessing(copy.deepcopy(params), "model", 0)
    train, _, _ = pre.fit_and_transform(X, y)
    query = X.iloc[:4].copy()
    query["b"] = [np.nan, 10000, -10000, 1]
    full = pre.transform(query.copy(), None)[0]
    parts = pd.concat([pre.transform(query.iloc[[i]].copy(), None)[0] for i in range(4)])
    cols = pre._kmeans._new_features
    pd.testing.assert_frame_equal(full[cols], parts[cols])
    assert list(full) == list(train)
    restored = Preprocessing()
    restored.from_json(pre.to_json(), str(tmp_path))
    pd.testing.assert_frame_equal(full, restored.transform(query.copy(), None)[0])
