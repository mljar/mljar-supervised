import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, ShuffleSplit

from supervised.preprocessing.preprocessing import Preprocessing


@pytest.mark.parametrize("folds", [1, 5])
def test_kmeans_repeated_validation_predictions_survive_reload(tmp_path, folds):
    X, y = make_regression(n_samples=100, n_features=6, random_state=17)
    X, y = pd.DataFrame(X), pd.Series(y)
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    query = X.iloc[:10].copy()
    results_path = tmp_path / "original"
    (results_path / "model").mkdir(parents=True)
    params = {
        "kmeans_features": {"results_path": str(results_path)},
        "columns_preprocessing": {},
        "target_preprocessing": [],
    }
    splitter = (
        ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
        if folds == 1
        else RepeatedKFold(n_splits=folds, n_repeats=3, random_state=42)
    )
    saved_learners = []
    for index, (train, _) in enumerate(splitter.split(X)):
        pre = Preprocessing(
            copy.deepcopy(params), "model", index % folds, index // folds
        )
        train_X, train_y, _ = pre.fit_and_transform(X.iloc[train], y.iloc[train])
        learner = Ridge().fit(train_X, train_y)
        expected = learner.predict(pre.transform(query.copy(), None)[0])
        saved_learners.append((json.loads(json.dumps(pre.to_json())), learner, expected))

    # Reload only after every repeat has saved, as in a completed AutoML run.
    # Moving the directory also checks that the new filenames remain portable.
    moved_path = tmp_path / "moved"
    results_path.rename(moved_path)
    for metadata, learner, expected in saved_learners:
        restored = Preprocessing()
        restored.from_json(metadata, str(moved_path))
        actual = learner.predict(restored.transform(query.copy(), None)[0])
        np.testing.assert_array_equal(actual, expected)

    assert len(list((moved_path / "model").glob("kmeans*.joblib"))) == folds * 3
