import shutil
import unittest

import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from sklearn import datasets
from sklearn.model_selection import train_test_split

from supervised import AutoML


@pytest.mark.parametrize(
    "task", ["binary_classification", "multiclass_classification", "regression"]
)
def test_unseen_feature_predictions_match_batches_and_reload(tmp_path, task):
    if task == "regression":
        values, target = datasets.make_regression(
            n_samples=240, n_features=4, random_state=42
        )
    else:
        values, target = datasets.make_classification(
            n_samples=240,
            n_features=4,
            n_redundant=0,
            n_informative=3,
            n_classes=3 if task == "multiclass_classification" else 2,
            random_state=42,
        )
    X = pd.DataFrame(values, columns=list("abcd"))
    X["category"] = [f"category_{i % 7}" for i in range(len(X))]
    model = AutoML(
        results_path=str(tmp_path / "automl"),
        algorithms=["Neural Network"],
        ml_task=task,
        total_time_limit=30,
        explain_level=0,
        verbose=0,
        n_jobs=1,
        train_ensemble=False,
        feature_discovery=False,
        validation_strategy={"validation_type": "kfold", "k_folds": 2, "shuffle": True},
    )
    model.fit(X, target)
    query = X.iloc[:24].copy()
    query.loc[query.index[::2], "category"] = "unseen_z"
    query.loc[query.index[1::4], "category"] = "unseen_a"
    full = model.predict_all(query)
    chunks = pd.concat(
        [model.predict_all(query.iloc[i : i + 3]) for i in range(0, len(query), 3)],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(full, chunks, check_exact=False, atol=1e-7, rtol=1e-5)
    restored = AutoML(results_path=str(tmp_path / "automl"), verbose=0)
    pd.testing.assert_frame_equal(
        full, restored.predict_all(query), check_exact=False, atol=1e-7, rtol=1e-5
    )
    pd.testing.assert_frame_equal(
        full.iloc[:3],
        restored.predict_all(query.iloc[:3]),
        check_exact=False,
        atol=1e-7,
        rtol=1e-5,
    )


class AutoMLPredictionAfterLoadTest(unittest.TestCase):
    automl_dir = "AutoMLPredictionAfterLoadTest"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_integration(self):
        a = AutoML(
            results_path=self.automl_dir,
            mode="Compete",
            algorithms=["Baseline", "CatBoost", "LightGBM", "Xgboost"],
            stack_models=True,
            total_time_limit=60,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": 3,
                "shuffle": True,
                "stratify": True,
                "random_seed": 123,
            },
        )

        X, y = datasets.make_classification(
            n_samples=1000,
            n_features=30,
            n_informative=29,
            n_redundant=1,
            n_classes=8,
            n_clusters_per_class=3,
            n_repeated=0,
            shuffle=False,
            random_state=0,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

        a.fit(X_train, y_train)
        p = a.predict_all(X_test)

        a2 = AutoML(results_path=self.automl_dir)
        p2 = a2.predict_all(X_test)

        assert_almost_equal(p["prediction_0"].iloc[0], p2["prediction_0"].iloc[0])
        assert_almost_equal(p["prediction_7"].iloc[0], p2["prediction_7"].iloc[0])
