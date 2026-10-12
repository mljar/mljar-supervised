import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from threadpoolctl import threadpool_limits

from supervised import AutoML
from supervised.algorithms import lightgbm


@pytest.mark.parametrize(
    "labels,oof_column",
    [
        (("e", "p"), "prediction_0_for_e_1_for_p"),
        (("0", "1"), "prediction"),
        (("2", "3"), "prediction_0_for_2_1_for_3"),
        (("class_a", "class_b"), "prediction_0_for_class_a_1_for_class_b"),
    ],
)
def test_binary_stacking_preserves_training_schema(labels, oof_column):
    X = pd.DataFrame({"feature": [1.0, 2.0]}, index=[10, 20])
    positive = np.array([0.2, 0.8])
    model = Mock()
    model.get_name.return_value = "Model"
    model.get_out_of_folds.return_value = pd.DataFrame({oof_column: positive})
    model.predict.return_value = pd.DataFrame(
        {
            f"prediction_{labels[0]}": 1 - positive,
            f"prediction_{labels[1]}": positive,
        }
    )
    automl = AutoML()
    automl._ml_task = "binary_classification"
    automl._stacked_models = [model]
    training = automl.get_stacked_data(X.copy())
    predicted = automl.get_stacked_data(X.copy(), mode="predict")
    pd.testing.assert_frame_equal(training, predicted)
    np.testing.assert_array_equal(predicted.iloc[:, -1], positive)
    assert X.index.tolist() == [10, 20]


@pytest.mark.parametrize(
    "task,columns",
    [
        ("regression", ["prediction"]),
        ("multiclass_classification", ["prediction_a", "prediction_b", "prediction_c"]),
    ],
)
def test_other_tasks_keep_stacking_columns(task, columns):
    X = pd.DataFrame({"feature": [1.0, 2.0]})
    predictions = pd.DataFrame({column: [0.2, 0.8] for column in columns})
    model = Mock()
    model.get_name.return_value = "Model"
    model.get_out_of_folds.return_value = predictions.copy()
    model.predict.return_value = predictions.copy()
    automl = AutoML()
    automl._ml_task = task
    automl._stacked_models = [model]
    pd.testing.assert_frame_equal(
        automl.get_stacked_data(X.copy()),
        automl.get_stacked_data(X.copy(), mode="predict"),
    )


def test_kmeans_stacking_predict_and_reload(tmp_path, monkeypatch):
    # Keep real two-fold training small, with enough candidates to enable stacking.
    monkeypatch.setitem(lightgbm.additional, "max_rounds", 5)
    X, y = make_classification(
        n_samples=160, n_features=6, n_informative=4, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = np.where(y == 0, "e", "p")
    root = tmp_path / "automl"
    automl = AutoML(
        results_path=str(root),
        mode="Compete",
        algorithms=["Baseline", "Decision Tree", "LightGBM"],
        total_time_limit=120,
        start_random_models=3,
        hill_climbing_steps=0,
        golden_features=False,
        features_selection=False,
        boost_on_errors=False,
        mix_encoding=False,
        kmeans_features=True,
        stack_models=True,
        train_ensemble=True,
        explain_level=0,
        n_jobs=1,
        verbose=0,
        validation_strategy={
            "validation_type": "kfold",
            "k_folds": 2,
            "shuffle": True,
            "stratify": True,
            "random_seed": 42,
        },
    )
    with threadpool_limits(limits=1):
        automl.fit(X, y)
        stacked = [
            m for m in automl._models
            if m._is_stacked and "KMeansFeatures" in m.get_name()
        ]
        assert stacked, "The test must exercise a stacked KMeans model"
        for model in automl._models:
            if model._is_stacked:
                predictions = automl._base_predict(X, model=model)
                assert np.isfinite(predictions.filter(like="prediction").to_numpy()).all()

        # Select this candidate explicitly so leaderboard scores cannot bypass
        # the failing preprocessing path, including after loading from disk.
        automl._best_model = stacked[0]
        probabilities = automl.predict_proba(X)
        labels = automl.predict(X)
        params_path = root / "params.json"
        params = json.loads(params_path.read_text())
        params["best_model"] = stacked[0].get_name()
        params["load_on_predict"] = params["saved"]
        params_path.write_text(json.dumps(params))
        loaded = AutoML(results_path=str(root))
        np.testing.assert_allclose(loaded.predict_proba(X), probabilities)
        np.testing.assert_array_equal(loaded.predict(X), labels)
