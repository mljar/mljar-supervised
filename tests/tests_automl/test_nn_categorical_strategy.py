import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from supervised import AutoML
from supervised.model_framework import ModelFramework


@pytest.mark.parametrize(
    "task", ["binary_classification", "multiclass_classification", "regression"]
)
def test_nn_mix_encoding_trains_predicts_and_reloads(tmp_path, task):
    rows = 240
    X = pd.DataFrame(
        {
            "category": ["a", "b", "c"] * (rows // 3),
            "numeric": np.random.default_rng(1234).normal(size=rows),
        }
    )
    if task == "regression":
        y = pd.Series(np.arange(rows) / rows + X["numeric"])
    else:
        y = pd.Series(np.arange(rows) % (2 if task == "binary_classification" else 3))
    results = tmp_path / task
    automl = AutoML(
        results_path=str(results),
        algorithms=["Neural Network"],
        ml_task=task,
        total_time_limit=60,
        train_ensemble=False,
        stack_models=False,
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        boost_on_errors=False,
        mix_encoding=True,
        start_random_models=1,
        hill_climbing_steps=0,
        explain_level=0,
        verbose=0,
        validation_strategy={
            "validation_type": "split",
            "train_ratio": 0.8,
            "shuffle": True,
            "stratify": task != "regression",
        },
    )
    # Bound test BLAS work without relying on optional NN n_jobs support.
    with threadpool_limits(limits=1):
        automl.fit(X.copy(), y)
    mixed = [model for model in automl._models if "categorical_mix" in model.get_name()]
    assert len(mixed) == 1
    assert mixed[0].params["preprocessing"]["columns_preprocessing"]["category"] == [
        "categorical_to_onehot"
    ]
    assert not (results / "errors.md").exists()
    new = pd.DataFrame({"category": ["a", "unseen"], "numeric": [0.5, 1.5]})
    expected = automl.predict(new.copy())
    mixed_predictions = mixed[0].predict(new.copy())
    restored = AutoML(results_path=str(results))
    np.testing.assert_allclose(expected, restored.predict(new.copy()))
    # AutoML may load only the best model, so explicitly reload the mixed model.
    restored_mixed = ModelFramework.load(str(results), mixed[0].get_name())
    pd.testing.assert_frame_equal(mixed_predictions, restored_mixed.predict(new.copy()))
