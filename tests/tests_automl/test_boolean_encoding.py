import numpy as np
import pandas as pd
import pytest

from supervised import AutoML


@pytest.mark.parametrize("dtype", [bool, object, "boolean", "category"])
def test_boolean_and_float_data_train_predict_and_reload(tmp_path, dtype):
    rng = np.random.default_rng(333)
    rows = 160
    X = pd.DataFrame(
        {
            "flag": pd.Series(np.arange(rows) % 2 == 0, dtype=dtype),
            "value": rng.normal(size=rows),
        }
    )
    y = pd.Series((X["flag"].astype(bool) | (X["value"] > 0)).astype(int))
    if dtype == "boolean":
        X.loc[[4, 10], "flag"] = pd.NA
    results = tmp_path / str(dtype)
    automl = AutoML(
        results_path=str(results),
        algorithms=["Decision Tree"],
        total_time_limit=30,
        train_ensemble=False,
        stack_models=False,
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        start_random_models=1,
        hill_climbing_steps=0,
        mix_encoding=False,
        explain_level=0,
        n_jobs=1,
        verbose=0,
        validation_strategy={
            "validation_type": "split",
            "train_ratio": 0.8,
            "shuffle": True,
            "stratify": True,
        },
    )
    automl.fit(X.copy(), y)
    assert any(model.get_type() == "Decision Tree" for model in automl._models)
    assert not (results / "errors.md").exists()
    new = pd.DataFrame(
        {
            "flag": pd.Series([False, True, False], dtype=dtype),
            "value": [0.1, -0.1, 1.5],
        }
    )
    expected = automl.predict_proba(new.copy())
    restored = AutoML(results_path=str(results))
    np.testing.assert_allclose(expected, restored.predict_proba(new.copy()))
