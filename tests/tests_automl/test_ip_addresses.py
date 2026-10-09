import numpy as np
import pandas as pd
import pytest

from supervised import AutoML


@pytest.mark.parametrize("algorithm", ["Linear", "Decision Tree", "CatBoost"])
def test_ip_training_prediction_and_reload(tmp_path, algorithm):
    X = pd.DataFrame(
        {
            "ip": [
                f"192.0.2.{i}" if i % 2 == 0 else f"2001:db8::{i:x}"
                for i in range(1, 65)
            ],
            "value": np.arange(64, dtype=float),
        }
    )
    X.loc[5, "ip"] = None
    y = pd.Series([0] * 32 + [1] * 32)
    automl = AutoML(
        results_path=str(tmp_path / algorithm),
        algorithms=[algorithm],
        total_time_limit=30,
        train_ensemble=False,
        stack_models=False,
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        start_random_models=1,
        hill_climbing_steps=0,
        explain_level=0,
        n_jobs=1,
        verbose=0,
    )
    automl.fit(X.copy(), y)
    assert "ip_transform" in automl._data_info["columns_info"]["ip"]
    new = pd.DataFrame(
        {"ip": ["192.0.2.200", "2001:db8::c8", None], "value": [100.0, 101.0, 102.0]}
    )
    expected = automl.predict_proba(new.copy())
    restored = AutoML(results_path=automl.results_path)
    np.testing.assert_allclose(expected, restored.predict_proba(new.copy()))
    with pytest.raises(ValueError, match="Invalid IP address in column 'ip'"):
        restored.predict(pd.DataFrame({"ip": ["not-an-ip"], "value": [100.0]}))
