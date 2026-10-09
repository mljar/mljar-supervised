from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from supervised import AutoML
from supervised.utils.importance import PermutationImportance
from supervised.utils.shap import PlotSHAP


def test_large_knn_automl_skips_interpretation_but_trains_and_predicts(tmp_path):
    X = pd.DataFrame(
        np.random.default_rng(1234).normal(size=(14000, 2)), columns=["a", "b"]
    )
    y = (X["a"] > 0).astype(int)
    results = tmp_path / "knn"
    automl = AutoML(
        results_path=str(results),
        algorithms=["Nearest Neighbors"],
        validation_strategy={
            "validation_type": "split",
            "train_ratio": 0.75,
            "shuffle": True,
            "stratify": True,
        },
        total_time_limit=30,
        train_ensemble=False,
        stack_models=False,
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        start_random_models=1,
        hill_climbing_steps=0,
        explain_level=2,
        n_jobs=1,
        verbose=0,
    )
    with patch.object(
        PermutationImportance, "compute_and_plot"
    ) as importance, patch.object(PlotSHAP, "compute") as shap:
        with pytest.warns(
            UserWarning,
            match="Skipping kNN interpretation: training rows=10500, validation rows=3500; row limit=10000",
        ):
            automl.fit(X.copy(), y)
        importance.assert_not_called()
        shap.assert_not_called()
    assert automl.predict(X.iloc[:10].copy()).shape == (10,)
    restored = AutoML(results_path=str(results))
    np.testing.assert_array_equal(
        automl.predict(X.iloc[:10].copy()), restored.predict(X.iloc[:10].copy())
    )
    assert list(results.rglob("learning_curves.png"))
    assert not list(results.rglob("*_importance.csv"))
