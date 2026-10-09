import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.preprocessing.preprocessing import Preprocessing
from supervised.tuner.data_info import DataInfo
from supervised.tuner.mljar_tuner import MljarTuner
from supervised.tuner.preprocessing_tuner import PreprocessingTuner


@pytest.mark.parametrize("algorithm", ["Neural Network", "Xgboost"])
@pytest.mark.parametrize(
    "task", ["binary_classification", "multiclass_classification", "regression"]
)
def test_switching_encoding_refreshes_scaling_without_mutating_original(
    algorithm, task
):
    X = pd.DataFrame(
        {
            "few": ["a", "b", "c"] * 30,
            "many": [f"value_{i}" for i in range(30)] * 3,
            "numeric": np.arange(90, dtype=float),
        }
    )
    y = pd.Series(np.arange(90) % (2 if task == "binary_classification" else 3))
    info = DataInfo.compute(X, y, task)
    required = AlgorithmsRegistry.registry[task][algorithm]["required_preprocessing"]
    preprocessing = PreprocessingTuner.get(required, info, task)
    # A non-categorical setting and the target configuration must survive.
    preprocessing["columns_preprocessing"]["numeric"] = ["scale_log_and_normal"]
    params = {
        "name": "original",
        "preprocessing": preprocessing,
        "learner": {"model_type": algorithm, "model_architecture_json": "old"},
        "validation_strategy": {},
    }
    original = copy.deepcopy(params)
    tuner = MljarTuner.__new__(MljarTuner)
    tuner._ml_task = task
    tuner._data_info = info
    tuner._optuna_time_budget = None
    tuner._unique_params_keys = []
    models = pd.DataFrame(
        {"model_type": [algorithm], "model": [SimpleNamespace(params=params)]}
    )
    with patch.object(
        tuner, "df_models_algorithms", return_value=(models, [algorithm])
    ):
        generated = tuner.get_mix_categorical_strategy([], None)
    assert len(generated) == 1
    changed = generated[0]
    steps = changed["preprocessing"]["columns_preprocessing"]
    assert steps["few"] == ["categorical_to_onehot"]
    assert steps["many"] == ["categorical_to_int"] + (
        ["scale_normal"] if algorithm == "Neural Network" else []
    )
    assert steps["numeric"] == ["scale_log_and_normal"]
    assert (
        changed["preprocessing"]["target_preprocessing"]
        == original["preprocessing"]["target_preprocessing"]
    )
    assert params == original
    assert "model_architecture_json" not in changed["learner"]
    actual, _, _ = Preprocessing(
        copy.deepcopy(changed["preprocessing"])
    ).fit_and_transform(X.copy(), y.copy())
    assert set(actual.columns) == {"few_a", "few_b", "few_c", "many", "numeric"}

    # Switching back adds scaling for NN integer codes and removes one-hot outputs.
    models["model"] = [SimpleNamespace(params=changed)]
    with patch.object(
        tuner, "df_models_algorithms", return_value=(models, [algorithm])
    ):
        reverted = tuner.get_all_int_categorical_strategy([], None)[0]
    assert reverted["preprocessing"]["columns_preprocessing"]["few"] == [
        "categorical_to_int"
    ] + (["scale_normal"] if algorithm == "Neural Network" else [])
    actual, _, _ = Preprocessing(
        copy.deepcopy(reverted["preprocessing"])
    ).fit_and_transform(X.copy(), y.copy())
    assert set(actual.columns) == {"few", "many", "numeric"}
