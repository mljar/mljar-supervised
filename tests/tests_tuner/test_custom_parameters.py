import copy
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.base import clone

from supervised import AutoML
from supervised.tuner.custom_parameters import validate_custom_params
from tests.tests_tuner.test_realmlp import tuner


def test_names_and_input_not_mutated():
    options = {"RealMLP": {"device": "auto"}, "random_forest": {"max_depth": 3}}
    expected = copy.deepcopy(options)
    result = validate_custom_params(options, "binary_classification")
    assert result == {"realmlp": {"device": "auto"}, "randomforest": {"max_depth": 3}}
    result["realmlp"]["device"] = "cpu"
    assert options == expected
    assert clone(AutoML(custom_params=options)).custom_params == options


@pytest.mark.parametrize(
    "options",
    [
        [],
        {"unknown": {}},
        {"realmlp": 5},
        {"realmlp": {"devices": "cpu"}},
        {"randomforest": {"model_type": "RealMLP"}},
        {"randomforest": {"foo": 1}},
        {"RealMLP": {}, "real_mlp": {}},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        validate_custom_params(options, "binary_classification")


def test_existing_algorithm_fixed_overrides():
    t = tuner(
        algorithms=["Random Forest"], custom_params={"random_forest": {"max_depth": 3}}
    )
    default = t._get_model_params("Random Forest", 1, "default")
    assert default["learner"]["max_depth"] == 3
    for seed in range(5):
        assert t._get_model_params("Random Forest", seed)["learner"]["max_depth"] == 3
    model = SimpleNamespace(params=default, get_name=lambda: "1_RandomForest")
    frame = pd.DataFrame({"model_type": ["Random Forest"], "model": [model]})
    proposal = dict(default["learner"], max_depth=7)
    with patch.object(
        t, "df_models_algorithms", return_value=(frame, ["Random Forest"])
    ), patch("supervised.tuner.mljar_tuner.HillClimbing.get", return_value=[proposal]):
        candidates = t.get_hill_climbing_params([model])
    assert candidates[0]["learner"]["max_depth"] == 3
