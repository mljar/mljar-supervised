import inspect
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.factory import AlgorithmFactory
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.sklearn import SklearnAlgorithm
from supervised.exceptions import AutoMLException

REQUIRED_METHODS = (
    "fit",
    "predict",
    "is_fitted",
    "file_extension",
    "copy",
    "save",
    "load",
)


def test_base_and_shared_adapter_cannot_be_instantiated():
    with pytest.raises(TypeError, match="abstract"):
        BaseAlgorithm({})
    with pytest.raises(TypeError, match="file_extension"):
        SklearnAlgorithm({})


@pytest.mark.parametrize("missing_method", REQUIRED_METHODS)
def test_incomplete_subclass_fails_before_training(missing_method):
    def implementation(self, *args, **kwargs):
        return None

    methods = {
        name: implementation for name in REQUIRED_METHODS if name != missing_method
    }
    incomplete = type("IncompleteAlgorithm", (BaseAlgorithm,), methods)
    with pytest.raises(TypeError, match=missing_method):
        incomplete({})


@pytest.mark.parametrize(
    "task,name,info",
    [
        (task, name, info)
        for task, algorithms in AlgorithmsRegistry.registry.items()
        for name, info in algorithms.items()
    ],
    ids=[
        f"{task}-{name}"
        for task, algorithms in AlgorithmsRegistry.registry.items()
        for name in algorithms
    ],
)
def test_registered_algorithms_construct_through_factory(task, name, info):
    assert not inspect.isabstract(info["class"])
    params = dict(info["default_params"], ml_task=task, model_type=name, n_jobs=1)
    learner = AlgorithmFactory.get_algorithm(params)
    assert isinstance(learner, BaseAlgorithm)
    assert learner.file_extension()


def test_factory_reports_incomplete_algorithm():
    with patch.object(
        AlgorithmsRegistry, "get_algorithm_class", return_value=SklearnAlgorithm
    ):
        with pytest.raises(AutoMLException, match="file_extension"):
            AlgorithmFactory.get_algorithm({"model_type": "Incomplete"})


@pytest.mark.parametrize("lazy_load", [False, True])
def test_shared_defaults_and_factory_reload(tmp_path, lazy_load):
    params = {"model_type": "Baseline", "ml_task": "binary_classification"}
    learner = AlgorithmFactory.get_algorithm(params)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1])
    learner.fit(X, y)
    expected = learner.predict(X)
    assert learner.update({}) is None
    assert learner.get_metric_name() is None
    assert_allclose(learner.predict_proba(X), np.column_stack((1 - expected, expected)))

    copied = learner.copy()
    assert copied is not learner
    assert copied.model is not learner.model
    assert_allclose(copied.predict(X), expected)

    model_path = str(tmp_path / learner.get_fname())
    learner.save(model_path)
    loaded = AlgorithmFactory.load(learner.get_params(), model_path, lazy_load)
    assert loaded.is_fitted() is (not lazy_load)
    assert_allclose(loaded.predict(X), expected)
    assert loaded.is_fitted()
    with patch.object(loaded, "load") as load:
        loaded.reload()
        load.assert_not_called()
