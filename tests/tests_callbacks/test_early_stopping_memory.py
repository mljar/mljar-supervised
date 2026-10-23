import gc
import weakref

import numpy as np

from supervised.algorithms.sdm import (
    KumoClassificationAlgorithm,
    TabICLv2ClassificationAlgorithm,
)
from supervised.callbacks.early_stopping import EarlyStopping

import pytest


@pytest.mark.parametrize(
    "cls", [KumoClassificationAlgorithm, TabICLv2ClassificationAlgorithm]
)
def test_completed_learner_releases_shared_network_but_keeps_predictions(cls):
    class Network:
        pass

    learner = cls({"ml_task": "binary_classification"})
    learner.model = Network()
    network = weakref.ref(learner.model)
    callback = EarlyStopping({"metric": {"name": "logloss"}})
    callback.add_and_set_learner(learner)
    callback.on_learner_train_start(None)
    callback.on_iteration_end(
        {"iter_cnt": 0},
        dict(
            y_validation_true=np.array([0, 1]),
            y_validation_predicted=np.array([0.2, 0.8]),
            validation_index=np.arange(2),
        ),
    )
    assert callback.best_models[learner.uid].model is learner.model
    callback.on_learner_train_end(None)
    learner.model = None  # ModelFramework saves/unloads the original here.
    gc.collect()
    assert network() is None
    callback.on_framework_train_end(None)
    assert callback.final_loss > 0
    np.testing.assert_array_equal(callback.best_y_oof.prediction, [0.2, 0.8])
