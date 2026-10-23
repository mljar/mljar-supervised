"""Bounded fold residency and prediction reuse across an ensemble/stack graph."""

import weakref

import numpy as np
import pandas as pd
import pytest

from supervised import AutoML
from supervised.algorithms.baseline import BaselineRegressorAlgorithm
from supervised.ensemble import Ensemble
from supervised.model_framework import ModelFramework
from supervised.prediction import PredictionContext


class ResidentModel:
    pass


class TrackingLearner(BaselineRegressorAlgorithm):
    def __init__(self, live, *, saved=True, failure=None):
        super().__init__({"ml_task": "regression"})
        self.model = None
        self.model_file_path = "saved-learner" if saved else None
        self.live = live
        self.failure = failure
        self.loads = 0
        if not saved:
            self.model = ResidentModel()

    def is_fitted(self):
        return self.model is not None

    def load(self, path):
        assert not self.live, "A previous fold is still resident"
        self.model = ResidentModel()
        self.live.add(self.model)
        self.loads += 1
        if self.failure == "load":
            raise RuntimeError("load failed")

    def predict(self, X):
        self.reload()
        if self.failure == "predict":
            raise RuntimeError("predict failed")
        return X["feature"].to_numpy()


class FoldPreprocessing:
    def __init__(self, offset, failure=None):
        self.offset = offset
        self.failure = failure

    def transform(self, X, y):
        if self.failure == "preprocess":
            raise RuntimeError("preprocess failed")
        X["feature"] += self.offset
        return X, y, None

    def inverse_scale_target(self, y):
        if self.failure == "inverse":
            raise RuntimeError("inverse failed")
        return 2 * y

    def prepare_target_labels(self, y):
        return pd.DataFrame({"prediction": y})


def framework(*, saved=True, failure=None):
    model = ModelFramework({"learner": {}, "validation_strategy": {}})
    live = weakref.WeakSet()
    model.learners = [TrackingLearner(live, saved=saved) for _ in range(10)]
    model.preprocessings = [FoldPreprocessing(i) for i in range(10)]
    model.learners[1].failure = failure
    model.preprocessings[1].failure = failure
    return model, live


def test_saved_folds_are_streamed_and_can_predict_again():
    model, live = framework()
    X = pd.DataFrame({"feature": [1.0, 3.0]}, index=[10, 20])
    original = X.copy()
    for _ in range(2):
        result = model.predict(X)
        np.testing.assert_allclose(result.prediction, [11.0, 15.0])
        assert not live
        assert all(learner.model is None for learner in model.learners)
        pd.testing.assert_frame_equal(X, original)
    assert all(learner.loads == 2 for learner in model.learners)


@pytest.mark.parametrize("failure", ["load", "predict", "preprocess", "inverse"])
def test_failed_fold_releases_its_model_and_can_retry(failure):
    model, live = framework(failure=failure)
    X = pd.DataFrame({"feature": [1.0]})
    with pytest.raises(RuntimeError, match=failure):
        model.predict(X)
    assert not live
    assert all(learner.model is None for learner in model.learners)
    model.learners[1].failure = None
    model.preprocessings[1].failure = None
    np.testing.assert_allclose(model.predict(X).prediction, [11.0])


def test_unsaved_folds_are_kept_for_subsequent_predictions():
    model, _ = framework(saved=False)
    models = [learner.model for learner in model.learners]
    X = pd.DataFrame({"feature": [2.0]})
    pd.testing.assert_frame_equal(model.predict(X), model.predict(X))
    assert [learner.model for learner in model.learners] == models


class CountingModel:
    _threshold = 0.5

    def __init__(self, name, predict, stacked=False):
        self.name = name
        self.function = predict
        self._is_stacked = stacked
        self.calls = 0

    def get_type(self):
        return "CountingModel"

    def get_name(self):
        return self.name

    def predict(self, X):
        self.calls += 1
        return self.function(X).reset_index(drop=True)


def ensemble(name, members, task="regression", stacked=False):
    model = Ensemble(ml_task=task, is_stacked=stacked)
    model._name = name
    model._threshold = 0.5
    model.selected_models = [
        {"model": member, "repeat": weight} for member, weight in members
    ]
    return model


def test_stacking_and_nested_ensembles_reuse_predictions_per_request():
    a = CountingModel("A", lambda X: pd.DataFrame({"prediction": X.feature}))
    b = CountingModel("B", lambda X: pd.DataFrame({"prediction": 2 * X.feature}))
    base = ensemble("Ensemble", [(a, 1), (b, 2)])

    def predict_stacked(X):
        assert X.columns.tolist() == ["feature", "Ensemble_prediction", "A_prediction"]
        assert X.index.tolist() == [0, 1]
        return pd.DataFrame({"prediction": X.A_prediction + X.Ensemble_prediction})

    stacked = CountingModel("Stacked", predict_stacked, stacked=True)
    best = ensemble("Ensemble_Stacked", [(a, 2), (base, 1), (stacked, 3)], stacked=True)
    automl = AutoML()
    automl._ml_task = "regression"
    automl._data_info = {"columns": ["feature"], "n_features": 1}
    automl.n_features_in_ = 1
    automl._best_model = best
    automl._stacked_models = [base, a]
    X = pd.DataFrame({"feature": [1.0, 3.0]})
    for call in [1, 2]:
        original = X.copy()
        result = automl.predict(X)
        np.testing.assert_allclose(result, X.feature * 35 / 18)
        assert [a.calls, b.calls, stacked.calls] == [call] * 3
        pd.testing.assert_frame_equal(X, original)
        # Same DataFrame identity, different data: the next request must be fresh.
        X["feature"] += 1


@pytest.mark.parametrize("classes", [2, 3])
def test_nested_classification_ensembles_preserve_probabilities_and_labels(classes):
    columns = [f"prediction_{c}" for c in ["cat", "dog", "bird"][:classes]]
    p = (
        np.array([[0.2, 0.8], [0.7, 0.3]])
        if classes == 2
        else np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    )
    a = CountingModel("same-name", lambda X: pd.DataFrame(p, columns=columns))
    b = CountingModel("same-name", lambda X: pd.DataFrame(p[:, ::-1], columns=columns))
    task = "binary_classification" if classes == 2 else "multiclass_classification"
    inner = ensemble("Ensemble", [(a, 1), (b, 1)], task)
    outer = ensemble("Outer", [(a, 1), (inner, 2)], task)
    automl = AutoML()
    automl._ml_task = task
    automl._data_info = {"columns": ["feature"], "n_features": 1}
    automl.n_features_in_ = 1
    automl._best_model = outer
    expected = (2 * p + p[:, ::-1]) / 3
    result = automl.predict_all(pd.DataFrame({"feature": [1.0, 2.0]}))
    np.testing.assert_allclose(result[columns], expected)
    np.testing.assert_array_equal(
        result.label, np.array(["cat", "dog", "bird"])[expected.argmax(axis=1)]
    )
    assert a.calls == b.calls == 1


def test_cache_distinguishes_input_frames_and_clears_after_failure():
    a = CountingModel("A", lambda X: pd.DataFrame({"prediction": X.feature}))
    X = pd.DataFrame({"feature": [1.0]})
    X_stacked = pd.DataFrame({"feature": [7.0]})
    context = PredictionContext()
    with pytest.raises(RuntimeError, match="downstream failed"):
        with context:
            assert context.predict(a, X).prediction.iloc[0] == 1
            assert context.predict(a, X_stacked).prediction.iloc[0] == 7
            cached = weakref.ref(context.predict(a, X))
            assert a.calls == 2
            raise RuntimeError("downstream failed")
    assert cached() is None
    with context:
        assert context.predict(a, X).prediction.iloc[0] == 1
    assert a.calls == 3
