"""Prediction reuse for one input batch and its ensemble dependencies."""

from collections import OrderedDict
from contextvars import ContextVar
import time
from functools import wraps

_current = ContextVar("mljar_prediction_context", default=None)


def current_prediction_context():
    return _current.get()


def prediction_scope(predict):
    """Also give direct ModelFramework predictions a bounded request lifetime."""

    @wraps(predict)
    def wrapped(*args, **kwargs):
        if current_prediction_context() is not None:
            return predict(*args, **kwargs)
        with PredictionContext():
            return predict(*args, **kwargs)

    return wrapped


class PredictionContext:
    """Keep intermediate predictions only for the lifetime of one request.

    Inputs are identified by object identity, not just model name or row count:
    original and stacked frames must never share an entry. Entries retain their
    inputs and model so Python cannot reuse their identities during the request.
    Callers must not mutate cached frames (copy before changing labels/columns).
    """

    def __init__(self):
        self._predictions = {}
        self.model_prediction_times = {}
        self._sdm_predictions = OrderedDict()
        self._sdm_prediction_bytes = 0
        self._network_key = None
        self._network = None

    def __enter__(self):
        self._token = _current.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._predictions.clear()
            self._sdm_predictions.clear()
            self._sdm_prediction_bytes = 0
            self.release_network()
        finally:
            _current.reset(self._token)

    def release_network(self):
        # SDM clears its mutable fold context in prediction_session's finally.
        # Only frozen weights may live here, never learners or encoded contexts.
        self._network = None
        self._network_key = None

    def network_for(self, key):
        if key != self._network_key:
            self.release_network()
        return self._network

    def keep_network(self, key, network):
        self._network_key, self._network = key, network

    def sdm_prediction(self, key):
        value = self._sdm_predictions.get(key)
        if value is not None:
            self._sdm_predictions.move_to_end(key)
            return value.copy()

    def keep_sdm_prediction(self, key, value, budget):
        # CPU predictions only; bounded independently of the number of models.
        budget = min(budget, 32 * 1024**2)
        previous = self._sdm_predictions.pop(key, None)
        if previous is not None:
            self._sdm_prediction_bytes -= previous.nbytes
        while (
            self._sdm_predictions and self._sdm_prediction_bytes + value.nbytes > budget
        ):
            _, old = self._sdm_predictions.popitem(last=False)
            self._sdm_prediction_bytes -= old.nbytes
        if value.nbytes <= budget:
            self._sdm_predictions[key] = value.copy()
            self._sdm_prediction_bytes += value.nbytes

    def predict(self, model, X, X_stacked=None):
        key = (id(model), id(X), id(X_stacked))
        if key not in self._predictions:
            started = time.perf_counter()
            if model.get_type() == "Ensemble":
                prediction = model.predict(X, X_stacked, prediction_context=self)
            else:
                prediction = model.predict(X)
                self.model_prediction_times[model.get_name()] = time.perf_counter() - started
            self._predictions[key] = (model, X, X_stacked, prediction)
        return self._predictions[key][-1]


def prediction_dependencies(model, stacked_models=()):
    """Unique framework models executed by a prediction, including stack inputs."""
    dependencies, visited = {}, set()

    def visit(current):
        name = current.get_name()
        if name in visited:
            return
        visited.add(name)
        if current._is_stacked:
            for base in stacked_models or ():
                visit(base)
        if current.get_type() == "Ensemble":
            for member in current.selected_models or ():
                visit(member["model"])
        else:
            dependencies[name] = current

    visit(model)
    return dependencies
