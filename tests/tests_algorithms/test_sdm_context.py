"""Context lifetime, deadline and batching checks without pretrained weights."""

import numpy as np
import pandas as pd
import pytest

sdm = pytest.importorskip("sdm")
torch = pytest.importorskip("torch")

from supervised.algorithms.sdm import KumoClassificationAlgorithm
from supervised.algorithms._sdm_backend import (
    _autocast,
    _cache_fits_memory,
    _network_lock,
)


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.models = torch.nn.ModuleDict({"classification": torch.nn.Identity()})
        self._cache = None
        self.fits = 0
        self.queries = []
        self.uncached = 0
        self.failure = None
        self.expire = None

    def fit(self, context, target, **kwargs):
        assert self._cache is None
        self._cache = object()
        self.fits += 1
        self.fit_options = kwargs
        if self.expire:
            self.expire()
        self.models["classification"](context.numerical)
        if self.failure == "fit":
            raise RuntimeError("fit failed")

    def predict(self, query, **kwargs):
        assert self._cache is not None
        if self.failure == "predict":
            raise RuntimeError("predict failed")
        self.queries.append(query.numerical.shape[-2])
        return self.output(query)

    def forward(self, *, x_query, **kwargs):
        self.uncached += 1
        return self.output(x_query)

    def output(self, query):
        p = torch.sigmoid(query.numerical[:, 0])
        return sdm.TableTensor(
            columns={sdm.Stype.numerical: ["1", "0"]},
            numerical=torch.stack([p, 1 - p], dim=-1),
        )

    def clear(self):
        self._cache = None


@pytest.fixture
def learner(monkeypatch):
    monkeypatch.setattr(
        "supervised.algorithms.realmlp_resources.available_memory", lambda: 16 * 1024**3
    )
    m = KumoClassificationAlgorithm(
        {
            "ml_task": "binary_classification",
            "seed": 42,
            "predict_batch_size": 4,
            "num_estimators": 3,
            "n_jobs": 1,
        }
    )
    m.context = m._preprocess(pd.DataFrame({"a": [-1.0, 1.0]}), fit=True)
    m.context_y = np.array([0, 1])
    m.classes_ = np.array([0, 1])
    m.model = Network()
    return m


def test_context_is_prepared_once_per_request_and_cleared(learner):
    X = pd.DataFrame({"a": np.arange(9, dtype=float)})
    for call in (1, 2):
        actual = learner.predict(X)
        np.testing.assert_allclose(actual, torch.sigmoid(torch.arange(9.0)).numpy())
        assert learner.model.fits == call
        assert learner.model._cache is None
    assert learner.model.queries == [4] * 6
    assert learner.model.fit_options["num_estimators"] == 3
    assert learner.model.fit_options["estimator_batch_size"] == 1
    assert learner.model.fit_options["generator"].initial_seed() == 42
    assert not learner.model.models["classification"]._forward_hooks
    assert _network_lock(learner.model) is _network_lock(learner.copy().model)
    assert learner.predict(X.iloc[:0]).shape == (0,)
    assert learner.model.fits == 2


@pytest.mark.parametrize("failure", ["fit", "predict", "deadline"])
def test_failed_requests_clear_context_and_deadline_hooks(learner, failure):
    learner.model.failure = failure
    if failure == "deadline":
        learner.model.expire = lambda: setattr(learner, "_deadline", 0)
    with pytest.raises(Exception, match="failed|time budget"):
        learner.predict(pd.DataFrame({"a": [0.0, 1.0]}))
    assert learner.model._cache is None
    assert not learner.model.models["classification"]._forward_hooks
    learner.model.failure = None
    learner.model.expire = None
    learner._deadline = None
    assert np.isfinite(learner.predict(pd.DataFrame({"a": [0.0]}))).all()


def test_low_memory_uses_uncached_inference_without_changing_predictions(learner):
    X = pd.DataFrame({"a": np.arange(7, dtype=float)})
    expected = learner.predict(X)
    assert _cache_fits_memory(learner)
    learner.params["memory_limit_gb"] = 0.001
    assert not _cache_fits_memory(learner)
    np.testing.assert_array_equal(expected, learner.predict(X))
    assert learner.model.fits == 1
    assert learner.model.uncached == 2
    assert learner.model._cache is None


def test_context_is_rebuilt_for_changed_context_and_shared_network(learner):
    other = learner.copy()
    assert other.model is learner.model
    other.context_y[:] = [1, 0]
    np.testing.assert_array_equal(learner.context_y, [0, 1])
    for model in (learner, other, learner):
        model.predict(pd.DataFrame({"a": [0.0]}))
        assert model.model._cache is None
    assert learner.model.fits == 3


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("precision", [None, "fp16", "fp32"])
def test_precision_setting_and_restoration_after_failure(
    learner, monkeypatch, device, precision
):
    from contextlib import nullcontext
    from unittest.mock import Mock

    learner.device = device
    if precision is not None:
        learner.params["precision"] = precision
    autocast = Mock(return_value=nullcontext())
    monkeypatch.setattr(torch.amp, "autocast", autocast)
    # PyTorch exposes this setting dynamically, so restore it explicitly:
    # monkeypatch's static attribute inspection can mistake it for a new attribute.
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with pytest.raises(RuntimeError, match="interrupted"):
            with _autocast(learner):
                assert torch.backends.cuda.matmul.allow_tf32 == (
                    device == "cpu" or precision != "fp32"
                )
                raise RuntimeError("interrupted")
        assert torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
    autocast.assert_called_once_with(
        device.split(":")[0],
        dtype=torch.float16,
        enabled=device.startswith("cuda") and precision != "fp32",
    )
