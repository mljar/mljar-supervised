import importlib.util
from contextlib import contextmanager

import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional torch")
def test_gpu_memory_check_collects_cycles_then_releases_cache(monkeypatch):
    import torch
    from supervised.algorithms import realmlp_resources

    events = []

    @contextmanager
    def device(name):
        events.append(("device", name))
        yield

    monkeypatch.setattr(
        realmlp_resources.gc, "collect", lambda: events.append("collect")
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))

    def free(name):
        events.append(("measure", name))
        return 4 * 1024**3, 8 * 1024**3

    monkeypatch.setattr(torch.cuda, "mem_get_info", free)
    assert realmlp_resources.available_gpu_memory("cuda:1") == (
        4 * 1024**3,
        8 * 1024**3,
    )
    assert events == [
        "collect",
        ("device", "cuda:1"),
        "empty_cache",
        ("measure", "cuda:1"),
    ]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional torch")
def test_fast_gpu_memory_probe_never_collects_or_empties_cache(monkeypatch):
    import torch
    from supervised.algorithms import realmlp_resources

    monkeypatch.setattr(
        realmlp_resources.gc, "collect", lambda: pytest.fail("Unexpected GC")
    )
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: pytest.fail("Unexpected cache cleanup")
    )
    devices = []

    def free(device):
        devices.append(device)
        return 3 * 1024**3, 8 * 1024**3

    monkeypatch.setattr(torch.cuda, "mem_get_info", free)
    assert realmlp_resources.available_gpu_memory("cuda:1", cleanup=False) == (
        3 * 1024**3,
        8 * 1024**3,
    )
    assert devices == ["cuda:1"]


@pytest.fixture(
    params=["KumoClassificationAlgorithm", "TabICLv2ClassificationAlgorithm"]
)
def admission(request, monkeypatch):
    from supervised.algorithms import sdm, realmlp_resources

    model = getattr(sdm, request.param)(
        {"ml_task": "binary_classification", "precision": "fp16"}
    )
    model.device = "cuda:1"
    monkeypatch.setattr(sdm, "available_memory", lambda: 100 * 1024**3)
    resources = (128, 10, 64 * 1024**2)
    required = (
        512 * 1024**2
        + resources[2]
        + sdm.working_memory_bytes(
            model.params, model.algorithm_short_name, *resources[:2], model.device
        )
    )
    calls = []

    def with_free_memory(*values):
        readings = iter(values)

        def measure(device, *, cleanup=True):
            calls.append((device, cleanup))
            return next(readings), 16 * 1024**3

        monkeypatch.setattr(realmlp_resources, "available_gpu_memory", measure)

    return model, resources, required, calls, with_free_memory


def test_sdm_sufficient_memory_uses_only_fast_probe(admission):
    model, resources, required, calls, measure = admission
    measure(16 * 1024**3)
    info = model._check_resources(*resources)
    assert info["estimated_vram_bytes"] == required
    assert calls == [("cuda:1", False)]


def test_sdm_reclaims_memory_then_rechecks_same_estimate(admission):
    model, resources, required, calls, measure = admission
    measure(0, 16 * 1024**3)
    info = model._check_resources(*resources)
    assert info["estimated_vram_bytes"] == required
    assert calls == [("cuda:1", False), ("cuda:1", True)]


def test_sdm_genuine_shortage_reports_latest_measurement(admission):
    from supervised.exceptions import AutoMLException

    model, resources, required, calls, measure = admission
    first, after_cleanup = required / 2, required / 4
    measure(first, after_cleanup)
    params = model.params.copy()
    with pytest.raises(AutoMLException, match="estimated VRAM exceeds budget") as exc:
        model._check_resources(*resources)
    message = str(exc.value)
    assert f"{required / 1024**3:.2f} GiB needed" in message
    assert f"{after_cleanup * 0.7 / 1024**3:.2f} GiB budget" in message
    assert f"({after_cleanup / 1024**3:.2f} GiB free)" in message
    assert calls == [("cuda:1", False), ("cuda:1", True)]
    assert model.device == "cuda:1"
    assert model.params == params


@pytest.mark.parametrize("fraction", [None, 0.5])
def test_sdm_memory_fraction_applies_before_and_after_cleanup(admission, fraction):
    model, resources, required, calls, measure = admission
    if fraction is not None:
        model.params["gpu_memory_fraction"] = fraction
    fraction = fraction or 0.7
    # Raw free memory is sufficient, but the configured allowance is not.
    measure(required * 1.01, required / fraction + 1024)
    model._check_resources(*resources)
    assert calls == [("cuda:1", False), ("cuda:1", True)]


def test_sdm_explicit_limit_is_enforced_even_after_cleanup(admission):
    from supervised.exceptions import AutoMLException

    model, resources, required, calls, measure = admission
    model.params["gpu_memory_limit_gb"] = (required - 1) / 1024**3
    measure(16 * 1024**3, 16 * 1024**3)
    with pytest.raises(AutoMLException, match="estimated VRAM exceeds budget"):
        model._check_resources(*resources)
    assert calls == [("cuda:1", False), ("cuda:1", True)]


def test_sdm_exact_budget_is_accepted(admission):
    model, resources, required, calls, measure = admission
    model.params.update(gpu_memory_fraction=0.5, gpu_memory_limit_gb=required / 1024**3)
    measure(2 * required)
    model._check_resources(*resources)
    assert calls == [("cuda:1", False)]


def test_sdm_rechecks_memory_for_each_fold(admission):
    from supervised.exceptions import AutoMLException

    model, resources, _, calls, measure = admission
    measure(16 * 1024**3, 0, 0)
    model._check_resources(*resources)
    with pytest.raises(AutoMLException, match="estimated VRAM exceeds budget"):
        model._check_resources(*resources)
    assert calls == [("cuda:1", False), ("cuda:1", False), ("cuda:1", True)]


def test_sdm_cpu_and_ram_rejection_do_not_probe_gpu(admission):
    from supervised.exceptions import AutoMLException

    model, resources, _, calls, measure = admission
    measure()  # Any GPU probe would fail.
    model.device = "cpu"
    assert model._check_resources(*resources)["estimated_vram_bytes"] == 0
    model.device = "cuda:1"
    model.params["memory_limit_gb"] = 0.001
    with pytest.raises(AutoMLException, match="estimated RAM exceeds budget"):
        model._check_resources(*resources)
    assert not calls
