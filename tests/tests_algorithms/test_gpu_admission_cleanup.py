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
