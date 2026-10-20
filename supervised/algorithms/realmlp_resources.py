"""Resource admission checks for RealMLP (no torch import at module scope)."""

import math
import gc
from pathlib import Path

from joblib import cpu_count

from supervised.exceptions import AutoMLException


class RealMLPResourceError(AutoMLException):
    pass


def available_memory():
    import psutil

    available = psutil.virtual_memory().available
    # Linux cgroup v2 and v1. Also inspect ancestors for nested containers.
    roots = [Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, location = line.split(":", 2)
            root = roots[0] if not controllers else roots[1]
            if not controllers or "memory" in controllers.split(","):
                candidate = root / location.lstrip("/")
                roots.extend(
                    [candidate, *[p for p in candidate.parents if root in p.parents]]
                )
    except (OSError, ValueError):
        pass
    for root in roots:
        for limit_name, used_name in [
            ("memory.max", "memory.current"),
            ("memory.limit_in_bytes", "memory.usage_in_bytes"),
        ]:
            try:
                limit = int((root / limit_name).read_text().strip())
                used = int((root / used_name).read_text().strip())
                available = min(available, max(0, limit - used))
            except (OSError, ValueError):
                pass
    return available


def thread_count(n_jobs):
    available = max(1, cpu_count())  # respects affinity and cgroup quotas
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs == 0:
        raise ValueError("RealMLP n_jobs must be a nonzero integer")
    return min(available, n_jobs) if n_jobs > 0 else max(1, available + 1 + n_jobs)


def available_gpu_memory(device, *, cleanup=True):
    """Measure driver memory, optionally releasing unreachable/cached allocations."""
    import torch

    if cleanup:
        gc.collect()
        if torch.cuda.is_initialized():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    return torch.cuda.mem_get_info(device)


def check_resources(model, X, X_validation, n_classes, device, params):
    """Use pytabkit's architecture-aware estimate without allocating data tensors."""
    import torch
    from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface
    from pytabkit.models.data.data import DictDataset, TensorInfo

    n_rows = len(X) + (0 if X_validation is None else len(X_validation))
    cats = X.select_dtypes(include="category").columns
    cat_sizes = [len(X[c].cat.categories) + 1 for c in cats]
    ds = DictDataset(
        tensors=None,
        tensor_infos={
            "x_cont": TensorInfo(feat_shape=[X.shape[1] - len(cats)]),
            "x_cat": TensorInfo(cat_sizes=cat_sizes),
            "y": TensorInfo(cat_sizes=[n_classes]),
        },
        device="cpu",
        n_samples=n_rows,
    )
    config = dict(model.get_config(), use_gpu=device.startswith("cuda"))
    estimate = NNAlgInterface(**config).get_required_resources(
        ds, n_cv=1, n_refit=0, n_splits=1, split_seeds=[0], n_train=len(X)
    )
    frame_bytes = int(X.memory_usage(deep=True).sum())
    if X_validation is not None:
        frame_bytes += int(X_validation.memory_usage(deep=True).sum())
    # Account for framework copies/conversion and overhead beyond upstream tensors.
    # CUDA context/Lightning imports also consume host RAM. GPU smoke runs on an
    # RTX 3070 reached ~1.7 GiB RSS even for tiny networks; do not use the much
    # smaller CPU-only process allowance for those fits.
    overhead = 2 * 1024**3 if device.startswith("cuda") else 512 * 1024**2
    host_need = int(estimate.cpu_ram_gb * 1024**3 * 1.5 + 6 * frame_bytes + overhead)
    host_free = available_memory()
    host_budget = host_free * params.get("memory_fraction", 0.6)
    if params.get("memory_limit_gb") is not None:
        host_budget = min(host_budget, params["memory_limit_gb"] * 1024**3)
    info = dict(
        device=device,
        n_threads=thread_count(params.get("n_jobs", -1)),
        rows=n_rows,
        features=X.shape[1],
        estimated_ram_bytes=host_need,
        available_ram_bytes=host_free,
        ram_budget_bytes=int(host_budget),
    )
    if host_need > host_budget:
        raise RealMLPResourceError(
            f"RealMLP estimated RAM {host_need / 1024**3:.2f} GiB exceeds "
            f"budget {host_budget / 1024**3:.2f} GiB. Use fewer features/rows or "
            "a smaller network, or explicitly adjust the memory budget."
        )
    if device.startswith("cuda"):
        free, total = available_gpu_memory(torch.device(device))
        need = int(estimate.gpu_ram_gb * 1024**3 * 1.5)
        budget = free * params.get("gpu_memory_fraction", 0.7)
        if params.get("gpu_memory_limit_gb") is not None:
            budget = min(budget, params["gpu_memory_limit_gb"] * 1024**3)
        info.update(
            estimated_vram_bytes=need,
            free_vram_bytes=free,
            total_vram_bytes=total,
            vram_budget_bytes=int(budget),
        )
        if need > budget:
            raise RealMLPResourceError(
                f"RealMLP estimated GPU memory {need / 1024**3:.2f} GiB exceeds "
                f"budget {budget / 1024**3:.2f} GiB on {device}. "
                "Use a smaller configuration or explicitly select CPU."
            )
    return info


def validate_resource_params(params):
    for name in ["memory_fraction", "gpu_memory_fraction"]:
        value = params.get(name, 0.6 if name == "memory_fraction" else 0.7)
        if (
            not isinstance(value, (float, int))
            or not math.isfinite(value)
            or not 0 < value <= 1
        ):
            raise ValueError(f"{name} must be in (0, 1]")
    for name in ["memory_limit_gb", "gpu_memory_limit_gb"]:
        value = params.get(name)
        if value is not None and (
            not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive")
