"""Conservative TabM-mini admission estimates, before allocating data tensors."""

from supervised.algorithms.realmlp_resources import (
    available_gpu_memory,
    available_memory,
    gpu_allocation_stats,
    thread_count,
)
from supervised.exceptions import AutoMLException, ResourceAdmissionError


class TabMResourceError(AutoMLException):
    pass


class TabMAdmissionError(TabMResourceError, ResourceAdmissionError):
    pass


def check_resources(model, X, X_validation, n_classes, device, params):
    import torch

    cfg = model.get_config()
    rows = len(X) + len(X_validation)
    cats = X.select_dtypes(include="category").columns
    numeric = X.shape[1] - len(cats)
    categories = sum(len(X[c].cat.categories) + 1 for c in cats)
    width, depth, k = cfg["d_block"], cfg["n_blocks"], cfg["tabm_k"]
    embedding, bins = cfg["d_embedding"], min(cfg["num_emb_n_bins"], len(X) - 1)
    batch = cfg["batch_size"]
    if batch == "auto":
        from pytabkit.models.alg_interfaces.tabm_interface import (
            get_tabm_auto_batch_size,
        )

        batch = get_tabm_auto_batch_size(len(X))
    batch = max(batch, 1024)  # upstream prediction/validation batch
    inputs = numeric * embedding + categories
    parameters = (
        numeric * (bins + 1) * embedding
        + inputs * (width + k)
        + depth * width**2
        + k * width * max(1, n_classes)
    )
    tensors = rows * (numeric * 4 + len(cats) * 8 + 8)
    # Parameters, gradients, Adam moments, best checkpoint; training activations;
    # independent per-member permutations, including sorting intermediates.
    working = 6 * 4 * parameters + 8 * batch * k * (
        inputs + numeric * bins + 3 * depth * width + max(1, n_classes)
    )
    working += 24 * len(X) * k
    frame_bytes = int(
        X.memory_usage(deep=True).sum() + X_validation.memory_usage(deep=True).sum()
    )
    host_need = int(
        2 * 1024**3
        + 6 * frame_bytes
        + 3 * tensors
        + (working if device == "cpu" else 0)
    )
    host_free = available_memory()
    budget = min(
        host_free * params.get("memory_fraction", 0.6),
        (params.get("memory_limit_gb") or float("inf")) * 1024**3,
    )
    info = dict(
        device=device,
        n_threads=thread_count(params.get("n_jobs", -1)),
        rows=rows,
        features=X.shape[1],
        estimated_ram_bytes=host_need,
        available_ram_bytes=host_free,
        ram_budget_bytes=int(budget),
    )
    if host_need > budget:
        raise TabMAdmissionError(
            "TabM estimated RAM exceeds its memory budget; use fewer rows/features or a smaller network",
            resources=info,
        )
    if device.startswith("cuda"):
        free, total = available_gpu_memory(torch.device(device))
        need = int(1024**3 + 1.5 * (2 * tensors + working))
        budget = min(
            free * params.get("gpu_memory_fraction", 0.7),
            (params.get("gpu_memory_limit_gb") or float("inf")) * 1024**3,
        )
        info.update(
            estimated_vram_bytes=need,
            free_vram_bytes=free,
            total_vram_bytes=total,
            vram_budget_bytes=int(budget),
        )
        if need > budget:
            info.update(gpu_allocation_stats(device))
            raise TabMAdmissionError(
                f"TabM estimated GPU memory {need / 1024**3:.2f} GiB exceeds "
                f"budget {budget / 1024**3:.2f} GiB on {device} "
                f"({free / 1024**3:.2f} GiB free); use a smaller configuration "
                "or explicitly select CPU",
                resources=info,
            )
    return info
