"""Verify the published installation, CUDA and pretrained model files before timing."""
from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import sys

from common import CANDIDATE_VERSIONS, NEW_ALGORITHMS, now, write_json


def check(version, device, output, download=True):
    import supervised
    from supervised import AutoML

    installed = importlib.metadata.version("mljar-supervised")
    module = Path(supervised.__file__).resolve()
    if installed != version or supervised.__version__ != version or not module.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Wrong MLJAR installation: requested {version}, metadata {installed}, module {module}")
    info = dict(checked=now(), version=installed, module=str(module), python=sys.version,
                executable=sys.executable, requested_device=device, cuda_available=False,
                packages={dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()})
    if version in CANDIDATE_VERSIONS:
        if version in ("1.4.0rc3", "1.4.0rc4", "1.4.0rc5"):
            from supervised.preprocessing.feature_discovery_transformer import check_dependencies as check_discovery
            check_discovery()
            info["features_goldmine_version"] = importlib.metadata.version("features-goldmine")
            print(f"Feature discovery: features-goldmine {info['features_goldmine_version']}", flush=True)
        import torch
        from supervised.algorithms.sdm import SDMAlgorithm, check_dependencies
        check_dependencies()
        info["cuda_available"] = torch.cuda.is_available()
        if device == "cuda" and not info["cuda_available"]:
            raise RuntimeError("CUDA is unavailable. Check the driver, or explicitly use --device cpu.")
        if device == "cuda":
            torch.ones(1, device="cuda").add_(1)
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            info.update(gpu=torch.cuda.get_device_name(), vram_free_bytes=free, vram_total_bytes=total)
        resolved = AutoML(mode="Compete")._get_algorithms()
        if device == "cuda" and not set(NEW_ALGORITHMS) <= set(resolved):
            raise RuntimeError(f"New algorithms missing from Compete: {resolved}")
        if download:
            from supervised.algorithms.sdm import (
                KumoClassificationAlgorithm, KumoRegressionAlgorithm,
                TabICLv2ClassificationAlgorithm, TabICLv2RegressionAlgorithm,
            )
            checkpoints = []
            for cls in (KumoClassificationAlgorithm, KumoRegressionAlgorithm,
                        TabICLv2ClassificationAlgorithm, TabICLv2RegressionAlgorithm):
                for size in (["small", "medium", "large"] if cls.algorithm_short_name == "Kumo" else [None]):
                    params = {"ml_task": "regression" if "Regression" in cls.__name__ else "binary_classification"}
                    if size:
                        params["size"] = size
                    model = cls(params)
                    print(f"Preparing pretrained weights: {cls.__name__} {size or ''}", flush=True)
                    # Download exactly the pinned files that this published release uses.
                    SDMAlgorithm._download(model)
                    checkpoints.append(dict(model.checkpoint_info, path=model.checkpoint_path,
                                            bytes=Path(model.checkpoint_path).stat().st_size))
            info["checkpoints"] = checkpoints
    else:
        resolved = AutoML(mode="Compete")._get_algorithms()
    info["available_algorithms"] = resolved
    if version == "1.3.2":
        info["selection_note"] = "1.3.2 disables Neural Network during fit when n_jobs is explicitly limited."
    write_json(output, info)
    print(f"Verified MLJAR {installed}: {module}", flush=True)
    print(f"Device: {device}; algorithms: {', '.join(resolved)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    check(args.version, args.device, args.output, not args.no_download)
