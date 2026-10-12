"""Optional TabM learner with MLJAR-owned validation and preprocessing."""

import copy
import importlib.util
import json
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.registry import (
    AlgorithmsRegistry,
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.algorithms.tabm_resources import (
    check_resources,
)
from supervised.exceptions import AutoMLException
from supervised.algorithms.realmlp import RealMLPAlgorithm
from supervised.algorithms.realmlp_resources import (
    thread_count,
    validate_resource_params,
)

INSTALL_MESSAGE = 'Install TabM with pip install "mljar-supervised[tabm]" (or "mljar-supervised[all]").'
SEARCH_PARAMS = {
    "n_blocks": [2, 3],
    "d_block": [64, 128],
    "lr": [0.0001, 0.0003, 0.001, 0.002, 0.003],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "weight_decay": [0.0, 0.0001, 0.0003, 0.001, 0.01],
}
DEFAULT_PARAMS = dict(
    n_blocks=2, d_block=128, lr=0.002, dropout=0.1, weight_decay=0.0003
)
OPTIONS = set(SEARCH_PARAMS) | {
    "device",
    "n_epochs",
    "batch_size",
    "predict_batch_size",
    "patience",
    "tabm_k",
    "num_emb_n_bins",
    "d_embedding",
    "memory_fraction",
    "gpu_memory_fraction",
    "memory_limit_gb",
    "gpu_memory_limit_gb",
}


def validate_options(options):
    if options is None:
        return
    if not isinstance(options, dict):
        raise ValueError('custom_params["tabm"] must be a dictionary')
    unknown = set(options) - OPTIONS
    if unknown:
        raise ValueError(f"Unknown TabM custom parameters: {sorted(unknown)}")
    for key in [
        "n_blocks",
        "d_block",
        "n_epochs",
        "predict_batch_size",
        "tabm_k",
        "num_emb_n_bins",
        "d_embedding",
    ]:
        if key in options and (type(options[key]) is not int or options[key] <= 0):
            raise ValueError(f"TabM {key} must be a positive integer")
    for key, values in {
        "n_blocks": [2, 3],
        "d_block": [64, 128],
        "tabm_k": [16, 32],
    }.items():
        if key in options and options[key] not in values:
            raise ValueError(f"TabM {key} must be one of {values}")
    if "num_emb_n_bins" in options and options["num_emb_n_bins"] < 2:
        raise ValueError("TabM num_emb_n_bins must be at least 2")
    batch = options.get("batch_size", "auto")
    if batch != "auto" and (type(batch) is not int or batch <= 0):
        raise ValueError("TabM batch_size must be auto or a positive integer")
    patience = options.get("patience", 16)
    if type(patience) is not int or patience < 0:
        raise ValueError("TabM patience must be a nonnegative integer")
    for key in ["lr", "weight_decay", "dropout"]:
        if key in options and (
            type(options[key]) not in [int, float]
            or not np.isfinite(options[key])
            or options[key] < 0
        ):
            raise ValueError(f"TabM {key} must be finite and nonnegative")
    if options.get("lr", 0.002) == 0 or options.get("dropout", 0.1) >= 1:
        raise ValueError("TabM requires lr > 0 and dropout < 1")
    device = options.get("device", "auto")
    if not isinstance(device, str) or not (
        device in ["auto", "cpu", "cuda"]
        or (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise ValueError("TabM device must be auto, cpu, cuda or cuda:<index>")
    validate_resource_params(options)


def dependencies_available():
    return all(
        importlib.util.find_spec(name) is not None for name in ["pytabkit", "torch"]
    )


def check_dependencies():
    if not dependencies_available():
        raise AutoMLException(INSTALL_MESSAGE)
    from packaging.version import Version

    if not Version("1.7.2") <= Version(version("pytabkit")) < Version("1.8"):
        raise AutoMLException("TabM requires pytabkit>=1.7.2,<1.8. " + INSTALL_MESSAGE)
    if not Version("2.7") <= Version(version("torch")) < Version("3"):
        raise AutoMLException("TabM requires torch>=2.7,<3. " + INSTALL_MESSAGE)


class TabMAlgorithm(RealMLPAlgorithm):
    algorithm_name = "TabM mini"
    algorithm_short_name = "TabM"

    def __init__(self, params):
        BaseAlgorithm.__init__(self, params)
        validate_options({k: v for k, v in params.items() if k in OPTIONS})
        if params.get("arch_type", "tabm-mini") != "tabm-mini":
            raise ValueError("Only TabM mini is supported")
        self.max_iters = 1
        self.training_info = {}
        self.columns = None
        self.category_maps = {}
        self.numeric_means = {}
        self.indicators = []
        self.device = "cpu"

    def file_extension(self):
        return "tabm"

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        X_validation=None,
        y_validation=None,
        sample_weight_validation=None,
        log_to_file=None,
        max_time=None,
    ):
        if sample_weight is not None or sample_weight_validation is not None:
            raise ValueError(
                "TabM does not support training or validation sample weights."
            )
        if X_validation is None or y_validation is None:
            raise ValueError("TabM requires both validation inputs and targets")
        check_dependencies()
        start = time.monotonic()
        import torch
        from supervised.algorithms._tabm_backend import (
            TabMClassifier,
            TabMRegressor,
        )

        self.library_version = version("pytabkit")
        device = self.params.get("device", "auto")
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            device = "cuda:0"
        if device.startswith("cuda") and (
            not torch.cuda.is_available()
            or int(device.split(":")[1]) >= torch.cuda.device_count()
        ):
            raise AutoMLException(
                f"TabM requested {device}, but it is unavailable. Select cpu or install compatible CUDA PyTorch."
            )
        self.device = device
        # Reject clearly oversized frames before copying/imputing them.
        from supervised.algorithms.tabm_resources import TabMResourceError
        from supervised.algorithms.realmlp_resources import available_memory

        raw_bytes = sum(
            (
                int(v.memory_usage(deep=True).sum())
                if isinstance(v, pd.DataFrame)
                else int(getattr(v, "nbytes", 0))
            )
            for v in [X, X_validation]
            if v is not None
        )
        budget = available_memory() * self.params.get("memory_fraction", 0.6)
        if self.params.get("memory_limit_gb") is not None:
            budget = min(budget, self.params["memory_limit_gb"] * 1024**3)
        if 6 * raw_bytes > budget:
            raise TabMResourceError(
                "TabM input/preprocessing memory estimate exceeds the RAM budget."
            )
        train = self._preprocess(X, fit=True)
        valid = None if X_validation is None else self._preprocess(X_validation)
        metric = self._metric()
        config = dict(DEFAULT_PARAMS)
        config.update(
            tabm_k=16,
            num_emb_n_bins=48,
            d_embedding=16,
            batch_size="auto",
            n_epochs=256,
            patience=16,
        )
        config.update(
            {
                k: v
                for k, v in self.params.items()
                if k in OPTIONS
                and k
                not in {
                    "device",
                    "predict_batch_size",
                    "memory_fraction",
                    "gpu_memory_fraction",
                    "memory_limit_gb",
                    "gpu_memory_limit_gb",
                }
            }
        )
        config.update(
            arch_type="tabm-mini",
            num_emb_type="pwl",
            device=device,
            n_threads=thread_count(self.params.get("n_jobs", -1)),
            random_state=self.params.get("seed", 1234),
            n_cv=1,
            n_refit=0,
            n_repeats=1,
            val_fraction=0,
            val_metric_name=metric,
            verbosity=0,
            compile_model=False,
            allow_amp=False,
        )
        if self.ml_task != REGRESSION:
            self.classes_ = np.unique(y)
        Model = TabMRegressor if self.ml_task == REGRESSION else TabMClassifier
        model = Model(**config)
        resources = check_resources(
            model,
            train,
            valid,
            0 if self.ml_task == REGRESSION else len(self.classes_),
            device,
            self.params,
        )
        remaining = None if max_time is None else max_time - (time.monotonic() - start)
        if remaining is not None and remaining <= 0:
            raise AutoMLException("TabM time budget exhausted before training.")
        model.deadline = None if max_time is None else start + max_time
        old_threads = torch.get_num_threads()
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            model.fit(
                train,
                np.asarray(y).ravel(),
                X_val=valid,
                y_val=(
                    None if y_validation is None else np.asarray(y_validation).ravel()
                ),
                cat_col_names=list(train.select_dtypes(include="category").columns),
                time_to_fit_in_seconds=remaining,
            )
        except torch.cuda.OutOfMemoryError as exc:
            del model
            torch.cuda.empty_cache()
            raise TabMResourceError(
                "TabM ran out of GPU memory. No architecture or device was changed."
            ) from exc
        finally:
            torch.set_num_threads(old_threads)
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
        backend = model.cv_alg_interface_.sub_split_interfaces[0]
        history = backend.history
        best_epoch = backend.best_epoch
        completed = len(history)
        reason = backend.stop_reason
        self.model = model
        if self.ml_task != REGRESSION:
            self.classes_ = model.classes_
        self.training_info = dict(
            resources=resources,
            best_epoch=best_epoch,
            completed_epochs=completed,
            stop_reason=reason,
            stopping_metric=metric,
            elapsed_seconds=time.monotonic() - start,
            hidden_sizes=[config["d_block"]] * config["n_blocks"],
            arch_type="tabm-mini",
            tabm_k=config["tabm_k"],
            pytabkit_version=self.library_version,
            torch_version=torch.__version__,
            history=history,
        )
        if log_to_file is not None:
            rows = [(h["epoch"], np.nan, h["validation"]) for h in history]
            if not rows:
                rows = [(0, np.nan, np.nan)]
            pd.DataFrame(rows).to_csv(log_to_file, index=False, header=False)
            with open(log_to_file + ".json", "w") as fout:
                json.dump(self.training_info, fout, indent=2)


class TabMClassificationAlgorithm(ClassifierMixin, TabMAlgorithm):
    pass


class TabMRegressionAlgorithm(RegressorMixin, TabMAlgorithm):
    pass


for task, cls in [
    (BINARY_CLASSIFICATION, TabMClassificationAlgorithm),
    (MULTICLASS_CLASSIFICATION, TabMClassificationAlgorithm),
    (REGRESSION, TabMRegressionAlgorithm),
]:
    preprocessing = ["datetime_transform", "text_transform", "ip_transform"]
    if task != REGRESSION:
        preprocessing.append("target_as_integer")
    AlgorithmsRegistry.add(
        task,
        cls,
        copy.deepcopy(SEARCH_PARAMS),
        preprocessing,
        {
            "max_rows_limit": None,
            "max_cols_limit": None,
            "supports_sample_weight": False,
        },
        copy.deepcopy(DEFAULT_PARAMS),
    )
