"""Optional NVIDIA SDM in-context learners (imports stay lightweight)."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.realmlp import RealMLPAlgorithm
from supervised.algorithms.realmlp_resources import (
    available_memory,
    thread_count,
    validate_resource_params,
)
from supervised.algorithms.registry import (
    AlgorithmsRegistry,
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.exceptions import AutoMLException

SDM_REVISION = "b97cb914e2bdcf0d165fc2c83b5fdf795aad6a69"
SDM_VERSION = "0.2.0rc2.dev18+gb97cb914e"
INSTALL_MESSAGE = (
    "Install mljar-supervised[tfm] and the tested SDM backend: pip install "
    '"git+https://github.com/NVIDIA/structured-data-models.git@'
    + SDM_REVISION
    + '". Python >=3.11 is required.'
)
CHECKPOINTS = {
    "Kumo": ("nvidia/Kumo-Tabular", "3c3e10bbdb590ace29e7026847f92db3c603096d"),
    "TabICLv2": ("jingang/TabICL", "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"),
}
SEARCH_PARAMS = {"num_estimators": [4, 8, 16], "max_context_size": [1024, 4096, 8192]}
DEFAULT_PARAMS = {"num_estimators": 8, "max_context_size": 4096}
OPTIONS = set(DEFAULT_PARAMS) | {
    "device",
    "predict_batch_size",
    "memory_fraction",
    "gpu_memory_fraction",
    "memory_limit_gb",
    "gpu_memory_limit_gb",
    "local_files_only",
    "cache_dir",
}


def validate_options(options, algorithm):
    allowed = OPTIONS | ({"size"} if algorithm == "Kumo" else set())
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown {algorithm} custom parameters: {sorted(unknown)}")
    for key in ["num_estimators", "max_context_size", "predict_batch_size"]:
        if key in options and (type(options[key]) is not int or options[key] <= 0):
            raise ValueError(f"{algorithm} {key} must be a positive integer")
    if options.get("size", "small") not in ["small", "medium", "large"]:
        raise ValueError("Kumo size must be small, medium or large")
    device = options.get("device", "auto")
    if not isinstance(device, str) or not (
        device in ["auto", "cpu", "cuda"]
        or (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise ValueError(f"{algorithm} device must be auto, cpu, cuda or cuda:<index>")
    if "local_files_only" in options and type(options["local_files_only"]) is not bool:
        raise ValueError("local_files_only must be boolean")
    if "cache_dir" in options and not isinstance(options["cache_dir"], str):
        raise ValueError("cache_dir must be a string")
    validate_resource_params(options)


def dependencies_available():
    return sys.version_info >= (3, 11) and all(
        importlib.util.find_spec(n) is not None
        for n in ["sdm", "torch", "huggingface_hub"]
    )


def check_dependencies():
    if not dependencies_available():
        raise AutoMLException(INSTALL_MESSAGE)
    if version("structured-data-models") != SDM_VERSION:
        raise AutoMLException(
            "This integration requires the tested SDM revision. " + INSTALL_MESSAGE
        )
    from packaging.version import Version

    if not Version("2.7") <= Version(version("torch")) < Version("3"):
        raise AutoMLException("SDM requires torch>=2.7,<3. " + INSTALL_MESSAGE)


def automatic_available():
    if not dependencies_available():
        return False
    try:
        check_dependencies()
        import torch

        return torch.cuda.is_available()
    except (AutoMLException, ImportError):
        return False


class SDMAlgorithm(RealMLPAlgorithm):
    """Pretrained inference with bounded, fold-local context and no fine-tuning."""

    def __init__(self, params):
        BaseAlgorithm.__init__(self, params)
        validate_options(
            {k: v for k, v in params.items() if k in OPTIONS | {"size"}},
            self.algorithm_short_name,
        )
        self.max_iters = 1
        self.training_info = {}
        self.columns = None
        self.category_maps = {}
        self.numeric_means = {}
        self.indicators = []
        self.device = "cpu"
        self._deadline = None

    def file_extension(self):
        return self.algorithm_short_name.lower()

    def is_fitted(self):
        return self.model is not None and hasattr(self, "context")

    def _check_time(self):
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise AutoMLException(
                f"{self.algorithm_short_name} time budget exhausted during context/validation inference; increase model_time_limit."
            )

    def _context_indices(self, y):
        rng = np.random.default_rng(self.params.get("seed", 1234))
        limit = min(len(y), self.params.get("max_context_size", 4096))
        if self.ml_task == REGRESSION:
            return rng.choice(len(y), size=limit, replace=False)
        # Guarantee a representative of every class, including singleton classes.
        classes = np.unique(y)
        if limit < len(classes):
            raise ValueError("max_context_size must be at least the number of classes")
        first = np.array([rng.choice(np.flatnonzero(y == c)) for c in classes])
        remaining = np.setdiff1d(np.arange(len(y)), first)
        result = np.concatenate(
            [first, rng.choice(remaining, size=limit - len(first), replace=False)]
        )
        rng.shuffle(result)
        return result

    def _check_resources(self, rows, cols, checkpoint_bytes=0):
        """Conservative admission estimate, not a hardware-independent size guarantee."""
        batch = self.params.get("predict_batch_size", 128)
        members = self.params.get("num_estimators", 8)
        # Cell embeddings, attention workspaces, model/loading copies, and host frames.
        # Larger Kumo checkpoints also widen/deepen attention and embeddings.
        size_factor = (
            {"small": 1, "medium": 3, "large": 6}[self.params.get("size", "small")]
            if self.algorithm_short_name == "Kumo"
            else 1
        )
        working = (rows + batch) * max(cols, 1) * 128 * 4 * 8 * size_factor
        host = (
            512 * 1024**2 + 3 * checkpoint_bytes + working + members * rows * cols * 16
        )
        budget = available_memory() * self.params.get("memory_fraction", 0.6)
        if self.params.get("memory_limit_gb") is not None:
            budget = min(budget, self.params["memory_limit_gb"] * 1024**3)
        if host > budget:
            raise AutoMLException(
                f"{self.algorithm_short_name} estimated RAM exceeds budget; reduce max_context_size/predict_batch_size."
            )
        gpu = 0
        if self.device.startswith("cuda"):
            from supervised.algorithms.realmlp_resources import available_gpu_memory

            free, _ = available_gpu_memory(self.device)
            gpu = 512 * 1024**2 + checkpoint_bytes + working
            gpu_budget = free * self.params.get("gpu_memory_fraction", 0.7)
            if self.params.get("gpu_memory_limit_gb") is not None:
                gpu_budget = min(
                    gpu_budget, self.params["gpu_memory_limit_gb"] * 1024**3
                )
            if gpu > gpu_budget:
                raise AutoMLException(
                    f"{self.algorithm_short_name} estimated VRAM exceeds budget: "
                    f"{gpu / 1024**3:.2f} GiB needed, {gpu_budget / 1024**3:.2f} GiB "
                    f"budget on {self.device} ({free / 1024**3:.2f} GiB free); "
                    "reduce max_context_size/predict_batch_size."
                )
        return {"estimated_ram_bytes": host, "estimated_vram_bytes": gpu}

    def _download(self):
        from huggingface_hub import hf_hub_download

        repo, revision = CHECKPOINTS[self.algorithm_short_name]
        task = "regressor" if self.ml_task == REGRESSION else "classifier"
        filename = (
            f"{self.params.get('size', 'small')}/{task}.pt"
            if self.algorithm_short_name == "Kumo"
            else f"tabicl-{task}-v2-20260212.ckpt"
        )
        kwargs = dict(
            repo_id=repo,
            revision=revision,
            local_files_only=self.params.get("local_files_only", False),
            cache_dir=self.params.get("cache_dir"),
        )

        # Prefer cached artifacts without a network metadata request.
        def download(name):
            from huggingface_hub.errors import LocalEntryNotFoundError

            try:
                return hf_hub_download(
                    filename=name, **dict(kwargs, local_files_only=True)
                )
            except LocalEntryNotFoundError:
                return hf_hub_download(filename=name, **kwargs)

        self.checkpoint_path = download(filename)
        self.notice_paths = [download("README.md")]
        if self.algorithm_short_name == "Kumo":
            self.notice_paths.append(download("LICENSE"))
        else:
            import sdm.models.tabiclv2

            self.notice_paths.append(
                str(Path(sdm.models.tabiclv2.__file__).parent / "LICENSE")
            )
        self.checkpoint_info = dict(repo=repo, revision=revision, filename=filename)

    def _load_network(self):
        from supervised.algorithms._sdm_backend import load_network

        self.model = load_network(
            self.algorithm_short_name,
            self.ml_task,
            self.checkpoint_path,
            self.device,
            size=self.params.get("size", "small"),
        )

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
                f"{self.algorithm_short_name} does not support sample weights"
            )
        if X_validation is None or y_validation is None:
            raise ValueError("SDM requires validation inputs and targets")
        check_dependencies()
        import torch

        self.model = None
        self.__dict__.pop("_validation_prediction", None)
        self.__dict__.pop("_evaluation_deadline", None)
        started = time.monotonic()
        self._deadline = None if max_time is None else started + max_time
        self.device = self.params.get("device", "auto")
        if self.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.device = "cuda:0"
        if self.device.startswith("cuda") and (
            not torch.cuda.is_available()
            or int(self.device.split(":")[1]) >= torch.cuda.device_count()
        ):
            raise AutoMLException(f"Requested device {self.device} is unavailable")
        # Validate without copying the full training frame: only selected context is retained.
        from scipy import sparse

        if sparse.issparse(X):
            raise ValueError("SDM does not support sparse inputs")
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if any(isinstance(t, pd.SparseDtype) for t in frame.dtypes):
            raise ValueError("SDM does not support sparse inputs")
        targets = np.asarray(y).ravel()
        if len(targets) != len(frame) or not len(targets):
            raise ValueError("SDM requires nonempty matching inputs and targets")
        if pd.isna(targets).any():
            raise ValueError("SDM targets cannot contain missing values")
        if self.ml_task != REGRESSION:
            self.classes_, targets = np.unique(targets, return_inverse=True)
            if len(self.classes_) < 2:
                raise ValueError("SDM classification requires at least two classes")
        elif not np.isfinite(targets).all():
            raise ValueError("SDM regression targets must be finite")
        indices = self._context_indices(targets)
        self._check_resources(len(indices), frame.shape[1])
        self.context = self._preprocess(frame.iloc[indices], fit=True)
        self.context_y = targets[indices]
        if not self.context.shape[1] or not any(self.context.nunique(dropna=False) > 1):
            raise ValueError("SDM requires at least one nonconstant context feature")
        # Kumo adds count features for high-cardinality categories before its 500-column cap.
        effective_cols = self.context.shape[1]
        if self.algorithm_short_name == "Kumo":
            effective_cols += sum(
                self.context[c].nunique() >= 50
                for c in self.context.select_dtypes("category")
            )
            if effective_cols > 500:
                raise ValueError(
                    "Kumo supports at most 500 effective columns; refusing silent feature truncation"
                )
        self._check_time()
        self._download()
        resources = self._check_resources(
            len(indices), effective_cols, Path(self.checkpoint_path).stat().st_size
        )
        self._check_time()
        self._load_network()
        self.library_version = version("structured-data-models")
        try:
            prediction = self.predict(X_validation)
            from supervised.utils.metric import Metric

            metric_name = self.params.get(
                "eval_metric_name", "rmse" if self.ml_task == REGRESSION else "logloss"
            )
            validation_target = np.asarray(y_validation).ravel()
            if self.ml_task != REGRESSION:
                mapping = {c: i for i, c in enumerate(self.classes_)}
                if any(c not in mapping for c in validation_target):
                    raise ValueError(
                        "SDM validation contains a class absent from the training fold"
                    )
                validation_target = np.array([mapping[c] for c in validation_target])
            if metric_name == "logloss" and self.ml_task != REGRESSION:
                from sklearn.metrics import log_loss

                score = float(
                    log_loss(
                        validation_target,
                        prediction,
                        labels=np.arange(len(self.classes_)),
                    )
                )
            else:
                score = float(
                    Metric({"name": metric_name})(validation_target, prediction)
                )
            self._check_time()
        except Exception:
            self.model = None
            raise
        finally:
            self._deadline = None
        self._evaluation_deadline = None if max_time is None else started + max_time
        self._validation_prediction = prediction
        self.training_info = dict(
            context_rows=len(indices),
            training_rows=len(frame),
            effective_columns=effective_cols,
            num_estimators=self.params.get("num_estimators", 8),
            size=(
                self.params.get("size", "small")
                if self.algorithm_short_name == "Kumo"
                else None
            ),
            device=self.device,
            checkpoint=self.checkpoint_info,
            resources=resources,
            validation_score=score,
            stopping_metric=metric_name,
            elapsed_seconds=time.monotonic() - started,
            stop_reason="pretrained_inference",
            sdm_version=self.library_version,
            torch_version=torch.__version__,
        )
        if log_to_file:
            display_score = -score if Metric.optimize_negative(metric_name) else score
            pd.DataFrame([(0, np.nan, display_score)]).to_csv(
                log_to_file, index=False, header=False
            )
            Path(log_to_file + ".json").write_text(
                json.dumps(self.training_info, indent=2)
            )
        return self

    def training_predictions(self, X_train, X_validation):
        # The framework's train/OOF scoring is part of the learner budget too.
        self._deadline = getattr(self, "_evaluation_deadline", None)
        try:
            train_prediction = self.predict(X_train)
            validation_prediction = self._validation_prediction
            self._check_time()
            return train_prediction, validation_prediction
        finally:
            self._deadline = None
            self.__dict__.pop("_validation_prediction", None)
            self.__dict__.pop("_evaluation_deadline", None)

    def predict(self, X):
        self.reload()
        if not self.is_fitted():
            raise AutoMLException("SDM model is not fitted")
        import torch
        from supervised.algorithms._sdm_backend import predict_batch

        outputs = []
        batch_size = self.params.get("predict_batch_size", 128)
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(thread_count(self.params.get("n_jobs", -1)))
            for start in range(0, len(X), batch_size):
                self._check_time()
                batch = (
                    X.iloc[start : start + batch_size]
                    if isinstance(X, pd.DataFrame)
                    else X[start : start + batch_size]
                )
                outputs.append(predict_batch(self, self._preprocess(batch)))
                self._check_time()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise AutoMLException(
                "SDM ran out of GPU memory; reduce max_context_size or predict_batch_size. No device or context was changed automatically."
            ) from exc
        finally:
            torch.set_num_threads(old_threads)
        if not outputs:
            return (
                np.empty((0, len(self.classes_)))
                if self.ml_task == MULTICLASS_CLASSIFICATION
                else np.empty(0)
            )
        result = np.concatenate(outputs)
        if not np.isfinite(result).all():
            raise AutoMLException("SDM returned nonfinite predictions")
        return result[:, 1] if self.ml_task == BINARY_CLASSIFICATION else result

    def copy(self):
        # Frozen network can be shared; all fold-specific state is copied independently.
        result = copy.copy(self)
        result.__dict__ = copy.deepcopy(
            {k: v for k, v in self.__dict__.items() if k != "model"}
        )
        result.model = self.model
        return result

    def save(self, model_file_path, weights_root=None):
        parent = Path(model_file_path).resolve().parent
        root = Path(weights_root).resolve() if weights_root is not None else parent
        destination = (
            root
            / "sdm_weights"
            / (self.algorithm_short_name + "-" + self.checkpoint_info["revision"])
        )
        if self.algorithm_short_name == "Kumo":
            # All sizes use classifier.pt/regressor.pt: never share across sizes.
            destination = destination / self.params.get("size", "small")
        destination.mkdir(parents=True, exist_ok=True)
        weight = destination / Path(self.checkpoint_path).name
        if (
            not weight.exists()
            or weight.stat().st_size != Path(self.checkpoint_path).stat().st_size
        ):
            # Publish complete files atomically when folds share a destination.
            with tempfile.NamedTemporaryFile(dir=destination, delete=False) as temp:
                temporary = Path(temp.name)
            try:
                shutil.copyfile(self.checkpoint_path, temporary)
                os.replace(temporary, weight)
            finally:
                temporary.unlink(missing_ok=True)
        notices = []
        for i, source in enumerate(self.notice_paths):
            path = destination / f"{i}-{Path(source).name}"
            if not path.exists():
                shutil.copyfile(source, path)
            notices.append(os.path.relpath(path, parent))
        payload = {
            k: v
            for k, v in self.__dict__.items()
            if k
            not in {
                "model",
                "model_file_path",
                "checkpoint_path",
                "notice_paths",
                "_validation_prediction",
                "_evaluation_deadline",
            }
        }
        payload.update(
            checkpoint_relative=os.path.relpath(weight, parent),
            notice_relative=notices,
            _deadline=None,
        )
        joblib.dump(payload, model_file_path, compress=3)
        self.model_file_path = model_file_path

    def load(self, model_file_path):
        check_dependencies()
        payload = joblib.load(model_file_path)
        parent = Path(model_file_path).resolve().parent
        self.checkpoint_path = str(parent / payload.pop("checkpoint_relative"))
        self.notice_paths = [str(parent / p) for p in payload.pop("notice_relative")]
        self.__dict__.update(payload)
        import torch

        if self.device.startswith("cuda") and (
            not torch.cuda.is_available()
            or int(self.device.split(":")[1]) >= torch.cuda.device_count()
        ):
            self.device = "cpu"
        self._check_resources(
            len(self.context),
            self.training_info.get("effective_columns", self.context.shape[1]),
            Path(self.checkpoint_path).stat().st_size,
        )
        self._load_network()  # Always local: never downloads when reloading an artifact.
        self.model_file_path = model_file_path


class KumoClassificationAlgorithm(ClassifierMixin, SDMAlgorithm):
    algorithm_name = algorithm_short_name = "Kumo"


class KumoRegressionAlgorithm(RegressorMixin, SDMAlgorithm):
    algorithm_name = algorithm_short_name = "Kumo"


class TabICLv2ClassificationAlgorithm(ClassifierMixin, SDMAlgorithm):
    algorithm_name = algorithm_short_name = "TabICLv2"


class TabICLv2RegressionAlgorithm(RegressorMixin, SDMAlgorithm):
    algorithm_name = algorithm_short_name = "TabICLv2"


for classification, regression in [
    (KumoClassificationAlgorithm, KumoRegressionAlgorithm),
    (TabICLv2ClassificationAlgorithm, TabICLv2RegressionAlgorithm),
]:
    for task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, REGRESSION]:
        preprocessing = ["datetime_transform", "text_transform", "ip_transform"]
        if task != REGRESSION:
            preprocessing.append("target_as_integer")
        defaults = dict(DEFAULT_PARAMS)
        search = copy.deepcopy(SEARCH_PARAMS)
        if classification.algorithm_short_name == "Kumo":
            defaults["size"] = "small"
            search["size"] = ["small", "medium", "large"]
        AlgorithmsRegistry.add(
            task,
            regression if task == REGRESSION else classification,
            search,
            preprocessing,
            {
                "max_rows_limit": None,
                "max_cols_limit": None,
                "supports_sample_weight": False,
            },
            defaults,
        )
