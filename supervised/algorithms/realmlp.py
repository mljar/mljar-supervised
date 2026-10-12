"""Optional RealMLP learner with MLJAR-owned validation and preprocessing."""

import copy
import importlib.util
import json
import time
import warnings
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import ClassifierMixin, RegressorMixin

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.registry import (
    AlgorithmsRegistry,
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.algorithms.realmlp_resources import (
    check_resources,
    thread_count,
    validate_resource_params,
)
from supervised.exceptions import AutoMLException


INSTALL_MESSAGE = 'Install RealMLP with pip install "mljar-supervised[realmlp]" (or "mljar-supervised[all]").'
SEARCH_PARAMS = {
    "n_hidden_layers": [1, 2, 3],
    "hidden_width": [32, 64, 128],
    "lr": [0.02, 0.04, 0.1, 0.2],
    "p_drop": [0.0, 0.15, 0.3],
    "wd": [0.001, 0.01, 0.02, 0.05],
}
OPTIONS = set(SEARCH_PARAMS) | {
    "device",
    "n_epochs",
    "batch_size",
    "predict_batch_size",
    "use_early_stopping",
    "early_stopping_additive_patience",
    "early_stopping_multiplicative_patience",
    "use_ls",
    "memory_fraction",
    "gpu_memory_fraction",
    "memory_limit_gb",
    "gpu_memory_limit_gb",
}


def validate_options(options):
    if options is None:
        return
    if not isinstance(options, dict):
        raise ValueError('custom_params["realmlp"] must be a dictionary or None')
    unknown = set(options) - OPTIONS
    if unknown:
        raise ValueError(f"Unknown RealMLP custom parameters: {sorted(unknown)}")
    for name in [
        "n_hidden_layers",
        "hidden_width",
        "n_epochs",
        "batch_size",
        "predict_batch_size",
    ]:
        if name in options and (type(options[name]) is not int or options[name] <= 0):
            raise ValueError(f"RealMLP {name} must be a positive integer")
    if options.get("n_hidden_layers", 2) not in [1, 2, 3]:
        raise ValueError("RealMLP supports 1, 2 or 3 hidden layers")
    if "hidden_width" in options and options["hidden_width"] not in [32, 64, 128]:
        raise ValueError("RealMLP hidden_width must be 32, 64 or 128")
    for name in ["lr", "wd", "p_drop"]:
        if name in options:
            value = options[name]
            if (
                not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"RealMLP {name} must be a finite nonnegative number")
    if options.get("lr", 0.04) <= 0 or options.get("p_drop", 0.15) >= 1:
        raise ValueError("RealMLP requires lr > 0 and 0 <= p_drop < 1")
    for name in ["use_early_stopping", "use_ls"]:
        if name in options and type(options[name]) is not bool:
            raise ValueError(f"RealMLP {name} must be boolean")
    for name in [
        "early_stopping_additive_patience",
        "early_stopping_multiplicative_patience",
    ]:
        if name in options and (
            not isinstance(options[name], (int, float))
            or not np.isfinite(options[name])
            or options[name] < 0
        ):
            raise ValueError(f"RealMLP {name} must be finite and nonnegative")
    device = options.get("device", "auto")
    if not isinstance(device, str) or not (
        device in ["auto", "cpu", "cuda"]
        or (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise ValueError("RealMLP device must be auto, cpu, cuda or cuda:<index>")
    validate_resource_params(options)


def dependencies_available():
    return all(importlib.util.find_spec(name) is not None for name in ["pytabkit", "torch"])


def check_dependencies():
    if not dependencies_available():
        raise AutoMLException(INSTALL_MESSAGE)
    from packaging.version import Version

    if not Version("1.7.2") <= Version(version("pytabkit")) < Version("1.8"):
        raise AutoMLException(
            "RealMLP requires pytabkit>=1.7.2,<1.8. " + INSTALL_MESSAGE
        )
    if not Version("2.7") <= Version(version("torch")) < Version("3"):
        raise AutoMLException("RealMLP requires torch>=2.7,<3. " + INSTALL_MESSAGE)


class RealMLPAlgorithm(BaseAlgorithm):
    algorithm_name = "RealMLP"
    algorithm_short_name = "RealMLP"
    supports_sample_weight = False

    def __init__(self, params):
        super().__init__(params)
        validate_options({k: v for k, v in params.items() if k in OPTIONS})
        self.max_iters = 1
        self.training_info = {}
        self.columns = None
        self.category_maps = {}
        self.numeric_means = {}
        self.indicators = []
        self.device = "cpu"

    @staticmethod
    def _frame(X):
        if sparse.issparse(X):
            raise ValueError(
                "RealMLP does not support sparse inputs; no automatic densification is performed."
            )
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if any(isinstance(dtype, pd.SparseDtype) for dtype in frame.dtypes):
            raise ValueError("RealMLP does not support pandas sparse columns.")
        if not frame.columns.is_unique:
            raise ValueError("RealMLP requires unique column names")
        return frame

    def _preprocess(self, X, fit=False):
        frame = self._frame(X)
        if fit:
            self.columns = list(frame.columns)
            self.category_maps, self.numeric_means, self.indicators = {}, {}, []
            for column in self.columns:
                series = frame[column]
                if pd.api.types.is_bool_dtype(
                    series.dtype
                ) or not pd.api.types.is_numeric_dtype(series.dtype):
                    # Fit only observed training values, not validation categories.
                    values = pd.unique(series.dropna())
                    self.category_maps[column] = {
                        value: i + 2 for i, value in enumerate(values)
                    }
                else:
                    numeric = (
                        pd.to_numeric(series)
                        .astype(float)
                        .replace([np.inf, -np.inf], np.nan)
                    )
                    self.numeric_means[column] = (
                        float(numeric.mean()) if numeric.notna().any() else 0.0
                    )
                    if numeric.isna().any():
                        self.indicators.append(column)
        if set(frame.columns) != set(self.columns):
            raise ValueError("RealMLP prediction columns must match training columns")
        result = {}
        # Internal names avoid collisions between user names and missing indicators.
        for i, column in enumerate(self.columns):
            series = frame[column]
            name = f"f{i}"
            if column in self.category_maps:
                mapping = self.category_maps[column]
                codes = series.astype(object).map(mapping).fillna(1).astype("int64")
                codes[series.isna()] = 0
                result[name] = pd.Categorical(codes, categories=range(len(mapping) + 2))
            else:
                numeric = (
                    pd.to_numeric(series, errors="raise")
                    .astype(float)
                    .replace([np.inf, -np.inf], np.nan)
                )
                result[name] = numeric.fillna(self.numeric_means[column]).to_numpy(
                    dtype=np.float32
                )
                if column in self.indicators:
                    result[f"missing{i}"] = pd.Categorical(
                        numeric.isna().astype(int), categories=[0, 1]
                    )
        return pd.DataFrame(result)

    def _metric(self):
        requested = self.params.get(
            "eval_metric_name", "rmse" if self.ml_task == REGRESSION else "logloss"
        )
        mapping = {
            "logloss": "cross_entropy",
            "auc": "1-auc_ovr_alt",
            "accuracy": "class_error",
            "rmse": "rmse",
            "mse": "rmse",
            "r2": "rmse",
            "mae": "mae",
        }
        metric = mapping.get(
            requested, "rmse" if self.ml_task == REGRESSION else "cross_entropy"
        )
        if requested not in mapping:
            warnings.warn(
                f"RealMLP uses {metric} for checkpoint selection; MLJAR still ranks models by {requested}."
            )
        return metric

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
                "RealMLP does not support training or validation sample weights."
            )
        if (X_validation is None) != (y_validation is None):
            raise ValueError("RealMLP requires both validation inputs and targets")
        check_dependencies()
        start = time.monotonic()
        import torch
        from supervised.algorithms._realmlp_backend import (
            RealMLPClassifier,
            RealMLPRegressor,
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
                f"RealMLP requested {device}, but it is unavailable. Select cpu or install compatible CUDA PyTorch."
            )
        self.device = device
        # Reject clearly oversized frames before copying/imputing them.
        from supervised.algorithms.realmlp_resources import (
            available_memory,
            RealMLPResourceError,
        )

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
            raise RealMLPResourceError(
                "RealMLP input/preprocessing memory estimate exceeds the RAM budget."
            )
        train = self._preprocess(X, fit=True)
        valid = None if X_validation is None else self._preprocess(X_validation)
        metric = self._metric()
        config = {
            k: self.params[k]
            for k in OPTIONS
            if k in self.params
            and k
            not in {
                "n_hidden_layers",
                "hidden_width",
                "device",
                "memory_fraction",
                "gpu_memory_fraction",
                "memory_limit_gb",
                "gpu_memory_limit_gb",
            }
        }
        config.update(
            hidden_sizes=[self.params.get("hidden_width", 64)]
            * self.params.get("n_hidden_layers", 2),
            device=device,
            n_threads=thread_count(self.params.get("n_jobs", -1)),
            random_state=self.params.get("seed", 1234),
            n_cv=1,
            n_refit=0,
            n_repeats=1,
            n_ens=1,
            val_fraction=0,
            val_metric_name=metric,
            verbosity=0,
        )
        config.setdefault("n_epochs", 256)
        config.setdefault("use_early_stopping", False)
        config.setdefault("early_stopping_additive_patience", 40)
        config.setdefault("early_stopping_multiplicative_patience", 3)
        if valid is None:
            config["use_early_stopping"] = False
        if self.ml_task != REGRESSION:
            self.classes_ = np.unique(y)
            config.setdefault(
                "use_ls", metric not in ["cross_entropy", "1-auc_ovr_alt"]
            )
        Model = RealMLPRegressor if self.ml_task == REGRESSION else RealMLPClassifier
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
            raise AutoMLException("RealMLP time budget exhausted before training.")
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
            raise RealMLPResourceError(
                "RealMLP ran out of GPU memory. No architecture or device was changed."
            ) from exc
        finally:
            torch.set_num_threads(old_threads)
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
        history = model.cv_alg_interface_.history
        if valid is not None and (
            not history or not any(np.isfinite(h["validation"]) for h in history)
        ):
            raise AutoMLException(
                "RealMLP finished without a finite validation checkpoint; increase its time budget or check the data."
            )
        best_epoch = None
        if valid is not None:
            best_epoch = int(
                model.cv_alg_interface_.fit_params[0]["stop_epoch"][metric]
            )
            if best_epoch < 1:
                raise AutoMLException(
                    "RealMLP did not complete a validation epoch; increase its time budget."
                )
        completed = history[-1]["epoch"] if history else None
        reason = "max_epochs"
        if completed is not None and completed < config["n_epochs"]:
            threshold = (
                config["early_stopping_multiplicative_patience"] * best_epoch
                + config["early_stopping_additive_patience"]
            )
            reason = (
                "early_stopping"
                if config["use_early_stopping"] and completed > threshold
                else "time_limit"
            )
        elif completed is None:
            reason = "finished_without_validation"
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
            hidden_sizes=config["hidden_sizes"],
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

    def predict(self, X):
        self.reload()
        import torch

        frame = self._frame(X)
        batch_size = self.params.get("predict_batch_size", 1024)
        outputs = []
        old_threads = torch.get_num_threads()
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            for start in range(0, len(frame), batch_size):
                batch = self._preprocess(frame.iloc[start : start + batch_size])
                outputs.append(
                    self.model.predict(batch)
                    if self.ml_task == REGRESSION
                    else self.model.predict_proba(batch)
                )
        finally:
            torch.set_num_threads(old_threads)
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
        if not outputs:
            return (
                np.empty((0, len(self.classes_)))
                if self.ml_task == MULTICLASS_CLASSIFICATION
                else np.empty(0)
            )
        prediction = np.concatenate(outputs)
        return prediction[:, 1] if self.ml_task == BINARY_CLASSIFICATION else prediction

    def is_fitted(self):
        return self.model is not None and hasattr(self.model, "alg_interface_")

    def predict_proba(self, X):
        prediction = self.predict(X)
        if self.ml_task == MULTICLASS_CLASSIFICATION:
            return prediction
        return np.column_stack((1 - prediction, prediction))

    def file_extension(self):
        return "realmlp"

    def get_metric_name(self):
        return self.training_info.get(
            "stopping_metric", self.params.get("eval_metric_name")
        )

    def get_params(self):
        result = super().get_params()
        result["training_info"] = {
            k: v for k, v in self.training_info.items() if k != "history"
        }
        return result

    def set_params(self, json_desc, learner_path):
        super().set_params(json_desc, learner_path)
        self.training_info = json_desc.get("training_info", {})

    def copy(self):
        self.model.to("cpu")
        try:
            result = copy.deepcopy(self)
            result.device = "cpu"
            return result
        finally:
            self.model.to(self.device)

    def save(self, model_file_path):
        self.model.to("cpu")
        try:
            payload = {
                key: value
                for key, value in self.__dict__.items()
                if key != "model_file_path"
            }
            payload["device"] = "cpu"
            joblib.dump(payload, model_file_path, compress=3)
        finally:
            self.model.to(self.device)
        self.model_file_path = model_file_path

    def load(self, model_file_path):
        check_dependencies()
        payload = joblib.load(model_file_path)
        self.__dict__.update(payload)
        self.model.to("cpu")
        self.device = "cpu"
        self.model.n_threads_ = thread_count(self.params.get("n_jobs", -1))
        self.model_file_path = model_file_path

    def interpret(
        self,
        X_train,
        y_train,
        X_validation,
        y_validation,
        model_file_path,
        learner_name,
        target_name=None,
        class_names=None,
        metric_name=None,
        ml_task=None,
        explain_level=2,
    ):
        if explain_level > 0:
            from supervised.utils.importance import PermutationImportance

            # Do not fork/copy CUDA learners into permutation-importance workers.
            X_val = X_validation
            y_val = y_validation
            if len(X_val) > 1000:
                from sklearn.model_selection import train_test_split

                X_val, _, y_val, _ = train_test_split(
                    X_val,
                    y_val,
                    train_size=1000,
                    random_state=123,
                    stratify=y_val if self.ml_task != REGRESSION else None,
                )
            PermutationImportance.compute_and_plot(
                self,
                X_val,
                y_val,
                model_file_path,
                learner_name,
                metric_name,
                ml_task,
                n_jobs=1,
            )


class RealMLPClassificationAlgorithm(ClassifierMixin, RealMLPAlgorithm):
    pass


class RealMLPRegressionAlgorithm(RegressorMixin, RealMLPAlgorithm):
    pass


for task, cls in [
    (BINARY_CLASSIFICATION, RealMLPClassificationAlgorithm),
    (MULTICLASS_CLASSIFICATION, RealMLPClassificationAlgorithm),
    (REGRESSION, RealMLPRegressionAlgorithm),
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
        {
            "n_hidden_layers": 2,
            "hidden_width": 64,
            "lr": 0.2 if task == REGRESSION else 0.04,
            "p_drop": 0.15,
            "wd": 0.02,
        },
    )
