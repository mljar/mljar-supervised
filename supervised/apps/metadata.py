import math
from copy import deepcopy

import numpy as np
import pandas as pd

from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.utils.utils import load_data

SINGLE_SAMPLE_FEATURE_LIMIT = 15


def collect_app_metadata(automl, title=None, selected_models=None):
    report = automl.report_structured(format="dict")
    feature_schema = deepcopy(automl._data_info.get("app_feature_schema"))
    if feature_schema is None:
        X = _load_training_frame(automl)
        feature_schema = _build_feature_schema(
            automl, X, automl._data_info.get("columns_values", {})
        )
    selected_models = list(selected_models or [])
    single_sample_available = len(feature_schema) <= SINGLE_SAMPLE_FEATURE_LIMIT
    notebooks = [
        {
            "filename": "predict_batch.ipynb",
            "kind": "batch_prediction",
            "title": "Batch Prediction",
        }
    ]
    default_notebook = "predict_batch.ipynb"
    if single_sample_available:
        notebooks.insert(
            0,
            {
                "filename": "predict_single.ipynb",
                "kind": "single_sample",
                "title": "Single Prediction",
            },
        )
        default_notebook = "predict_single.ipynb"
    return {
        "schema_version": 1,
        "bundle_type": "automl_prediction_bundle",
        "title": title or _default_title(automl),
        "model_task": automl._ml_task,
        "results_dir_name": "automl",
        "default_notebook": default_notebook,
        "notebooks": notebooks,
        "feature_schema": feature_schema,
        "class_labels": _get_class_labels(automl),
        "global_feature_importance": report.get(
            "global_feature_importance",
            {"available": False, "reason": "not_available"},
        ),
        "leaderboard": report.get("leaderboard", {}),
        "selected_model": report.get("selected_model"),
        "automl_bundle": {
            "archive_name": "automl.zip",
            "runtime_root": ".automl_runtime",
            "extracted_dir": "automl",
            "selected_models": selected_models,
        },
        "supports": {
            "single_sample": single_sample_available,
            "batch_prediction": True,
            "predict_proba": automl._ml_task != REGRESSION,
        },
        "python_requires": ">=3.10",
    }


def _default_title(automl):
    return "MLJAR AutoML"


def _load_training_frame(automl):
    x_path = getattr(automl, "_X_path", None)
    store = getattr(automl, "_training_store", None)
    if not x_path or store is None:
        return None
    try:
        X = load_data(x_path, store=store)
    except Exception:
        return None
    if X is None:
        return None
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(X.columns[0], str):
        X.columns = [str(c) for c in X.columns]
    return X


def _build_feature_schema(automl, X, columns_values=None, tolerate_errors=False):
    columns_info = deepcopy(automl._data_info.get("columns_info", {}))
    columns_values = columns_values or {}
    feature_schema = []
    for column in automl._data_info.get("columns", []):
        info = columns_info.get(column, [])
        series = X[column] if X is not None and column in X.columns else None
        try:
            feature = _build_feature(column, info, series, columns_values.get(column))
        except (ValueError, TypeError, OverflowError):
            if not tolerate_errors:
                raise
            # Optional UI statistics must not prevent otherwise valid training.
            feature = _build_feature(column, info, None, columns_values.get(column))
        feature_schema.append(feature)
    return feature_schema


def _build_feature(name, info, series, fallback_values=None):
    non_null = _clean_column(series) if series is not None else None
    kind = _infer_kind(info, non_null, cleaned=True)
    feature = {
        "name": name,
        "kind": kind,
        "widget": _infer_widget(kind, series),
        "required": "missing_values" not in info,
        "transformations": list(info),
        "default": None,
    }
    if series is None:
        return _fill_missing_feature_defaults(feature, fallback_values)

    if non_null.empty:
        return _fill_missing_feature_defaults(feature, fallback_values)

    if kind == "numeric":
        return _build_numeric_feature(feature, non_null)
    if kind == "boolean":
        return _build_boolean_feature(feature, non_null)
    if kind == "datetime":
        return _build_datetime_feature(feature, non_null)
    return _build_textual_feature(feature, non_null)


def _clean_column(series):
    # Wide NumPy frames often expose strided columns. Copy only this column so
    # subsequent statistics scan contiguous memory, without retaining the frame.
    values = series.to_numpy(copy=False)
    if not values.flags.c_contiguous:
        series = series.copy(deep=True)
    missing = series.isna()
    return series[~missing] if missing.any() else series


def _unique_up_to(series, limit):
    # A prefix can prove the threshold has been reached; otherwise use the
    # complete column to preserve pandas' equality and extension-dtype rules.
    prefix = pd.unique(series.iloc[:32])
    return prefix if len(prefix) >= limit or len(series) <= 32 else pd.unique(series)


def _infer_kind(info, series, cleaned=False):
    if "datetime_transform" in info:
        return "datetime"
    if "text_transform" in info or "ip_transform" in info:
        return "text"
    if series is not None:
        non_null = series if cleaned else _clean_column(series)
        unique_values = _unique_up_to(non_null, 3)
        if len(unique_values) <= 2 and _can_be_boolean(unique_values):
            return "boolean"
    return "categorical" if "categorical" in info else "numeric"


def _infer_widget(kind, series):
    if kind == "numeric":
        return "number"
    if kind == "boolean":
        return "checkbox"
    if kind == "categorical":
        return "select"
    return "text"


def _build_numeric_feature(feature, series):
    clean = _clean_column(pd.to_numeric(series, errors="coerce"))
    if clean.empty:
        return _fill_missing_feature_defaults(feature)
    numeric_dtype = _detect_numeric_dtype(clean, cleaned=True)
    min_value = float(clean.min())
    max_value = float(clean.max())
    median_value = float(clean.median())
    feature.update(
        {
            "dtype": numeric_dtype,
            "default": _coerce_json_number(median_value, clean, numeric_dtype),
            "min": _coerce_json_number(min_value, clean, numeric_dtype),
            "max": _coerce_json_number(max_value, clean, numeric_dtype),
            "step": _numeric_step(clean, numeric_dtype, cleaned=True),
            "distribution": _numeric_distribution(clean),
        }
    )
    return feature


def _build_boolean_feature(feature, series):
    normalized = [
        _normalize_bool(v) for v in series.tolist() if _normalize_bool(v) is not None
    ]
    default = bool(pd.Series(normalized).mode().iloc[0]) if normalized else False
    feature.update(
        {
            "dtype": "bool",
            "default": default,
            "choices": [False, True],
            "distribution": {
                "false": int(sum(not value for value in normalized)),
                "true": int(sum(bool(value) for value in normalized)),
            },
        }
    )
    return feature


def _build_datetime_feature(feature, series):
    dates = pd.to_datetime(series, errors="coerce").dropna()
    if dates.empty:
        return _fill_missing_feature_defaults(feature)
    feature.update(
        {
            "dtype": "datetime",
            "default": dates.iloc[0].isoformat(),
            "min": dates.min().isoformat(),
            "max": dates.max().isoformat(),
        }
    )
    return feature


def _build_textual_feature(feature, series):
    values = [str(v) for v in series.tolist()]
    counts = pd.Series(values).value_counts()
    choices = counts.index.tolist()[:20]
    feature.update(
        {
            "dtype": "str",
            "default": choices[0] if choices else str(values[0]),
            "choices": choices,
            "distribution": {
                "labels": counts.index.tolist()[:10],
                "counts": [int(v) for v in counts.tolist()[:10]],
            },
        }
    )
    return feature


def _fill_missing_feature_defaults(feature, fallback_values=None):
    if feature["kind"] == "numeric":
        feature.update(
            {
                "dtype": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
            }
        )
    elif feature["kind"] == "boolean":
        feature.update({"dtype": "bool", "default": False, "choices": [False, True]})
    else:
        choices = [str(v) for v in (fallback_values or [])]
        feature.update(
            {
                "dtype": "str",
                "default": choices[0] if choices else "",
                "choices": choices,
            }
        )
    return feature


def _numeric_distribution(series):
    counts, bins = np.histogram(
        series.astype(float), bins=min(10, max(3, len(_unique_up_to(series, 10))))
    )
    return {
        "bin_edges": [float(v) for v in bins.tolist()],
        "counts": [int(v) for v in counts.tolist()],
    }


def _numeric_step(series, numeric_dtype=None, cleaned=False):
    clean = series if cleaned else _clean_column(pd.to_numeric(series, errors="coerce"))
    if clean.empty:
        return 0.1
    if (numeric_dtype or _detect_numeric_dtype(clean)) == "int":
        return 1

    decimal_places = 0
    for value in clean:
        if not math.isfinite(value):
            continue
        text = f"{float(value):.10f}".rstrip("0").rstrip(".")
        if "." in text:
            decimal_places = max(decimal_places, len(text.split(".")[1]))
            # The result is capped at four decimal places. More values cannot
            # change it after this point; avoid formatting the entire column.
            if decimal_places >= 4:
                return 0.0001

    if decimal_places == 0:
        return 1
    decimal_places = min(decimal_places, 4)
    return float(10 ** (-decimal_places))


def _detect_numeric_dtype(series, cleaned=False):
    clean = series if cleaned else _clean_column(pd.to_numeric(series, errors="coerce"))
    if clean.empty:
        return "float"
    if pd.api.types.is_integer_dtype(clean):
        return "int"
    values = clean.astype(float).to_numpy()
    # Most continuous columns can rule out integer values using a small prefix.
    prefix = values[:32]
    if not (
        np.all(np.isfinite(prefix)) and np.allclose(prefix, np.round(prefix), atol=1e-9)
    ):
        return "float"
    if np.all(np.isfinite(values)) and np.allclose(values, np.round(values), atol=1e-9):
        return "int"
    return "float"


def _coerce_json_number(value, series, numeric_dtype=None):
    if (numeric_dtype or _detect_numeric_dtype(series)) == "int" and math.isfinite(
        value
    ):
        return int(round(value))
    return float(value)


def _can_be_boolean(values):
    normalized = {_normalize_bool(v) for v in values}
    normalized.discard(None)
    return normalized.issubset({False, True}) and len(normalized) > 0


def _normalize_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        if pd.isna(value):
            return None
        if float(value) in (0.0, 1.0):
            return bool(int(value))
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    return None


def _get_class_labels(automl):
    if automl._ml_task == REGRESSION:
        return []
    if automl._best_model is None:
        automl.load(automl._get_results_path())
    try:
        return [
            str(label)
            for label in automl._best_model.preprocessings[-1].get_target_class_names()
        ]
    except Exception:
        if automl._ml_task == BINARY_CLASSIFICATION:
            return ["0", "1"]
        if automl._ml_task == MULTICLASS_CLASSIFICATION:
            n_classes = int(automl._data_info.get("n_classes", 0))
            return [str(i) for i in range(n_classes)]
        return []
