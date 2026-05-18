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


def collect_app_metadata(automl, title=None, selected_models=None):
    report = automl.report_structured(format="dict")
    X = _load_training_frame(automl)
    feature_schema = _build_feature_schema(automl, X)
    selected_models = list(selected_models or [])
    return {
        "schema_version": 1,
        "bundle_type": "automl_prediction_bundle",
        "title": title or _default_title(automl),
        "model_task": automl._ml_task,
        "results_dir_name": "automl",
        "default_notebook": "predict_single.ipynb",
        "notebooks": [
            {
                "filename": "predict_single.ipynb",
                "kind": "single_sample",
                "title": "Single Prediction",
            },
            {
                "filename": "predict_batch.ipynb",
                "kind": "batch_prediction",
                "title": "Batch Prediction",
            },
        ],
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
            "single_sample": True,
            "batch_prediction": True,
            "predict_proba": automl._ml_task != REGRESSION,
        },
        "python_requires": ">=3.10",
    }


def _default_title(automl):
    task = automl._ml_task.replace("_", " ").title() if automl._ml_task else "AutoML"
    return f"MLJAR AutoML {task} App"


def _load_training_frame(automl):
    x_path = getattr(automl, "_X_path", None)
    if not x_path:
        return None
    try:
        X = load_data(x_path)
    except Exception:
        return None
    if X is None:
        return None
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(X.columns[0], str):
        X.columns = [str(c) for c in X.columns]
    return X


def _build_feature_schema(automl, X):
    columns_info = deepcopy(automl._data_info.get("columns_info", {}))
    feature_schema = []
    for column in automl._data_info.get("columns", []):
        info = columns_info.get(column, [])
        series = X[column] if X is not None and column in X.columns else None
        feature_schema.append(_build_feature(column, info, series))
    return feature_schema


def _build_feature(name, info, series):
    kind = _infer_kind(info, series)
    feature = {
        "name": name,
        "kind": kind,
        "widget": _infer_widget(kind, series),
        "required": "missing_values" not in info,
        "transformations": list(info),
        "default": None,
    }
    if series is None:
        return _fill_missing_feature_defaults(feature)

    non_null = series.dropna()
    if non_null.empty:
        return _fill_missing_feature_defaults(feature)

    if kind == "numeric":
        return _build_numeric_feature(feature, non_null)
    if kind == "boolean":
        return _build_boolean_feature(feature, non_null)
    if kind == "datetime":
        return _build_datetime_feature(feature, non_null)
    return _build_textual_feature(feature, non_null)


def _infer_kind(info, series):
    if "datetime_transform" in info:
        return "datetime"
    if "text_transform" in info:
        return "text"
    if "categorical" in info:
        if series is not None:
            non_null = series.dropna()
            unique_values = pd.unique(non_null)
            if len(unique_values) <= 2 and _can_be_boolean(unique_values):
                return "boolean"
        return "categorical"
    if series is not None:
        non_null = series.dropna()
        if not non_null.empty:
            unique_values = pd.unique(non_null)
            if len(unique_values) <= 2 and _can_be_boolean(unique_values):
                return "boolean"
            if pd.api.types.is_numeric_dtype(non_null):
                return "numeric"
    return "numeric"


def _infer_widget(kind, series):
    if kind == "numeric":
        return "number"
    if kind == "boolean":
        return "checkbox"
    if kind == "categorical":
        return "select"
    return "text"


def _build_numeric_feature(feature, series):
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return _fill_missing_feature_defaults(feature)
    min_value = float(clean.min())
    max_value = float(clean.max())
    median_value = float(clean.median())
    feature.update(
        {
            "dtype": _detect_numeric_dtype(clean),
            "default": _coerce_json_number(median_value, clean),
            "min": min_value,
            "max": max_value,
            "step": _numeric_step(min_value, max_value),
            "distribution": _numeric_distribution(clean),
        }
    )
    return feature


def _build_boolean_feature(feature, series):
    normalized = [_normalize_bool(v) for v in series.tolist() if _normalize_bool(v) is not None]
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


def _fill_missing_feature_defaults(feature):
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
        feature.update({"dtype": "str", "default": "", "choices": []})
    return feature


def _numeric_distribution(series):
    counts, bins = np.histogram(series.astype(float), bins=min(10, max(3, series.nunique())))
    return {
        "bin_edges": [float(v) for v in bins.tolist()],
        "counts": [int(v) for v in counts.tolist()],
    }


def _numeric_step(min_value, max_value):
    spread = abs(max_value - min_value)
    if spread == 0:
        return 1.0
    step = spread / 100.0
    if step >= 1:
        return float(max(1.0, round(step, 2)))
    return float(round(step, 4))


def _detect_numeric_dtype(series):
    if pd.api.types.is_integer_dtype(series):
        return "int"
    return "float"


def _coerce_json_number(value, series):
    if _detect_numeric_dtype(series) == "int" and math.isfinite(value):
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
        return [str(label) for label in automl._best_model.preprocessings[-1].get_target_class_names()]
    except Exception:
        if automl._ml_task == BINARY_CLASSIFICATION:
            return ["0", "1"]
        if automl._ml_task == MULTICLASS_CLASSIFICATION:
            n_classes = int(automl._data_info.get("n_classes", 0))
            return [str(i) for i in range(n_classes)]
        return []
