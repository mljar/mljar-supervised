import json
import os
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
from tabulate import tabulate


def _get_package_version():
    try:
        return version("mljar-supervised")
    except PackageNotFoundError:
        return "unknown"


def _to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_bool(value):
    try:
        return bool(value)
    except Exception:
        return False


def _serialize_metrics_value(value):
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="split")
    if isinstance(value, pd.Series):
        return value.to_dict()
    return value


def _to_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _extract_hyperparameters(model):
    try:
        if hasattr(model, "learner_params") and isinstance(model.learner_params, dict):
            return _to_json_safe(model.learner_params)
    except Exception:
        pass

    # Fallback for models without learner_params (e.g. Ensemble).
    try:
        model_type = model.get_type()
    except Exception:
        model_type = None
    if model_type is not None:
        return {"model_type": model_type}
    return {}


def _extract_additional_metrics(model):
    try:
        additional = model.get_additional_metrics()
        if additional is None:
            return None

        metrics = {}
        for key in [
            "max_metrics",
            "accuracy_threshold_metrics",
            "confusion_matrix",
            "threshold",
        ]:
            if key in additional:
                metrics[key] = _serialize_metrics_value(additional.get(key))
        if "fairness_metrics" in additional:
            metrics["fairness_metrics_details"] = _extract_fairness_metrics_details(
                additional.get("fairness_metrics")
            )
        return metrics
    except Exception:
        return None


def _extract_fairness_metrics_details(fairness_metrics):
    if not isinstance(fairness_metrics, dict):
        return None

    fairness_threshold = None
    optimization = fairness_metrics.get("fairness_optimization")
    if isinstance(optimization, dict):
        fairness_threshold = optimization.get("fairness_threshold")

    details = {}
    for feature_name, values in fairness_metrics.items():
        if feature_name == "fairness_optimization" or not isinstance(values, dict):
            continue
        details[feature_name] = {
            "metrics": _serialize_metrics_value(values.get("metrics")),
            "stats": _serialize_metrics_value(values.get("stats")),
            "fairness_metric_name": values.get("fairness_metric_name"),
            "fairness_metric_value": _to_float_or_none(values.get("fairness_metric_value")),
            "is_fair": _to_bool(values.get("is_fair")),
            "privileged_value": values.get("privileged_value"),
            "underprivileged_value": values.get("underprivileged_value"),
            "fairness_threshold": fairness_threshold,
        }
    return details


def _compute_feature_importance_summary(model_dir):
    imp_files = []
    try:
        imp_files = [
            f
            for f in os.listdir(model_dir)
            if f.endswith("_importance.csv") and "shap" not in f
        ]
    except Exception:
        return {"available": False, "reason": "model_dir_unreadable"}

    if not imp_files:
        return {"available": False, "reason": "no_importance_files"}

    frames = []
    for fname in imp_files:
        fpath = os.path.join(model_dir, fname)
        try:
            df = pd.read_csv(fpath, index_col=0)
            if df.empty:
                continue
            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.empty:
                continue
            frames.append(numeric_df)
        except Exception:
            continue

    if not frames:
        return {"available": False, "reason": "no_valid_importance_data"}

    concat = pd.concat(frames, axis=1, join="outer")
    mean_importance = concat.mean(axis=1).fillna(0.0)
    mean_importance = mean_importance.sort_values(ascending=False)
    n = int(mean_importance.shape[0])

    if n == 0:
        return {"available": False, "reason": "empty_importance"}

    if n < 10:
        k = 3
    elif n < 20:
        k = 5
    else:
        k = 10
    k = min(k, n)

    top_slice = mean_importance.head(k)
    worst_slice = mean_importance.tail(k)

    return {
        "available": True,
        "n_features": n,
        "selection_k": k,
        "top": [
            {"feature": str(feature), "importance": float(value)}
            for feature, value in top_slice.items()
        ],
        "worst": [
            {"feature": str(feature), "importance": float(value)}
            for feature, value in worst_slice.items()
        ],
    }


def _load_model_importance_vector(model_dir):
    imp_files = []
    try:
        imp_files = [
            f
            for f in os.listdir(model_dir)
            if f.endswith("_importance.csv") and "shap" not in f
        ]
    except Exception:
        return None

    if not imp_files:
        return None

    frames = []
    for fname in imp_files:
        fpath = os.path.join(model_dir, fname)
        try:
            df = pd.read_csv(fpath, index_col=0)
            if df.empty:
                continue
            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.empty:
                continue
            frames.append(numeric_df)
        except Exception:
            continue

    if not frames:
        return None

    concat = pd.concat(frames, axis=1, join="outer")
    mean_importance = concat.mean(axis=1).fillna(0.0)
    if mean_importance.empty:
        return None
    return mean_importance


def _compute_global_feature_importance(automl):
    rank_series = []
    for model in automl._models:
        model_dir = os.path.join(automl._results_path, model.get_name())
        importance = _load_model_importance_vector(model_dir)
        if importance is None:
            continue
        # 1 = most important feature, higher rank = less important.
        rank_series.append(importance.rank(ascending=False, method="average"))

    if not rank_series:
        return {"available": False, "reason": "no_importance_files"}

    rank_df = pd.concat(rank_series, axis=1)
    n_models_used = int(rank_df.shape[1])
    n_features = int(rank_df.shape[0])
    if n_features == 0:
        return {"available": False, "reason": "empty_rank_data"}

    models_present = rank_df.notna().sum(axis=1)
    fill_values = pd.Series([s.max() + 1.0 for s in rank_series], index=rank_df.columns)
    rank_df = rank_df.fillna(fill_values)
    mean_rank = rank_df.mean(axis=1).sort_values(ascending=True)

    if n_features < 10:
        k = 3
    elif n_features < 20:
        k = 5
    else:
        k = 10
    k = min(k, n_features)

    top_slice = mean_rank.head(k)
    bottom_slice = mean_rank.tail(k)

    return {
        "available": True,
        "method": "mean_rank_across_models",
        "n_models_used": n_models_used,
        "n_features": n_features,
        "selection_k": k,
        "top": [
            {
                "feature": str(feature),
                "mean_rank": float(rank_value),
                "models_present": int(models_present.get(feature, 0)),
            }
            for feature, rank_value in top_slice.items()
        ],
        "bottom": [
            {
                "feature": str(feature),
                "mean_rank": float(rank_value),
                "models_present": int(models_present.get(feature, 0)),
            }
            for feature, rank_value in bottom_slice.items()
        ],
    }


def _model_to_dict(automl, model):
    model_name = model.get_name()
    model_dir = os.path.join(automl._results_path, model_name)

    model_dict = {
        "name": model_name,
        "model_type": model.get_type(),
        "metric_type": automl._eval_metric,
        "metric_value": _to_float_or_none(model.get_final_loss()),
        "train_time": _to_float_or_none(model.get_train_time()),
        "is_valid": _to_bool(model.is_valid()),
        "is_stacked": _to_bool(getattr(model, "_is_stacked", False)),
        "involved_models": model.involved_model_names(),
        "artifacts": {
            "model_dir": model_dir,
            "readme_md": os.path.join(model_dir, "README.md"),
            "readme_html": os.path.join(model_dir, "README.html"),
        },
    }
    model_dict["hyperparameters"] = _extract_hyperparameters(model)
    model_dict["metrics"] = _extract_additional_metrics(model)
    model_dict["feature_importance"] = _compute_feature_importance_summary(model_dir)

    if automl._fairness_metric is not None:
        sensitive = []
        for sf in model.get_sensitive_features_names():
            sensitive.append(
                {
                    "feature": sf,
                    "fairness_metric_value": _to_float_or_none(
                        model.get_fairness_metric(sf)
                    ),
                }
            )
        model_dict["fairness"] = {
            "is_fair": _to_bool(model.is_fair()),
            "worst_fairness": _to_float_or_none(model.get_worst_fairness()),
            "best_fairness": _to_float_or_none(model.get_best_fairness()),
            "sensitive_features": sensitive,
        }

    return model_dict


def build_structured_report(automl):
    leaderboard_df = automl.get_leaderboard(original_metric_values=True)
    leaderboard = leaderboard_df.to_dict(orient="records")

    best_model = None
    if automl._best_model is not None:
        best_model = _model_to_dict(automl, automl._best_model)

    fairness_summary = None
    if automl._fairness_metric is not None and automl._best_model is not None:
        fairness_summary = {
            "fairness_metric": automl._fairness_metric,
            "fairness_threshold": automl._fairness_threshold,
            "best_model_is_fair": _to_bool(automl._best_model.is_fair()),
            "best_model_worst_fairness": _to_float_or_none(
                automl._best_model.get_worst_fairness()
            ),
            "best_model_best_fairness": _to_float_or_none(
                automl._best_model.get_best_fairness()
            ),
            "sensitive_features": [
                {
                    "feature": sf,
                    "fairness_metric_value": _to_float_or_none(
                        automl._best_model.get_fairness_metric(sf)
                    ),
                }
                for sf in automl._best_model.get_sensitive_features_names()
            ],
        }

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mljar_supervised_version": _get_package_version(),
        "results_path": automl._results_path,
        "run_summary": {
            "mode": automl._mode,
            "ml_task": automl._ml_task,
            "eval_metric": automl._eval_metric,
            "n_rows": getattr(automl, "n_rows_in_", None),
            "n_features": getattr(automl, "n_features_in_", None),
            "n_classes": getattr(automl, "n_classes", None),
            "n_models": len(automl._models),
            "best_model_name": None
            if automl._best_model is None
            else automl._best_model.get_name(),
            "fit_level": automl._fit_level,
            "total_time_limit": automl._total_time_limit,
            "model_time_limit": automl._model_time_limit,
            "validation_strategy": automl._validation_strategy,
            "train_ensemble": automl._train_ensemble,
            "stack_models": automl._stack_models,
            "fairness_metric": automl._fairness_metric,
            "fairness_threshold": automl._fairness_threshold,
        },
        "leaderboard": leaderboard,
        "global_feature_importance": _compute_global_feature_importance(automl),
        "best_model": best_model,
        "fairness_summary": fairness_summary,
        "models": [_model_to_dict(automl, m) for m in automl._models],
        "artifacts": {
            "leaderboard_csv": os.path.join(automl._results_path, "leaderboard.csv"),
            "main_readme_md": os.path.join(automl._results_path, "README.md"),
            "main_readme_html": os.path.join(automl._results_path, "README.html"),
            "report_structured_json": os.path.join(
                automl._results_path, "report_structured.json"
            ),
            "errors_md": os.path.join(automl._results_path, "errors.md"),
        },
    }
    return payload


def save_structured_report(payload, results_path):
    fname = os.path.join(results_path, "report_structured.json")
    with open(fname, "w") as fout:
        json.dump(payload, fout, indent=4)
    return fname


def _append_split_table(lines, title, table_obj, heading_level="###"):
    if table_obj is None:
        return
    columns = table_obj.get("columns", [])
    data = table_obj.get("data", [])
    index = table_obj.get("index")
    show_index = False
    if index is not None and len(index) == len(data):
        # Hide synthetic/default row index (0..n-1) to keep tables readable.
        try:
            show_index = list(index) != list(range(len(data)))
        except Exception:
            show_index = True
    if show_index:
        columns = ["index"] + list(columns)
        data = [[index[i]] + row for i, row in enumerate(data)]
    lines.append(f"{heading_level} {title}")
    lines.append("")
    lines.append(tabulate(data, headers=columns, tablefmt="pipe"))
    lines.append("")


def _append_model_metrics(lines, model):
    metrics = model.get("metrics") or {}
    _append_split_table(
        lines, "Metric details", metrics.get("max_metrics"), heading_level="##"
    )
    _append_split_table(
        lines, "Confusion matrix", metrics.get("confusion_matrix"), heading_level="##"
    )
    threshold = metrics.get("threshold")
    if threshold is not None:
        lines.append(f"Threshold: `{threshold}`")
        lines.append("")


def _append_hyperparameters(lines, model):
    hyperparameters = dict(model.get("hyperparameters") or {})
    hyperparameters.pop("model_type", None)
    model_type = str(model.get("model_type") or "")
    if model_type in {"Baseline", "Decision Tree", "Neural Network"}:
        hyperparameters.pop("n_jobs", None)
    lines.append("## Hyperparameters")
    lines.append("")
    if not hyperparameters:
        lines.append("_No hyperparameters available._")
        lines.append("")
        return

    rows = [[str(k), hyperparameters.get(k)] for k in sorted(hyperparameters.keys())]
    lines.append(tabulate(rows, headers=["Parameter", "Value"], tablefmt="pipe"))
    lines.append("")


def _append_feature_importance(lines, model, heading_level="###"):
    fi = model.get("feature_importance") or {}
    if fi.get("available"):
        k = fi.get("selection_k")
        top_rows = [[row.get("feature"), row.get("importance")] for row in fi.get("top", [])]
        worst_rows = [
            [row.get("feature"), row.get("importance")] for row in fi.get("worst", [])
        ]
        lines.append(
            f"{heading_level} Most Influential Features (Top {k} by permutation importance)"
        )
        lines.append("")
        lines.append(tabulate(top_rows, headers=["Feature", "Importance"], tablefmt="pipe"))
        lines.append("")
        lines.append(
            f"{heading_level} Least Influential Features (Bottom {k} by permutation importance)"
        )
        lines.append("")
        lines.append(
            tabulate(worst_rows, headers=["Feature", "Importance"], tablefmt="pipe")
        )
        lines.append("")


def _append_fairness(lines, model, title):
    fairness = model.get("fairness")
    if fairness is None:
        return
    lines.append(title)
    lines.append("")
    fairness_rows = [
        ["is_fair", fairness.get("is_fair")],
        ["worst_fairness", fairness.get("worst_fairness")],
        ["best_fairness", fairness.get("best_fairness")],
    ]
    lines.append(tabulate(fairness_rows, headers=["Field", "Value"], tablefmt="pipe"))
    sf = fairness.get("sensitive_features", [])
    if sf:
        lines.append("")
        lines.append(
            tabulate(
                [[s.get("feature"), s.get("fairness_metric_value")] for s in sf],
                headers=["Sensitive Feature", "Fairness Metric Value"],
                tablefmt="pipe",
            )
        )
    lines.append("")


def _append_fairness_metrics_details(lines, model, heading_level="###"):
    metrics = model.get("metrics") or {}
    fairness_details = metrics.get("fairness_metrics_details") or {}
    if not fairness_details:
        return

    for feature_name, values in fairness_details.items():
        lines.append(f"{heading_level} Fairness metrics for {feature_name} feature")
        lines.append("")

        _append_split_table(lines, "Fairness group metrics", values.get("metrics"))
        _append_split_table(lines, "Fairness summary statistics", values.get("stats"))

        fairness_metric_name = values.get("fairness_metric_name")
        fairness_metric_value = values.get("fairness_metric_value")
        is_fair = values.get("is_fair")
        fairness_threshold = values.get("fairness_threshold")
        privileged_value = values.get("privileged_value")
        underprivileged_value = values.get("underprivileged_value")

        fair_str = "fair" if is_fair else "unfair"
        lines.append(f"{heading_level} Is model fair for {feature_name} feature?")
        lines.append("")

        threshold_str = ""
        if fairness_threshold is not None and fairness_metric_name is not None:
            if "ratio" in str(fairness_metric_name).lower():
                threshold_str = f"It should be higher than {fairness_threshold}."
            else:
                threshold_str = f"It should be lower than {fairness_threshold}."

        lines.append(f"Model is {fair_str} for {feature_name} feature.")
        if fairness_metric_name is not None and fairness_metric_value is not None:
            lines.append(
                f"The {fairness_metric_name} is {fairness_metric_value}. {threshold_str}".strip()
            )
        if is_fair is False:
            if underprivileged_value is not None:
                lines.append(f"Underprivileged value is {underprivileged_value}.")
            if privileged_value is not None:
                lines.append(f"Privileged value is {privileged_value}.")
        lines.append("")


def _find_model(models, model_name):
    for model in models:
        if model.get("name") == model_name:
            return model
    return None


def build_compact_view(payload, model_name=None):
    compact = {
        "created_at_utc": payload.get("created_at_utc"),
        "mljar_supervised_version": payload.get("mljar_supervised_version"),
        "results_path": payload.get("results_path"),
        "leaderboard": payload.get("leaderboard", []),
        "global_feature_importance": payload.get("global_feature_importance"),
    }
    if model_name is not None:
        selected = _find_model(payload.get("models", []), model_name)
        if selected is None:
            available_models = sorted(
                [m.get("name") for m in payload.get("models", []) if m.get("name")]
            )
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {available_models}"
            )
        compact["selected_model"] = selected
    return compact


def to_markdown(payload, model_name=None):
    lines = []
    selected_model = payload.get("selected_model")
    if selected_model is None:
        lines.append("# MLJAR AutoML Report")
    else:
        lines.append(f"# MLJAR AutoML report for {selected_model.get('name')}")
    lines.append("")

    if selected_model is None:
        lines.append("## Leaderboard")
        lines.append("")
        leaderboard = payload.get("leaderboard", [])
        if leaderboard:
            df = pd.DataFrame(leaderboard)
            base_cols = [
                c
                for c in ["name", "model_type", "metric_type", "metric_value", "train_time"]
                if c in df.columns
            ]
            fairness_cols = [
                c
                for c in df.columns
                if (c == "fairness_metric" or c.startswith("fairness_") or c == "is_fair")
                and c not in base_cols
            ]
            preferred_cols = base_cols + fairness_cols
            if preferred_cols:
                df = df[preferred_cols]
            lines.append(tabulate(df.values, headers=list(df.columns), tablefmt="pipe"))
        else:
            lines.append("_No models found._")
        lines.append("")

        global_fi = payload.get("global_feature_importance") or {}
        if global_fi.get("available"):
            k = global_fi.get("selection_k")
            lines.append("## Global Feature Importance (Averaged Across Models)")
            lines.append("")
            lines.append(f"Method: `{global_fi.get('method')}`")
            lines.append("")
            lines.append(
                f"Models used: `{global_fi.get('n_models_used')}`, Features: `{global_fi.get('n_features')}`"
            )
            lines.append("")

            top_rows = [
                [row.get("feature"), row.get("mean_rank")]
                for row in global_fi.get("top", [])
            ]
            bottom_rows = [
                [row.get("feature"), row.get("mean_rank")]
                for row in global_fi.get("bottom", [])
            ]

            lines.append(f"### Most Influential Features (Top {k} by mean rank)")
            lines.append("")
            lines.append(
                tabulate(
                    top_rows,
                    headers=["Feature", "Mean Rank"],
                    tablefmt="pipe",
                )
            )
            lines.append("")

            lines.append(f"### Least Influential Features (Bottom {k} by mean rank)")
            lines.append("")
            lines.append(
                tabulate(
                    bottom_rows,
                    headers=["Feature", "Mean Rank"],
                    tablefmt="pipe",
                )
            )
            lines.append("")
    else:
        summary_rows = [
            ["model_type", selected_model.get("model_type")],
            ["metric_type", selected_model.get("metric_type")],
            ["metric_value", selected_model.get("metric_value")],
            ["train_time", selected_model.get("train_time")],
            ["is_valid", selected_model.get("is_valid")],
            ["is_stacked", selected_model.get("is_stacked")],
        ]
        lines.append(tabulate(summary_rows, headers=["Field", "Value"], tablefmt="pipe"))
        lines.append("")
        _append_hyperparameters(lines, selected_model)
        _append_model_metrics(lines, selected_model)
        _append_feature_importance(lines, selected_model, heading_level="##")
        _append_fairness_metrics_details(lines, selected_model, heading_level="##")

    return "\n".join(lines)
