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
        return metrics
    except Exception:
        return None


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


def _append_split_table(lines, title, table_obj):
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
    lines.append(f"### {title}")
    lines.append("")
    lines.append(tabulate(data, headers=columns, tablefmt="pipe"))
    lines.append("")


def to_markdown(payload, model_details=True):
    lines = []
    lines.append("# MLJAR AutoML Report")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    run_summary = payload.get("run_summary", {})
    for key, value in run_summary.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")

    lines.append("## Leaderboard")
    lines.append("")
    leaderboard = payload.get("leaderboard", [])
    if leaderboard:
        df = pd.DataFrame(leaderboard)
        preferred_cols = [
            c
            for c in ["name", "model_type", "metric_type", "metric_value", "train_time"]
            if c in df.columns
        ]
        if preferred_cols:
            df = df[preferred_cols]
        lines.append(tabulate(df.values, headers=list(df.columns), tablefmt="pipe"))
    else:
        lines.append("_No models found._")
    lines.append("")

    lines.append("## Best Model")
    lines.append("")
    best_model = payload.get("best_model")
    if best_model is None:
        lines.append("_No best model available._")
    else:
        scalar_keys = [
            "name",
            "model_type",
            "metric_type",
            "metric_value",
            "train_time",
            "is_valid",
            "is_stacked",
        ]
        for key in scalar_keys:
            if key in best_model:
                lines.append(f"- **{key}**: {best_model.get(key)}")
        if best_model.get("involved_models"):
            lines.append("- **involved_models**:")
            for model_name in best_model.get("involved_models", []):
                lines.append(f"  - {model_name}")
        lines.append("")

        best_metrics = best_model.get("metrics") or {}
        _append_split_table(lines, "Metric details", best_metrics.get("max_metrics"))
        _append_split_table(
            lines,
            "Metric details with threshold from accuracy metric",
            best_metrics.get("accuracy_threshold_metrics"),
        )
        _append_split_table(
            lines, "Confusion matrix", best_metrics.get("confusion_matrix")
        )
        threshold = best_metrics.get("threshold")
        if threshold is not None:
            lines.append(f"Threshold: `{threshold}`")
            lines.append("")

        best_fi = best_model.get("feature_importance") or {}
        if best_fi.get("available"):
            k = best_fi.get("selection_k")
            top_rows = [
                [row.get("feature"), row.get("importance")]
                for row in best_fi.get("top", [])
            ]
            worst_rows = [
                [row.get("feature"), row.get("importance")]
                for row in best_fi.get("worst", [])
            ]
            lines.append(
                f"### Most Influential Features (Top {k} by permutation importance)"
            )
            lines.append("")
            lines.append(
                tabulate(top_rows, headers=["Feature", "Importance"], tablefmt="pipe")
            )
            lines.append("")
            lines.append(
                f"### Least Influential Features (Bottom {k} by permutation importance)"
            )
            lines.append("")
            lines.append(
                tabulate(
                    worst_rows, headers=["Feature", "Importance"], tablefmt="pipe"
                )
            )
            lines.append("")

        best_fairness = best_model.get("fairness")
        if best_fairness is not None:
            lines.append("### Fairness (Best Model)")
            lines.append("")
            fairness_rows = [
                ["is_fair", best_fairness.get("is_fair")],
                ["worst_fairness", best_fairness.get("worst_fairness")],
                ["best_fairness", best_fairness.get("best_fairness")],
            ]
            lines.append(
                tabulate(fairness_rows, headers=["Field", "Value"], tablefmt="pipe")
            )
            sf = best_fairness.get("sensitive_features", [])
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
    lines.append("")

    fairness = payload.get("fairness_summary")
    if fairness is not None:
        lines.append("## Fairness Summary")
        lines.append("")
        rows = []
        for key in [
            "fairness_metric",
            "fairness_threshold",
            "best_model_is_fair",
            "best_model_worst_fairness",
            "best_model_best_fairness",
        ]:
            rows.append([key, fairness.get(key)])
        lines.append(tabulate(rows, headers=["Field", "Value"], tablefmt="pipe"))
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

    if model_details:
        lines.append("## Model Details")
        lines.append("")
        models = payload.get("models", [])
        if not models:
            lines.append("_No model details available._")
            lines.append("")
        for model in models:
            lines.append(f"## {model.get('name')}")
            lines.append("")
            summary_rows = [
                ["model_type", model.get("model_type")],
                ["metric_type", model.get("metric_type")],
                ["metric_value", model.get("metric_value")],
                ["train_time", model.get("train_time")],
                ["is_valid", model.get("is_valid")],
                ["is_stacked", model.get("is_stacked")],
            ]
            lines.append(
                tabulate(summary_rows, headers=["Field", "Value"], tablefmt="pipe")
            )
            lines.append("")

            metrics = model.get("metrics") or {}
            _append_split_table(lines, "Metric details", metrics.get("max_metrics"))
            _append_split_table(
                lines,
                "Metric details with threshold from accuracy metric",
                metrics.get("accuracy_threshold_metrics"),
            )
            _append_split_table(
                lines, "Confusion matrix", metrics.get("confusion_matrix")
            )
            threshold = metrics.get("threshold")
            if threshold is not None:
                lines.append(f"Threshold: `{threshold}`")
                lines.append("")

            fi = model.get("feature_importance") or {}
            if fi.get("available"):
                k = fi.get("selection_k")
                top_rows = [
                    [row.get("feature"), row.get("importance")]
                    for row in fi.get("top", [])
                ]
                worst_rows = [
                    [row.get("feature"), row.get("importance")]
                    for row in fi.get("worst", [])
                ]
                lines.append(
                    f"### Most Influential Features (Top {k} by permutation importance)"
                )
                lines.append("")
                lines.append(
                    tabulate(
                        top_rows,
                        headers=["Feature", "Importance"],
                        tablefmt="pipe",
                    )
                )
                lines.append("")

                lines.append(
                    f"### Least Influential Features (Bottom {k} by permutation importance)"
                )
                lines.append("")
                lines.append(
                    tabulate(
                        worst_rows,
                        headers=["Feature", "Importance"],
                        tablefmt="pipe",
                    )
                )
                lines.append("")

            fairness = model.get("fairness")
            if fairness is not None:
                lines.append("### Fairness")
                lines.append("")
                fairness_rows = [
                    ["is_fair", fairness.get("is_fair")],
                    ["worst_fairness", fairness.get("worst_fairness")],
                    ["best_fairness", fairness.get("best_fairness")],
                ]
                lines.append(
                    tabulate(
                        fairness_rows, headers=["Field", "Value"], tablefmt="pipe"
                    )
                )
                sf = fairness.get("sensitive_features", [])
                if sf:
                    lines.append("")
                    lines.append(
                        tabulate(
                            [
                                [s.get("feature"), s.get("fairness_metric_value")]
                                for s in sf
                            ],
                            headers=["Sensitive Feature", "Fairness Metric Value"],
                            tablefmt="pipe",
                        )
                    )
                lines.append("")

    return "\n".join(lines)
