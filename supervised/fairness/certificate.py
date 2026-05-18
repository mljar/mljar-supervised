from datetime import date
from urllib.parse import urlencode

from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)


FAIRNESS_CERTIFICATE_URL = "https://mljar.com/fairness-certificate/"


def _task_type_label(ml_task):
    labels = {
        BINARY_CLASSIFICATION: "Binary Classification",
        MULTICLASS_CLASSIFICATION: "Multiclass Classification",
        REGRESSION: "Regression",
    }
    return labels.get(ml_task, str(ml_task))


def _format_value(value):
    if value is None:
        return ""
    try:
        return f"{float(value):g}"
    except Exception:
        return str(value)


def _is_ratio_metric(metric_name):
    return "ratio" in str(metric_name).lower()


def _format_sensitive_feature(feature_name):
    if "__" not in str(feature_name):
        return str(feature_name)
    feature, class_name = str(feature_name).split("__", maxsplit=1)
    return f"{feature} (class: {class_name})"


def fairness_metrics_to_details(fairness_metrics):
    if not isinstance(fairness_metrics, dict):
        return {}

    fairness_threshold = None
    optimization = fairness_metrics.get("fairness_optimization")
    if isinstance(optimization, dict):
        fairness_threshold = optimization.get("fairness_threshold")

    details = {}
    for feature_name, values in fairness_metrics.items():
        if feature_name == "fairness_optimization" or not isinstance(values, dict):
            continue
        details[feature_name] = {
            "fairness_metric_name": values.get("fairness_metric_name"),
            "fairness_metric_value": values.get("fairness_metric_value"),
            "fairness_threshold": values.get(
                "fairness_threshold", fairness_threshold
            ),
            "is_fair": values.get("is_fair"),
        }
    return details


def _select_decisive_feature_name(fairness_details, worst_fairness=None):
    if not fairness_details:
        return None

    if worst_fairness is not None:
        for feature_name, values in fairness_details.items():
            try:
                if abs(float(values.get("fairness_metric_value")) - float(worst_fairness)) < 1e-12:
                    return feature_name
            except Exception:
                continue

    feature_items = list(fairness_details.items())
    metric_name = feature_items[0][1].get("fairness_metric_name")
    if _is_ratio_metric(metric_name):
        return min(
            feature_items,
            key=lambda item: float(item[1].get("fairness_metric_value", float("inf"))),
        )[0]
    return max(
        feature_items,
        key=lambda item: float(item[1].get("fairness_metric_value", float("-inf"))),
    )[0]


def build_certificate_info(
    model_name,
    ml_task,
    fairness_details,
    worst_fairness=None,
    is_fair=None,
    issue_date=None,
):
    if not fairness_details:
        return None
    if is_fair is False:
        return None

    feature_name = _select_decisive_feature_name(fairness_details, worst_fairness)
    if feature_name is None:
        return None

    values = fairness_details.get(feature_name) or {}
    fairness_metric_name = values.get("fairness_metric_name")
    fairness_metric_value = values.get("fairness_metric_value")
    fairness_threshold = values.get("fairness_threshold")
    if fairness_metric_name is None or fairness_metric_value is None:
        return None

    params = {
        "modelName": str(model_name),
        "taskType": _task_type_label(ml_task),
        "fairnessMetric": str(fairness_metric_name),
        "achievedScore": _format_value(fairness_metric_value),
        "requiredThreshold": _format_value(fairness_threshold),
        "status": "PASSED",
        "sensitiveFeature": _format_sensitive_feature(feature_name),
        "issueDate": issue_date or date.today().isoformat(),
    }
    return {
        "certificate_url": f"{FAIRNESS_CERTIFICATE_URL}?{urlencode(params)}",
        "certificate_params": params,
    }
