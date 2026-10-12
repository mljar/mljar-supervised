"""Validate algorithm-specific fixed overrides for the public AutoML API."""

import copy

from supervised.algorithms.registry import AlgorithmsRegistry


def algorithm_key(name):
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


def validate_custom_params(custom_params, ml_task):
    if custom_params is None:
        return {}
    if not isinstance(custom_params, dict):
        raise ValueError("custom_params must be a dictionary keyed by algorithm name")
    registry = AlgorithmsRegistry.registry[ml_task]
    names = {algorithm_key(name): name for name in registry if name}
    result = {}
    protected = {
        "model_type",
        "ml_task",
        "seed",
        "n_jobs",
        "num_class",
        "objective",
        "eval_metric",
        "eval_metric_name",
        "metric",
        "custom_eval_metric_name",
    }
    for name, values in custom_params.items():
        if not isinstance(name, str) or algorithm_key(name) not in names:
            raise ValueError(f"Unknown algorithm in custom_params: {name!r}")
        key = algorithm_key(name)
        if key in result:
            raise ValueError(f"Duplicate algorithm in custom_params: {name!r}")
        if not isinstance(values, dict):
            raise ValueError(f"custom_params[{name!r}] must be a dictionary")
        if key == "realmlp":
            from supervised.algorithms.realmlp import validate_options

            validate_options(values)
        else:
            info = registry[names[key]]
            allowed = (set(info["params"]) | set(info["default_params"])) - protected
            unknown = set(values) - allowed
            if unknown:
                raise ValueError(
                    f"Unsupported custom_params for {name}: {sorted(unknown)}. "
                    f"Supported parameters: {sorted(allowed)}"
                )
        result[key] = copy.deepcopy(values)
    return result
