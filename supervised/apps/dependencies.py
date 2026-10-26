"""Describe an exported prediction environment without importing optional backends."""

from importlib.metadata import version
import json
from pathlib import Path
from packaging.version import InvalidVersion, Version

from supervised.exceptions import AutoMLException
from supervised.utils.artifact_paths import resolve_artifact_path

MERCURY_REQUIREMENT = "mercury>=3.2.12,<4"
MODEL_EXTRAS = {"RealMLP": "realmlp", "TabM": "tabm", "Kumo": "tfm", "TabICLv2": "tfm"}


def selected_frameworks(results_path, selected_models):
    for name in selected_models:
        directory = Path(resolve_artifact_path(results_path, name))
        path = directory / "framework.json"
        if path.exists():
            yield name, json.loads(path.read_text(encoding="utf-8"))
        elif not (directory / "ensemble.json").exists():
            raise AutoMLException(f"Cannot export app: missing metadata for selected model {name}.")


def runtime_environment(results_path, selected_models):
    extras = {"apps"}
    algorithms = set()
    joblib_versions = set()
    for _, framework in selected_frameworks(results_path, selected_models):
        kinds = [framework.get("params", {}).get("learner", {}).get("model_type")]
        kinds += [learner.get("params", {}).get("model_type") for learner in framework.get("learners", [])]
        algorithms.update(kind for kind in kinds if kind)
        if framework.get("joblib_version"):
            saved_version = framework["joblib_version"]
            try:
                Version(saved_version)
            except (InvalidVersion, TypeError) as error:
                raise AutoMLException("Cannot export app: invalid saved joblib version.") from error
            joblib_versions.add(saved_version)
    if len(joblib_versions) > 1:
        raise AutoMLException("Cannot export app: selected models require different joblib versions.")
    extras.update(MODEL_EXTRAS[kind] for kind in algorithms if kind in MODEL_EXTRAS)
    # Pin the exporting MLJAR build and joblib (the loader requires an exact match).
    # Development checkouts must supply their matching wheel; do not fall back to
    # an older public release with the same version label.
    from supervised import __version__

    requirements = [
        f"mljar-supervised[{','.join(sorted(extras))}]=={__version__}",
        MERCURY_REQUIREMENT,
        f"joblib=={next(iter(joblib_versions), version('joblib'))}",
        f"scikit-learn=={version('scikit-learn')}",
    ]
    return dict(python_requires=">=3.11", mljar_version=__version__,
                algorithms=sorted(algorithms), extras=sorted(extras), requirements=requirements)
