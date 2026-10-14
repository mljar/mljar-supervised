"""Fold-local Goldmine discovery and portable replay of frozen features."""

import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from packaging.version import Version

from supervised.exceptions import AutoMLException
from supervised.preprocessing.exclude_missing_target import ExcludeRowsMissingTarget
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.validation.validation_step import ValidationStep


def check_dependencies():
    message = (
        "Feature discovery requires Python 3.10+ and features-goldmine>=1.1,<1.2. "
        'Install with: pip install "mljar-supervised[feature-discovery]"'
    )
    if sys.version_info < (3, 10):
        raise AutoMLException(message)
    try:
        version = Version(importlib.metadata.version("features-goldmine"))
        if not Version("1.1") <= version < Version("1.2"):
            raise AutoMLException(message)
        from features_goldmine import GoldenFeatures
    except ImportError as exc:
        raise AutoMLException(message) from exc
    return GoldenFeatures


def supported_validation(params):
    return params.get("validation_type", "kfold") in ("split", "kfold") and params.get(
        "shuffle", True
    )


def discovery_input(X):
    X = PreprocessingUtils.convert_timedeltas(
        X.copy(deep=False), PreprocessingUtils.timedelta_columns(X)
    )
    return X.select_dtypes(include=["number", "bool", "object", "string", "category"])


def fingerprint(X, y):
    # Includes values, dtypes, column order and row index, not just row count.
    return joblib.hash((X, y))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


class FeatureDiscovery:
    """Prepare each fold once; persist training encodings separately from inference."""

    def __init__(self, results_path, ml_task, seed, n_jobs, data_store=None):
        self.root = Path(results_path)
        self.task = {
            "binary_classification": "binary",
            "multiclass_classification": "multiclass",
            "regression": "regression",
        }[ml_task]
        self.seed = seed
        self.n_jobs = joblib.effective_n_jobs(n_jobs)
        self.data_store = data_store

    def prepare(self, validation_params, time_limit):
        GoldenFeatures = check_dependencies()
        package_version = importlib.metadata.version("features-goldmine")
        if not supported_validation(validation_params):
            raise AutoMLException(
                "Feature discovery needs shuffled split or K-fold validation."
            )
        validation = ValidationStep(validation_params, data_store=self.data_store)
        count = validation.get_repeats() * validation.get_n_splits()
        deadline = time.monotonic() + time_limit
        folds, profiles = {}, None
        for repeat in range(validation.get_repeats()):
            for fold in range(validation.get_n_splits()):
                train, _ = validation.get_split(fold, repeat)
                X, y, weights, _ = ExcludeRowsMissingTarget.transform(
                    train["X"], train["y"], train.get("sample_weight")
                )
                X = discovery_input(X)
                if not len(X.columns):
                    return None
                digest = fingerprint(X, y)
                identity = joblib.hash(
                    (digest, weights, self.task, self.seed, package_version)
                )
                relative = Path("feature_discovery") / identity
                directory = self.root / relative
                manifest = directory / "manifest.json"
                if manifest.exists() and (directory / "training.joblib").exists():
                    info = json.loads(manifest.read_text())
                    if not (directory / "generator.joblib").exists():
                        raise AutoMLException(
                            "Missing feature discovery artifact: " + str(directory)
                        )
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    generator = GoldenFeatures(
                        time_limit=remaining / (count - len(folds)),
                        max_feature_sets=3,
                        task=self.task,
                        random_state=self.seed,
                        n_jobs=self.n_jobs,
                    )
                    started = time.monotonic()
                    outputs = generator.fit_transform(X, y, feature_set="all")
                    names = list(
                        dict.fromkeys(c for frame in outputs.values() for c in frame)
                    )
                    # A single union matrix avoids storing shared profile columns repeatedly.
                    columns = {}
                    for frame in outputs.values():
                        for name in frame:
                            if name not in columns:
                                columns[name] = frame[name]
                    matrix = pd.DataFrame(columns, index=X.index)
                    rename = {}
                    used = set(X.columns)
                    for i, name in enumerate(names):
                        output = f"feature_discovery_{i}"
                        while output in used:
                            output += "_"
                        rename[name] = output
                        used.add(output)
                    info = {
                        "features_goldmine_version": package_version,
                        "fingerprint": digest,
                        "input_columns": list(X.columns),
                        "sets": {
                            name: list(frame.columns)
                            for name, frame in outputs.items()
                            if len(frame.columns)
                        },
                        "rename": rename,
                        "discovery_seconds": time.monotonic() - started,
                        "discovery_uses_sample_weight": False,
                    }
                    directory.mkdir(parents=True, exist_ok=True)
                    generator.save(directory / "generator.joblib", report=True)
                    joblib.dump(matrix, directory / "training.tmp")
                    os.replace(
                        directory / "training.tmp", directory / "training.joblib"
                    )
                    write_json(manifest, info)
                    del generator, outputs, matrix, columns
                available = list(info["sets"])
                profiles = (
                    available
                    if profiles is None
                    else [p for p in profiles if p in available]
                )
                folds[f"{repeat}:{fold}"] = {**info, "directory": relative.as_posix()}
                del X, y, train
        if not profiles:
            return None
        # Profiles must differ in at least one fold to justify a new model.
        unique, seen = [], set()
        for profile in profiles:
            signature = tuple(tuple(sorted(f["sets"][profile])) for f in folds.values())
            if signature not in seen:
                unique.append(profile)
                seen.add(signature)
        return {"folds": folds, "profiles": unique[:3]}


class FeatureDiscoveryTransformer:
    def __init__(self, info, feature_set, results_path):
        self.info = info
        self.feature_set = feature_set
        self.root = Path(results_path)
        self.generator = None
        self._new_columns = [info["rename"][c] for c in info["sets"][feature_set]]

    def _input(self, X):
        missing = set(self.info["input_columns"]) - set(X.columns)
        if missing:
            raise AutoMLException(
                f"Missing feature discovery inputs: {sorted(missing)}"
            )
        return discovery_input(X[self.info["input_columns"]])

    def fit_transform(self, X, y):
        X = self._input(X)
        if fingerprint(X, y) != self.info["fingerprint"]:
            raise AutoMLException(
                "Feature discovery training rows or targets changed; refusing to reuse encodings."
            )
        path = self.root / self.info["directory"] / "training.joblib"
        if not path.exists():
            raise AutoMLException(
                "Feature discovery training cache is missing; restart with a new results directory."
            )
        frame = joblib.load(path)
        if not frame.index.equals(X.index):
            raise AutoMLException("Feature discovery training row order changed.")
        return frame[self.info["sets"][self.feature_set]].rename(
            columns=self.info["rename"]
        )

    def transform(self, X):
        if self.generator is None:
            GoldenFeatures = check_dependencies()
            self.generator = GoldenFeatures.load(
                self.root / self.info["directory"] / "generator.joblib"
            )
        return self.generator.transform(
            self._input(X), feature_set=self.feature_set
        ).rename(columns=self.info["rename"])

    def to_json(self):
        return {"info": self.info, "feature_set": self.feature_set}

    @classmethod
    def from_json(cls, data, results_path):
        return cls(data["info"], data["feature_set"], results_path)

    def markdown(self):
        directory = self.info["directory"]
        lines = [
            f"Profile: **{self.feature_set}**, {len(self._new_columns)} added features; "
            f"discovery {self.info['discovery_seconds']:.2f} seconds. "
            f"[Transformations](../{directory}/generator.md), "
            f"[JSON](../{directory}/generator.json).\n",
            "Discovery is unweighted; learner training and evaluation retain their sample weights.\n",
            "| Model column | Discovered feature |\n|---|---|",
        ]
        for source in self.info["sets"][self.feature_set]:
            escaped = str(source).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {self.info['rename'][source]} | {escaped} |")
        return "\n".join(lines) + "\n"
