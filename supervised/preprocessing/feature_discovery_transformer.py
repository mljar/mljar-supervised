"""Fold-local Goldmine discovery and portable replay of frozen features."""

import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from packaging.version import Version

from supervised.exceptions import AutoMLException
from supervised.preprocessing.exclude_missing_target import ExcludeRowsMissingTarget
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.validation.validation_step import ValidationStep
from supervised.preprocessing.feature_identity import (
    SCHEMA_VERSION, build_identities, digest as identity_digest, file_digest, input_identity, validate_manifest,
)


def check_dependencies():
    message = (
        "Feature discovery requires Python 3.11+ and features-goldmine>=1.1,<1.2. "
        "It is included in the standard MLJAR installation. "
        'Repair this environment with: python -m pip install "features-goldmine>=1.1,<1.2"'
    )
    if sys.version_info < (3, 11):
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


def _values_signature(values):
    dtype = values.dtype
    schema = str(dtype)
    if isinstance(dtype, pd.CategoricalDtype):
        schema = (schema, dtype.ordered, _values_signature(dtype.categories))
    # Hash owned, contiguous arrays instead of pandas' block managers and caches.
    # Selecting identical columns can otherwise change a DataFrame's hash.
    return schema, joblib.hash(np.array(values, copy=True, order="C"))


def fingerprint(X, y, version=2):
    if version == 1:
        return joblib.hash((X, y))
    if version != 2:
        raise AutoMLException(
            f"Unsupported feature discovery fingerprint version: {version}"
        )
    return joblib.hash(
        (
            version,
            _values_signature(X.columns),
            _values_signature(X.index),
            [_values_signature(X.iloc[:, i]) for i in range(X.shape[1])],
            _values_signature(y.index),
            _values_signature(y),
        )
    )


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

    def prepare(self, validation_params, time_limit, progress=None, *, full_data=False):
        GoldenFeatures = check_dependencies()
        package_version = importlib.metadata.version("features-goldmine")
        if not supported_validation(validation_params):
            raise AutoMLException(
                "Feature discovery needs shuffled split or K-fold validation."
            )
        validation = None if full_data else ValidationStep(validation_params, data_store=self.data_store)
        repeats = 1 if full_data else validation.get_repeats()
        n_splits = 1 if full_data else validation.get_n_splits()
        count = repeats * n_splits
        deadline = time.monotonic() + time_limit
        folds, profiles = {}, None

        def emit(message):
            if progress is not None:
                progress("Feature discovery: " + message)

        scope = "all training data" if full_data else f"{n_splits} folds"
        if repeats > 1:
            scope += f" × {repeats} repeats"
        emit(f"searching {scope} (budget {time_limit:.1f}s).")
        for repeat in range(repeats):
            for fold in range(n_splits):
                label = "all training data" if full_data else f"fold {fold + 1}/{n_splits}"
                if repeats > 1:
                    label += f", repeat {repeat + 1}/{repeats}"
                if full_data:
                    from supervised.utils.utils import load_data
                    train = {
                        "X": load_data(validation_params["X_path"], store=self.data_store),
                        "y": load_data(validation_params["y_path"], store=self.data_store)["target"],
                    }
                else:
                    train, _ = validation.get_split(fold, repeat)
                X, y, weights, _ = ExcludeRowsMissingTarget.transform(
                    train["X"], train["y"], train.get("sample_weight")
                )
                X = discovery_input(X)
                if not len(X.columns):
                    emit("skipped: no supported input columns.")
                    return None
                digest = fingerprint(X, y)
                identity = joblib.hash(
                    (digest, weights, self.task, self.seed, package_version, SCHEMA_VERSION)
                )
                relative = Path("feature_discovery") / identity
                directory = self.root / relative
                manifest = directory / "manifest.json"
                cached = manifest.exists() and (directory / "training.joblib").exists()
                if cached:
                    info = json.loads(manifest.read_text())
                    if not (directory / "generator.joblib").exists():
                        raise AutoMLException(
                            "Missing feature discovery artifact: " + str(directory)
                        )
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        emit("stopped: discovery budget exhausted before all folds completed.")
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
                    directory.mkdir(parents=True, exist_ok=True)
                    generator.save(directory / "generator.joblib", report=True)
                    generator_sha256 = file_digest(directory / "generator.joblib")
                    generator_id = identity_digest({"fit_scope": identity, "state": generator_sha256})
                    identities = build_identities(
                        names, getattr(generator, "_selected_metadata_by_name", {}),
                        generator_id, list(X.columns), package_version,
                    )
                    rename = {name: record["storage_column"] for name, record in identities.items()}
                    info = {
                        "identity_schema_version": SCHEMA_VERSION,
                        "generator_id": generator_id,
                        "generator_sha256": generator_sha256,
                        "fit_scope": identity,
                        "feature_identities": identities,
                        "features_goldmine_version": package_version,
                        "fingerprint": digest,
                        "fingerprint_version": 2,
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
                    joblib.dump(matrix, directory / "training.tmp")
                    os.replace(
                        directory / "training.tmp", directory / "training.joblib"
                    )
                    write_json(manifest, info)
                    del generator, outputs, matrix, columns
                # Preview selected transformations, not hashed storage names. Keep
                # each fitted fold separate: matching labels do not imply the
                # same learned transformation. Full details remain in generator.md.
                selected = list(dict.fromkeys(
                    name for columns in info["sets"].values() for name in columns
                ))
                if selected:
                    records = info.get("feature_identities", {})
                    labels = [
                        " ".join(str(records.get(name, {}).get("display_label", name)).split())
                        for name in selected[:3]
                    ]
                    preview = "; ".join(
                        name if len(name) <= 140 else name[:137] + "..."
                        for name in labels
                    )
                    suffix = f" (+{len(selected) - 3} more)" if len(selected) > 3 else ""
                    noun = "transformation" if len(selected) == 1 else "transformations"
                    summary = f"selected {len(selected)} {noun}: {preview}{suffix}"
                else:
                    summary = "no transformations selected"
                timing = "cached" if cached else f"{info['discovery_seconds']:.2f}s"
                emit(f"{label}: {summary} ({timing}).")
                available = list(info["sets"])
                profiles = (
                    available
                    if profiles is None
                    else [p for p in profiles if p in available]
                )
                folds[f"{repeat}:{fold}"] = {**info, "directory": relative.as_posix()}
                del X, y, train
        if not profiles:
            emit("no feature set is available in every training fold; no models scheduled.")
            return None
        # Profiles must differ in at least one fold to justify a new model.
        unique, seen = [], set()
        for profile in profiles:
            signature = tuple(tuple(sorted(f["sets"][profile])) for f in folds.values())
            if signature not in seen:
                unique.append(profile)
                seen.add(signature)
        emit("feature sets ready for model training: " + ", ".join(unique[:3]) + ".")
        return {"folds": folds, "profiles": unique[:3]}


class FeatureDiscoveryTransformer:
    def __init__(self, info, feature_set, results_path):
        self.info = info
        self.feature_set = feature_set
        self.root = Path(results_path)
        self.generator = None
        self._identity_checked = False
        validate_manifest(info)
        self._new_columns = [info["rename"][c] for c in info["sets"][feature_set]]

    def validate_selection(self, selection):
        if selection.get("generator_id") != self.info.get("generator_id") or not self.info.get("generator_id"):
            raise AutoMLException("Feature selection generator mismatch")
        allowed = {input_identity(c) for c in self.info["input_columns"]}
        if not set(selection.get("dropped_input_identities", [])).issubset(allowed):
            raise AutoMLException("Feature selection references unknown input identities")
        if selection.get("dropped_generated_identities"):
            raise AutoMLException("Shared selection of generated features is disabled")

    def _check_identity(self):
        if not self._identity_checked and "identity_schema_version" in self.info:
            if file_digest(self.root / self.info["directory"] / "generator.joblib") != self.info["generator_sha256"]:
                raise AutoMLException("Feature discovery fitted generator identity mismatch")
            self._identity_checked = True

    def _input(self, X):
        missing = set(self.info["input_columns"]) - set(X.columns)
        if missing:
            raise AutoMLException(
                f"Missing feature discovery inputs: {sorted(missing)}"
            )
        # Avoid changing the pandas representation of legacy (version 1) inputs
        # when all columns already match. Version 2 hashes values independently.
        if list(X.columns) != self.info["input_columns"]:
            X = X[self.info["input_columns"]]
        return discovery_input(X)

    def fit_transform(self, X, y):
        self._check_identity()
        X = self._input(X)
        if (
            fingerprint(X, y, self.info.get("fingerprint_version", 1))
            != self.info["fingerprint"]
        ):
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
        self._check_identity()
        if self.generator is None:
            GoldenFeatures = check_dependencies()
            self.generator = GoldenFeatures.load(
                self.root / self.info["directory"] / "generator.joblib"
            )
        frame = self.generator.transform(
            self._input(X), feature_set=self.feature_set
        )
        expected = self.info["sets"][self.feature_set]
        if set(frame.columns) != set(expected) or frame.columns.has_duplicates:
            raise AutoMLException("Feature discovery output schema changed")
        return frame[expected].rename(columns=self.info["rename"])

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
