"""Retrieve saved importance and compute missing learner artifacts on demand."""

import copy
import json
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_consistent_length, column_or_1d

from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION, REGRESSION
from supervised.ensemble import Ensemble
from supervised.exceptions import AutoMLException
from supervised.model_framework import ModelFramework
from supervised.utils.importance import (
    PermutationImportance,
    aggregate_importance,
    load_importance_file,
)


def get_feature_importance(automl, model="best", kind="raw", X=None, y=None):
    if not isinstance(model, str):
        raise ValueError("model must be 'best', 'all', or a saved model name")
    if kind not in ("raw", "normalized"):
        raise ValueError("kind must be 'raw' or 'normalized'")
    if (X is None) != (y is None):
        raise ValueError("Provide both X and y to compute missing feature importance")
    if automl._best_model is None:
        automl._check_can_load()
    if automl._best_model is None:
        raise AutoMLException(
            "This model has not been fitted yet. Please call `fit()` first."
        )

    names = list(
        dict.fromkeys(
            list(automl._model_subpaths) + [m.get_name() for m in automl._models]
        )
    )
    if model == "best":
        names = [automl._best_model.get_name()]
    elif model != "all":
        if model not in names:
            raise AutoMLException(
                f"Unknown model '{model}'. Available models: {', '.join(names)}"
            )
        names = [model]

    reader = _ImportanceReader(automl, X, y)
    frames = []
    for name in names:
        importance = reader.read(name)
        if importance is None:
            continue
        importance = importance.copy()
        if kind == "normalized":
            span = importance.max() - importance.min()
            importance = (
                (importance - importance.min()) / span if span else importance * 0.0
            )
        frame = importance.rename_axis("feature").reset_index(name="importance")
        frame.insert(0, "model", name)
        frames.append(
            frame.sort_values(["importance", "feature"], ascending=[False, True])
        )
    if not frames:
        return pd.DataFrame(
            {
                "model": pd.Series(dtype=str),
                "feature": pd.Series(dtype=str),
                "importance": pd.Series(dtype=float),
            }
        )
    return pd.concat(frames, ignore_index=True)


class _ImportanceReader:
    def __init__(self, automl, X, y):
        self.automl = automl
        self.root = automl._results_path
        self.X, self.y = X, y
        self.prepared = False
        self.models = {m.get_name(): m for m in automl._models}
        self.vectors = {}
        self.stacked_X = None

    def _warn(self, name, reason):
        warnings.warn(
            f"Feature importance for '{name}': {reason}", UserWarning, stacklevel=3
        )

    def _load_model(self, name):
        if name not in self.models:
            ensemble_path = os.path.join(self.root, name, "ensemble.json")
            if os.path.isfile(ensemble_path):
                with open(ensemble_path) as stream:
                    members = json.load(stream)["selected_models"]
                for member in members:
                    self._load_model(member["model"])
                self.models[name] = Ensemble.load(self.root, name, self.models)
            else:
                self.models[name] = ModelFramework.load(self.root, name, lazy_load=True)
        return self.models[name]

    def _prepare_data(self):
        if self.prepared:
            return
        X = self.X.copy() if isinstance(self.X, pd.DataFrame) else np.asarray(self.X)
        self.X = self.automl._build_dataframe(X)
        columns = self.automl._data_info["columns"]
        missing = [c for c in columns if c not in self.X.columns]
        if missing:
            raise ValueError(f"Missing columns in X: {', '.join(missing)}")
        self.X = self.X[columns].copy()
        self.automl._validate_X_predict(self.X)
        self.y = pd.Series(column_or_1d(self.y), name="target")
        check_consistent_length(self.X, self.y)
        if self.y.isna().any():
            raise ValueError("y must not contain missing targets")
        if self.automl._ml_task == MULTICLASS_CLASSIFICATION:
            self.y = self.y.astype(str)
        self.prepared = True

    def _model_data(self, framework):
        if not framework._is_stacked:
            return self.X
        if self.stacked_X is None:
            # Use the saved stacking inputs, including when the best model did
            # not need stacking and AutoML.load() loaded only its dependencies.
            with open(os.path.join(self.root, "params.json")) as stream:
                names = json.load(stream).get("stacked")
            if not names:
                raise ValueError("saved stacking dependencies are unavailable")
            context = copy.copy(self.automl)
            context._stacked_models = [self._load_model(name) for name in names]
            self.stacked_X = context.get_stacked_data(self.X.copy(), mode="predict")
        return self.stacked_X

    def read(self, name):
        if name in self.vectors:
            return self.vectors[name]
        # Also prevents accidental recursion in malformed ensemble metadata.
        self.vectors[name] = None
        directory = os.path.join(self.root, name)
        ensemble_path = os.path.join(directory, "ensemble.json")
        try:
            with open(
                ensemble_path
                if os.path.isfile(ensemble_path)
                else os.path.join(directory, "framework.json")
            ) as stream:
                metadata = json.load(stream)
        except (OSError, ValueError) as exc:
            self._warn(name, f"cannot read model metadata ({exc}); returning no rows")
            return None
        if "selected_models" in metadata:
            result = self._ensemble(name, metadata["selected_models"])
        else:
            result = self._learners(name, directory, metadata)
        self.vectors[name] = result
        return result

    def _ensemble(self, name, members):
        vectors, weights = [], []
        for member in members:
            vector = self.read(member["model"])
            if vector is None:
                self._warn(
                    name, "a member's importance is unavailable; returning no rows"
                )
                return None
            vectors.append(vector)
            weights.append(float(member["repeat"]))
        if not vectors or sum(weights) <= 0:
            self._warn(name, "no weighted members are available; returning no rows")
            return None
        # A missing feature contributes zero for that member. This is a summary
        # of member importances, not a permutation measurement of the ensemble.
        frame = pd.concat(vectors, axis=1).fillna(0.0)
        return frame.mul(np.asarray(weights) / sum(weights), axis=1).sum(axis=1)

    def _learners(self, name, directory, metadata):
        learner_names = [learner["name"] for learner in metadata.get("learners", [])]
        paths = [
            os.path.join(directory, f"{learner}_importance.csv")
            for learner in learner_names
        ]
        vectors = [load_importance_file(path) for path in paths]
        missing = [i for i, vector in enumerate(vectors) if vector is None]
        if not learner_names:
            self._warn(name, "no learners are available; returning no rows")
        elif missing and self.X is None:
            self._warn(
                name,
                "missing or invalid learner importance; provide X and y to compute it. "
                "Returning available learners only",
            )
        elif missing:
            self._prepare_data()
            try:
                framework = self._load_model(name)
                X = self._model_data(framework)
                for i in missing:
                    try:
                        vectors[i] = self._compute(framework, i, X, paths[i])
                    except Exception as exc:
                        self._warn(
                            name,
                            f"cannot compute {learner_names[i]} ({exc}); "
                            "returning available learners only",
                        )
            except Exception as exc:
                self._warn(
                    name,
                    f"cannot load evaluation model ({exc}); returning available learners only",
                )
        return aggregate_importance(vectors)

    def _compute(self, framework, index, X, path):
        learner = framework.learners[index]
        if learner.algorithm_short_name == "Nearest Neighbors":
            from supervised.algorithms.knn import KNN_INTERPRET_ROWS_LIMIT

            if len(X) > KNN_INTERPRET_ROWS_LIMIT:
                raise ValueError(
                    f"kNN evaluation rows exceed the interpretation limit of {KNN_INTERPRET_ROWS_LIMIT}"
                )
        X_eval, y_eval, _ = framework.preprocessings[index].transform(
            X.copy(), self.y.copy()
        )
        was_unloaded = learner.model is None
        try:
            learner.reload()
            if framework._ml_task != REGRESSION and not hasattr(learner, "classes_"):
                learner.classes_ = getattr(
                    learner.model,
                    "classes_",
                    np.arange(learner.params.get("num_class", 2)),
                )
            frame = PermutationImportance.compute(
                learner, X_eval, y_eval, framework._ml_task, self.automl._n_jobs
            )
        finally:
            if was_unloaded:
                learner.model = None
        # Save only missing/invalid files, atomically; failed persistence should
        # not discard a successfully computed result.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=os.path.dirname(path), suffix=".tmp", delete=False
            ) as stream:
                temporary = stream.name
                frame.to_csv(stream, index=False)
            os.replace(temporary, path)
        except OSError as exc:
            self._warn(
                framework.get_name(), f"computed importance could not be saved ({exc})"
            )
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        return frame.set_index("feature")["mean_importance"].rename("importance")
