import json
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from threadpoolctl import threadpool_limits

from supervised import AutoML
from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.ensemble import Ensemble
from supervised.model_framework import ModelFramework
from supervised.preprocessing.kmeans_transformer import KMeansTransformer


def legacy_path(path, style):
    if style == "windows":
        return path.replace("/", "\\")
    if style == "mixed":
        return path.replace("/", "\\", 1)
    return path


@pytest.mark.parametrize(
    "task,classes,algorithm",
    [
        ("regression", 0, "Decision Tree"),
        ("regression", 0, "LightGBM"),
        ("binary_classification", 2, "Decision Tree"),
        ("multiclass_classification", 3, "Decision Tree"),
    ],
)
def test_relocated_legacy_artifacts_keep_predictions(
    tmp_path, task, classes, algorithm, monkeypatch
):
    if classes:
        X, y = make_classification(
            n_samples=90,
            n_features=5,
            n_informative=3,
            n_classes=classes,
            n_clusters_per_class=1,
            random_state=123,
        )
    else:
        X, y = make_regression(n_samples=90, n_features=5, random_state=123)
    X = pd.DataFrame(X, columns=list("abcde"))
    root = tmp_path / "original"
    model = AutoML(
        results_path=str(root),
        ml_task=task,
        algorithms=["Baseline", algorithm],
        train_ensemble=True,
        stack_models=False,
        validation_strategy={
            "validation_type": "kfold",
            "k_folds": 2,
            "shuffle": True,
            "stratify": bool(classes),
            "random_seed": 123,
        },
        explain_level=0,
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        boost_on_errors=False,
        mix_encoding=False,
        start_random_models=1,
        hill_climbing_steps=0,
        total_time_limit=60,
        n_jobs=1,
        verbose=0,
    )
    model.fit(X, y)
    expected = model.predict_all(X)
    references = {
        m.get_name(): (m.predict(X), m.get_out_of_folds()) for m in model._models
    }
    assert any(isinstance(m, Ensemble) for m in model._models)
    backup = tmp_path / "backup"
    root.rename(backup)
    for style in ["posix", "windows", "mixed"]:
        moved = tmp_path / ("relocated ż space " + style)
        shutil.copytree(backup, moved)
        for file in moved.glob("*/framework.json"):
            data = json.loads(file.read_text())
            assert all("\\" not in path for path in data["saved"])
            assert "\\" not in data["predictions_fname"]
            data["saved"] = [legacy_path(path, style) for path in data["saved"]]
            data["predictions_fname"] = legacy_path(data["predictions_fname"], style)
            file.write_text(json.dumps(data))
        for file in moved.glob("*/ensemble.json"):
            data = json.loads(file.read_text())
            assert "\\" not in data["predictions_fname"]
            data["predictions_fname"] = legacy_path(data["predictions_fname"], style)
            file.write_text(json.dumps(data))
        # The original root is absent for every path style.
        with monkeypatch.context() as patch:
            patch.setattr(
                BaseAlgorithm, "reload",
                lambda self: pytest.fail("AutoML.load must not load learners"),
            )
            loaded = AutoML(results_path=str(moved))
            loaded.load(str(moved))
        pd.testing.assert_frame_equal(loaded.predict_all(X), expected)
        for lazy in [True, False]:
            models = {}
            for member in model._models:
                name = member.get_name()
                restored = (
                    Ensemble.load(str(moved), name, models)
                    if isinstance(member, Ensemble)
                    else ModelFramework.load(str(moved), name, lazy_load=lazy)
                )
                models[name] = restored
                np.testing.assert_array_equal(restored.predict(X), references[name][0])
                pd.testing.assert_frame_equal(
                    restored.get_out_of_folds(),
                    references[name][1],
                    check_exact=False,
                    rtol=1e-12,
                    atol=1e-12,
                )


@pytest.mark.parametrize("style", ["posix", "windows", "mixed"])
def test_kmeans_legacy_path_after_relocation(tmp_path, style):
    root = tmp_path / "original"
    (root / "model" / "nested").mkdir(parents=True)
    X = pd.DataFrame(
        np.random.default_rng(123).normal(size=(40, 3)), columns=list("abc")
    )
    transformer = KMeansTransformer(str(root), "model/nested", 0)
    with threadpool_limits(limits=1):
        transformer.fit(X, None)
        expected = transformer.transform(X.copy())
        data = transformer.to_json()
        assert data["result_file"] == "model/nested/kmeans_fold_0.joblib"
        data["result_file"] = legacy_path(data["result_file"], style)
        moved = tmp_path / "moved ż space"
        root.rename(moved)
        loaded = KMeansTransformer()
        loaded.from_json(data, str(moved))
        pd.testing.assert_frame_equal(loaded.transform(X.copy()), expected)
        assert loaded.to_json()["result_file"] == "model/nested/kmeans_fold_0.joblib"
