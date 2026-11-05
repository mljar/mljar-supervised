import importlib.util
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from supervised import AutoML
from supervised.algorithms.realmlp import validate_options
from supervised.model_framework import ModelFramework
from supervised.shared_realmlp import SharedRealMLPSelection, checkpoint_epochs
from supervised.utils.metric import Metric
from supervised.utils.validation_scores import mean_fold_score


def test_schedule_and_options():
    assert checkpoint_epochs(256) == list(range(10, 251, 10)) + [256]
    assert checkpoint_epochs(20) == [10, 20]
    assert checkpoint_epochs(5) == [5]
    for value in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match="checkpoint_interval"):
            validate_options({"checkpoint_interval": value})
    with pytest.raises(ValueError, match="boolean"):
        validate_options({"shared_checkpoint_selection": "yes"})


def automl(root, task, options, validation=None):
    return AutoML(
        results_path=str(root), mode="Compete", algorithms=["RealMLP"],
        ml_task=task, eval_metric="rmse" if task == "regression" else (
            "auc" if task == "binary_classification" else "logloss"),
        total_time_limit=120, train_ensemble=False, stack_models=False,
        explain_level=0, golden_features=False, kmeans_features=False,
        features_selection=False, feature_discovery=False, boost_on_errors=False,
        mix_encoding=False, start_random_models=1, hill_climbing_steps=0,
        n_jobs=1, verbose=0,
        validation_strategy=validation or dict(validation_type="kfold", k_folds=2,
                                              repeats=2, shuffle=True,
                                              stratify=task != "regression"),
        custom_params={"realmlp": dict(device="cpu", n_epochs=12, **options)},
    )


optional = pytest.mark.skipif(importlib.util.find_spec("pytabkit") is None,
                              reason="optional dependency")


@optional
@pytest.mark.parametrize("task", ["regression", "binary_classification", "multiclass_classification"])
def test_shared_epochs_fold_predictions_and_portable_reload(tmp_path, task):
    if task == "regression":
        X, y = make_regression(n_samples=90, n_features=4, noise=8, random_state=12)
        y = 1000 + 20 * y
    else:
        X, y = make_classification(n_samples=90, n_features=4, n_informative=3,
                                   n_redundant=0, n_classes=3 if task.startswith("multi") else 2,
                                   random_state=12)
    X = pd.DataFrame(X, columns=list("abcd"))
    root = tmp_path / "original"
    # The omitted flag tests that shared selection is the CV default. Explicit
    # per-fold early stopping must not truncate the shared candidate schedule.
    model = automl(root, task, {"use_early_stopping": True})
    model.fit(X, y)
    before = model.predict_all(X)
    saved = json.loads((root / "params.json").read_text())["saved"]
    assert saved
    metric = Metric({"name": "rmse" if task == "regression" else (
        "auc" if task == "binary_classification" else "logloss")})
    for name in saved:
        path = root / name
        report = json.loads((path / "shared_realmlp.json").read_text())
        assert [row["epoch"] for row in report["scores"]] == [10, 12]
        best = sorted(report["scores"], key=lambda r: (
            -r["mean_fold_score"] if metric.is_negative() else r["mean_fold_score"], r["epoch"]))[0]
        assert report["selected_epoch"] == best["epoch"]
        for point in report["scores"]:
            logged = []
            for fold in report["folds"]:
                curve = pd.read_csv(path / f"{fold['learner']}_training.log", header=None)
                value = float(curve.loc[curve[0] == point["epoch"], 2].iloc[0])
                # pytabkit logs 1-AUC; MLJAR reports AUC. The other test metrics
                # use the same scale. Logs are rounded to six decimals.
                logged.append(1 - value if task == "binary_classification" else value)
            assert point["mean_fold_score"] == pytest.approx(np.mean(logged), rel=1e-5, abs=1e-6)
        loaded = ModelFramework.load(str(root), name)
        frames = loaded.get_validation_predictions()
        pred_cols = [c for c in frames if "prediction" in c]
        for i, learner in enumerate(loaded.learners):
            repeat, fold = divmod(i, 2)
            indices = np.load(root / f"folds/fold_{fold}_repeat_{repeat}_validation_indices.npy")
            prep = loaded.preprocessings[i]
            xp, _, _ = prep.transform(X.iloc[indices].copy(), None)
            actual = np.asarray(prep.inverse_scale_target(learner.predict(xp))).reshape(len(indices), -1)
            expected = frames.xs(i, level="validation_fold").loc[indices, pred_cols]
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
            assert learner.training_info["best_epoch"] == report["selected_epoch"]
            log = json.loads((path / f"{learner.name}_training.log.json").read_text())
            assert log["best_epoch"] == report["selected_epoch"]
            assert log["individual_best_epoch"] == learner.training_info["individual_best_epoch"]
            assert learner.training_info["completed_epochs"] == 12
            assert learner.model._checkpoint_directory is None
            learner.unload()
        p = frames[pred_cols[0]] if len(pred_cols) == 1 else frames[pred_cols]
        assert loaded.get_final_loss() == pytest.approx(mean_fold_score(metric, frames.target, p))
        completion = json.loads((path / "training_completion.json").read_text())
        assert completion["status"] == "trained"
    assert not list(root.rglob(".realmlp-*"))
    moved = tmp_path / "moved"
    shutil.move(root, moved)
    after = AutoML(results_path=str(moved)).predict_all(X)
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)


@optional
@pytest.mark.parametrize("split", [False, True])
def test_opt_out_and_single_split_keep_individual_checkpoints(tmp_path, split):
    X, y = make_regression(n_samples=70, n_features=3, random_state=9)
    validation = dict(validation_type="split", train_ratio=0.8, shuffle=True) if split else None
    options = {} if split else {"shared_checkpoint_selection": False}
    m = automl(tmp_path / "model", "regression", options, validation)
    m.fit(pd.DataFrame(X), y)
    assert m._models
    assert not list(tmp_path.rglob("shared_realmlp.json"))
    assert not list(tmp_path.rglob("epoch_*.pt"))
    for model in m._models:
        assert "shared_realmlp" not in model.params


@optional
@pytest.mark.parametrize("failure", ["restore", "missing_checkpoint"])
def test_failed_restore_is_not_trained_or_published(tmp_path, monkeypatch, failure):
    from supervised.algorithms import _realmlp_backend as backend

    original = SharedRealMLPSelection.finalize

    def fail_during_finalize(selection, framework):
        before = {Path(l.model_file_path): hashlib.sha256(Path(l.model_file_path).read_bytes()).digest()
                  for l in framework.learners}
        if failure == "missing_checkpoint":
            next(selection.cache.glob("*/epoch_*.pt")).unlink()
            return original(selection, framework)
        restore = backend.restore_checkpoint
        calls = 0
        def fail(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected restore failure")
            return restore(*args)
        with monkeypatch.context() as patch:
            patch.setattr(backend, "restore_checkpoint", fail)
            try:
                return original(selection, framework)
            finally:
                assert all(hashlib.sha256(p.read_bytes()).digest() == digest for p, digest in before.items())

    monkeypatch.setattr(SharedRealMLPSelection, "finalize", fail_during_finalize)
    X, y = make_regression(n_samples=70, n_features=3, random_state=9)
    m = automl(tmp_path / "model", "regression", {"checkpoint_interval": 20})
    with pytest.raises(Exception):
        m.fit(pd.DataFrame(X), y)
    assert not list(tmp_path.rglob("shared_realmlp.json"))
    assert not list(tmp_path.rglob(".realmlp-*"))
    reports = list(tmp_path.rglob("training_completion.json"))
    assert reports
    assert all(json.loads(p.read_text())["status"] == "not_trained" for p in reports)
