import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from supervised import AutoML
from supervised.algorithms import shared_boosting as backend
from supervised.algorithms import xgboost, lightgbm, catboost
from supervised.model_framework import ModelFramework
from supervised.shared_boosting import aggregate_oof
from supervised.utils.metric import Metric


@pytest.mark.parametrize("algorithm", ["Xgboost", "LightGBM", "CatBoost"])
@pytest.mark.parametrize(
    "task", ["binary_classification", "multiclass_classification", "regression"]
)
def test_shared_rounds_oof_and_reload(tmp_path, monkeypatch, algorithm, task):
    for module in [xgboost, lightgbm, catboost]:
        monkeypatch.setitem(module.additional, "max_rounds", 40)
    if task == "regression":
        values, target = make_regression(
            n_samples=180, n_features=5, noise=10, random_state=14
        )
        metric_name = "rmse"
    else:
        values, target = make_classification(
            n_samples=180,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            n_classes=3 if task.startswith("multi") else 2,
            random_state=14,
        )
        metric_name = "logloss" if task.startswith("multi") else "auc"
    X = pd.DataFrame(values, columns=list("abcde"))
    root = tmp_path / "automl"
    automl = AutoML(
        results_path=str(root),
        mode="Compete",
        algorithms=[algorithm],
        ml_task=task,
        eval_metric=metric_name,
        total_time_limit=300,
        train_ensemble=False,
        stack_models=False,
        explain_level=0,
        validation_strategy={
            "validation_type": "kfold",
            "k_folds": 3,
            "shuffle": True,
            "stratify": task != "regression",
        },
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        feature_discovery=False,
        boost_on_errors=False,
        mix_encoding=False,
        start_random_models=1,
        hill_climbing_steps=0,
        n_jobs=1,
        verbose=0,
    )
    automl.fit(X.copy(), target)
    names = json.loads((root / "params.json").read_text())["saved"]
    assert names
    metric = Metric({"name": metric_name})
    for name in names:
        path = root / name
        report = json.loads((path / "shared_boosting.json").read_text())
        count = report["selected_rounds"]
        assert count in {1, 10, 20, 30, 40}
        assert report["patience_rounds"] == 300
        scores = pd.read_csv(path / "shared_boosting_scores.csv")
        best = (
            scores.mean_fold_score.idxmax()
            if metric.is_negative()
            else scores.mean_fold_score.idxmin()
        )
        assert scores.iloc[best]["rounds"] == count
        loaded = ModelFramework.load(str(root), name)
        oof = pd.read_csv(path / "predictions_out_of_folds.csv")
        prediction_cols = [c for c in oof if "prediction" in c]
        folds = []
        for k, learner in enumerate(loaded.learners):
            va = np.load(root / f"folds/fold_{k}_validation_indices.npy")
            xp, _, _ = loaded.preprocessings[k].transform(X.iloc[va].copy(), None)
            pred = loaded.preprocessings[k].inverse_scale_target(learner.predict(xp))
            assert backend.round_count(learner) == count
            assert learner.params["selected_boosting_rounds"] == count
            np.testing.assert_allclose(
                np.asarray(pred).reshape(len(va), -1),
                oof[prediction_cols].iloc[va],
                rtol=1e-6,
                atol=1e-7,
            )
            folds.append(metric(oof.target.iloc[va], pred))
            learner.unload()
        assert report["selection_aggregation"] == "mean_fold"
        assert loaded.get_final_loss() == pytest.approx(np.mean(folds), rel=1e-7, abs=1e-7)
        sign = -1 if metric.is_negative() else 1
        assert report["mean_fold_score"] == pytest.approx(
            sign * np.mean(folds), rel=1e-7, abs=1e-7
        )
        assert not list(path.glob(".boosting-*"))
        leaderboard = automl.get_leaderboard(original_metric_values=True).set_index(
            "name"
        )
        assert leaderboard.loc[name, "mean_fold_metric_value"] == pytest.approx(
            report["mean_fold_score"]
        )
        assert leaderboard.loc[name, "selected_boosting_rounds"] == count
        before = loaded.predict(X.iloc[:12].copy())
        after = ModelFramework.load(str(root), name).predict(X.iloc[:12].copy())
        np.testing.assert_array_equal(before, after)


def test_checkpoint_patience_and_round_boundaries():
    stop = backend.CheckpointStopping()
    assert not stop.update(1, 0.5, True)
    assert not stop.update(10, 0.6, True)
    # Improvements between checkpoints must not reset patience.
    assert not stop.update(11, 0.9, True)
    assert not stop.update(300, 0.59, True)
    assert stop.update(310, 0.59, True)
    assert stop.best_round == 10 and stop.reason == "patience"
    assert backend.checkpoints(3) == [1]
    assert backend.checkpoints(25) == [1, 10, 20]
    minimized = backend.CheckpointStopping()
    minimized.update(1, 4, False)
    minimized.update(10, 2, False)
    assert minimized.best_round == 10


@pytest.mark.parametrize("algorithm", ["Xgboost", "LightGBM", "CatBoost"])
def test_time_limited_fold_excludes_entire_candidate(tmp_path, monkeypatch, capsys, algorithm):
    for module in [xgboost, lightgbm, catboost]:
        monkeypatch.setitem(module.additional, "max_rounds", 20)
    native_fit = backend.fit

    def interrupted_fit(learner, *args, **kwargs):
        assert args[-1] is None  # Routine budgets must not reach the native fit.
        stopping = native_fit(learner, *args, **kwargs)
        # One fold has already completed; a later interrupted fold must still
        # prevent the whole candidate from being published or ensembled.
        if learner.name == "learner_fold_1_repeat_0":
            stopping.reason = "time_limit"
        return stopping

    monkeypatch.setattr(backend, "fit", interrupted_fit)
    X, y = make_classification(n_samples=100, n_features=4, random_state=17)
    root = tmp_path / "interrupted"
    automl = AutoML(
        results_path=str(root), mode="Compete",
        algorithms=["Decision Tree", algorithm], eval_metric="auc",
        total_time_limit=300, n_jobs=1, explain_level=0,
        validation_strategy={"validation_type": "kfold", "k_folds": 2,
                             "repeats": 2, "shuffle": True, "stratify": True},
        train_ensemble=True, stack_models=False, start_random_models=1,
        hill_climbing_steps=0, golden_features=False, feature_discovery=False,
        kmeans_features=False, features_selection=False, mix_encoding=False,
        boost_on_errors=False,
    )
    automl.fit(X, y)
    skips = json.loads((root / "budget_skips.json").read_text())
    interrupted = [s for s in skips if s.get("phase") == "boosting_fit"]
    assert interrupted
    for skip in interrupted:
        assert skip["learner"] == "learner_fold_1_repeat_0"
        assert skip["trained_rounds"] == 20
        assert skip["stop_reason"] == "time_limit"
        assert skip["learner_allowance"] is None
        assert skip["learner_elapsed"] >= 0
        path = root / skip["name"]
        assert not (path / "framework.json").exists()
        assert not (path / "shared_boosting.json").exists()
        assert not list(path.glob(".boosting-*"))
    excluded = {s["name"] for s in interrupted}
    params = json.loads((root / "params.json").read_text())
    assert excluded.isdisjoint(params["saved"])
    assert params["best_model"] not in excluded
    assert excluded.isdisjoint(automl.get_leaderboard()["name"])
    for model in automl._models:
        if model.get_type() == "Ensemble":
            assert excluded.isdisjoint(model.involved_model_names())
    assert "skipping the incomplete candidate" in capsys.readouterr().out
    assert not (root / "errors.md").exists()
    assert len(AutoML(results_path=str(root)).predict(X[:3])) == 3


def test_repeated_oof_averages_predictions_and_retains_weights():
    a = pd.DataFrame(
        {"target": [0, 1], "prediction": [0.2, 0.8], "sample_weight": [2.0, 3.0]},
        index=[0, 1],
    )
    b = pd.DataFrame(
        {"target": [1, 0], "prediction": [0.6, 0.4], "sample_weight": [3.0, 2.0]},
        index=[1, 0],
    )
    result = aggregate_oof([a, b])
    np.testing.assert_allclose(result.prediction, [0.3, 0.7])
    np.testing.assert_array_equal(result.sample_weight, [2.0, 3.0])


@pytest.mark.parametrize("algorithm", ["Xgboost", "LightGBM", "CatBoost"])
def test_weighted_repeated_cv_and_categorical_labels(tmp_path, monkeypatch, algorithm):
    for module in [xgboost, lightgbm, catboost]:
        monkeypatch.setitem(module.additional, "max_rounds", 20)
    X, y = make_classification(n_samples=120, n_features=4, random_state=17)
    X = pd.DataFrame(X, columns=list("abcd"))
    X["category"] = np.where(X.a > 0, "positive", "negative")
    labels = np.where(y, "yes", "no")
    weights = np.linspace(0.5, 2, len(y))
    root = tmp_path / "weighted"
    automl = AutoML(
        results_path=str(root),
        mode="Compete",
        algorithms=[algorithm],
        eval_metric="auc",
        total_time_limit=300,
        n_jobs=1,
        verbose=0,
        validation_strategy={
            "validation_type": "kfold",
            "k_folds": 2,
            "repeats": 2,
            "stratify": True,
            "shuffle": True,
        },
        train_ensemble=False,
        stack_models=False,
        explain_level=0,
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        feature_discovery=False,
        boost_on_errors=False,
        mix_encoding=False,
        start_random_models=1,
        hill_climbing_steps=0,
    )
    automl.fit(X.copy(), labels, sample_weight=weights)
    model = automl._models[0]
    path = root / model.get_name()
    loaded = ModelFramework.load(str(root), model.get_name())
    assert len(loaded.learners) == 4
    report = json.loads((path / "shared_boosting.json").read_text())
    frames = []
    losses = []
    metric = Metric({"name": "auc"})
    for i, learner in enumerate(loaded.learners):
        repeat, fold = divmod(i, 2)
        # Repeated folds use a repeat suffix in saved indices.
        va = np.load(root / f"folds/fold_{fold}_repeat_{repeat}_validation_indices.npy")
        data, _, _ = loaded.preprocessings[i].transform(X.iloc[va].copy(), None)
        pred = learner.predict(data)
        learner.unload()
        frames.append(
            pd.DataFrame(
                {"target": y[va], "prediction": pred, "sample_weight": weights[va]},
                index=va,
            )
        )
        losses.append(metric(y[va], pred, weights[va]))
    expected = aggregate_oof(frames)
    saved = pd.read_csv(path / "predictions_out_of_folds.csv")
    pc = next(c for c in saved if "prediction" in c)
    np.testing.assert_allclose(saved[pc], expected.prediction, atol=1e-7, rtol=0)
    assert report["pooled_score"] == pytest.approx(
        -metric(y, expected.prediction, weights), abs=1e-7
    )
    assert report["mean_fold_score"] == pytest.approx(-np.mean(losses), abs=1e-7)
    assert loaded.get_final_loss() == pytest.approx(np.mean(losses), abs=1e-7)


def test_temporary_predictions_cleaned_on_failure(tmp_path, monkeypatch):
    from supervised.shared_boosting import SharedBoostingSelection
    from supervised.exceptions import BudgetExceededError
    import supervised.shared_boosting as shared

    selection = SharedBoostingSelection(tmp_path, Metric({"name": "auc"}), deadline=0)
    cache = selection.cache
    np.save(cache / "checkpoint.npy", np.arange(10))
    try:
        with pytest.raises(BudgetExceededError):
            selection.check_budget()
    finally:
        selection.close()
    assert not cache.exists()


def test_common_prefix_intersection_and_exact_tie(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from supervised.callbacks.early_stopping import EarlyStopping
    from supervised.shared_boosting import SharedBoostingSelection

    selection = SharedBoostingSelection(tmp_path, Metric({"name": "auc"}))
    early = EarlyStopping({"metric": {"name": "auc"}})
    early.target_columns = "target"
    saved = []
    learners = []
    for i, counts in enumerate([[1, 10, 20], [1, 10]]):
        uid = str(i)
        selection.folds.append({"uid": uid, "learner": uid, "checkpoints": counts})
        early.best_y_predicted[uid] = pd.DataFrame(
            {"target": [0, 1], "prediction": [0.9, 0.1]}, index=[2 * i, 2 * i + 1]
        )
        for count in counts:
            np.save(selection.cache / f"{uid}-{count}.npy", np.array([0.2, 0.8]))
        learners.append(
            SimpleNamespace(
                uid=uid,
                model_file_path=uid,
                reload=lambda: None,
                save=lambda path: saved.append(path),
                unload=lambda: None,
            )
        )
    selected = []
    monkeypatch.setattr(
        backend,
        "select_rounds",
        lambda learner, count: selected.append((learner.uid, count)),
    )
    framework = SimpleNamespace(
        learners=learners, callbacks=SimpleNamespace(get=lambda name: early), params={}
    )
    try:
        selection.finalize(framework)
        early.on_framework_train_end(None)
        assert selected == [("0", 1), ("1", 1)]
        assert saved == ["0", "1"]
        report = json.loads((tmp_path / "shared_boosting.json").read_text())
        assert [row["rounds"] for row in report["scores"]] == [1, 10]
        assert early.final_loss == -1
        np.testing.assert_array_equal(early.best_y_oof.prediction, [0.2, 0.8, 0.2, 0.8])
    finally:
        selection.close()


def test_checkpoint_time_limit_can_stop_between_ticks(monkeypatch):
    from types import SimpleNamespace

    clock = [0.0]
    monkeypatch.setattr(backend, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    stop = backend.CheckpointStopping(max_time=1)
    assert not stop.update(1, 0.6, True)
    clock[0] = 2.0
    assert stop.update(2, 0.7, True)
    assert stop.best_round == 1 and stop.reason == "time_limit"


def test_shared_round_selection_uses_fold_scores_before_repeat_averaging(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from supervised.callbacks.early_stopping import EarlyStopping
    from supervised.shared_boosting import SharedBoostingSelection

    selection = SharedBoostingSelection(tmp_path, Metric({"name": "mse"}))
    early = EarlyStopping({"metric": {"name": "mse"}})
    early.target_columns = "target"
    learners = []
    for i in range(2):
        uid = str(i)
        selection.folds.append({"uid": uid, "learner": uid, "checkpoints": [1, 10]})
        early.best_y_predicted[uid] = pd.DataFrame({"target": [0], "prediction": [1]})
        # At round 1, repeat errors cancel exactly but each fold has MSE 16.
        np.save(selection.cache / f"{uid}-1.npy", np.array([4 if i == 0 else -4]))
        np.save(selection.cache / f"{uid}-10.npy", np.array([1]))
        learners.append(SimpleNamespace(
            uid=uid, model_file_path=uid, reload=lambda: None,
            save=lambda path: None, unload=lambda: None,
        ))
    selected = []
    monkeypatch.setattr(backend, "select_rounds", lambda learner, count: selected.append(count))
    framework = SimpleNamespace(
        learners=learners, callbacks=SimpleNamespace(get=lambda name: early), params={}
    )
    try:
        selection.finalize(framework)
        early.on_framework_train_end(None)
        assert selected == [10, 10]
        assert early.final_loss == 1
        assert framework.params["shared_boosting"]["selection_aggregation"] == "mean_fold"
    finally:
        selection.close()


def test_leaderboard_keeps_legacy_pooled_scores_when_loading_old_models():
    model = ModelFramework(
        {
            "learner": {"model_type": "Xgboost"},
            "validation_strategy": {},
            "shared_boosting": {"mean_fold_score": 0.75, "selected_rounds": 20},
        }
    )
    model.final_loss = -0.8
    model.train_time = 1
    automl = AutoML(verbose=0)
    automl._models = [model]
    automl._eval_metric = "auc"
    internal = automl.get_leaderboard()
    assert internal.metric_value.iloc[0] == -0.8
    assert internal.mean_fold_metric_value.iloc[0] == -0.75
    external = automl.get_leaderboard(original_metric_values=True)
    assert external.metric_value.iloc[0] == 0.8
    assert external.mean_fold_metric_value.iloc[0] == 0.75
    assert external.selected_boosting_rounds.iloc[0] == 20
