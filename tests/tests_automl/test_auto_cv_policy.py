import json

import numpy as np
import pandas as pd
import pytest

from supervised import AutoML


@pytest.mark.parametrize("rows,minority,task,folds", [
    (4999, 1000, "binary_classification", 5),
    (5000, 250, "binary_classification", 10),
    (5000, 249, "binary_classification", 5),
    (10000, 249, "multiclass_classification", 5),
    (10000, 250, "multiclass_classification", 10),
    (4999, None, "regression", 5),
    (5000, None, "regression", 10),
    (512, 178, "binary_classification", 5),
])
@pytest.mark.parametrize("mode", ["Compete", "Optuna"])
def test_automatic_cv_boundaries(rows, minority, task, folds, mode):
    y = pd.Series(np.arange(rows, dtype=float))
    if minority is not None:
        counts = [rows - minority, minority]
        if task == "multiclass_classification":
            counts = [rows - minority - 1000, 1000, minority]
        y = pd.Series(pd.Categorical(np.repeat(np.arange(len(counts)), counts),
                                     categories=list(range(len(counts))) + [99]))
    model = AutoML(mode=mode, ml_task=task, verbose=0)
    model._mode, model._ml_task = mode, task
    model._configure_auto_cv(y)
    strategy = model._get_validation_strategy()
    assert strategy["k_folds"] == folds
    assert strategy.get("repeats", 1) == (3 if folds == 5 else 1)
    assert ("stratify" in strategy) == (task != "regression")
    assert model._auto_cv_policy["smallest_class_count"] == minority
    assert model._auto_cv_policy["training_rows"] == rows


@pytest.mark.parametrize("mode,expected", [("Perform", 5), ("Explain", None)])
def test_existing_mode_defaults(mode, expected):
    model = AutoML(mode=mode, ml_task="binary_classification", verbose=0)
    model._mode, model._ml_task = mode, "binary_classification"
    model._configure_auto_cv(pd.Series([0, 1] * 5000))
    assert model._get_validation_strategy().get("k_folds") == expected


@pytest.mark.parametrize("strategy", [
    {"validation_type": "kfold", "k_folds": 10, "shuffle": True, "stratify": True},
    {"validation_type": "split", "train_ratio": 0.8},
    {"validation_type": "custom"},
    {"validation_type": "kfold", "k_folds": 5, "shuffle": True, "stratify": True},
    {"validation_type": "kfold", "k_folds": 5, "repeats": 2, "shuffle": True},
])
def test_explicit_validation_is_preserved(strategy):
    model = AutoML(mode="Compete", validation_strategy=strategy,
                   ml_task="binary_classification", verbose=0)
    model._mode, model._ml_task = "Compete", "binary_classification"
    model._configure_auto_cv(pd.Series([0, 1] * 50))
    assert model._auto_cv_policy is None
    assert model._get_validation_strategy() == strategy


@pytest.mark.parametrize("budget,explicit,folds", [(3600, False, 5), (None, False, 5), (None, True, 10)])
def test_fit_resolves_folds_and_reload_preserves_policy(tmp_path, budget, explicit, folds):
    # Missing targets and sample weights must not inflate the minority count.
    X = pd.DataFrame(np.random.default_rng(7).normal(size=(650, 3)))
    y = pd.Series(["no"] * 450 + ["yes"] * 150 + [None] * 50)
    weights = np.array([1.] * 450 + [100.] * 150 + [1.] * 50)
    root = tmp_path / "run"
    model = AutoML(
        results_path=str(root), mode="Compete", algorithms=["Decision Tree"],
        ml_task="binary_classification", eval_metric="auc", total_time_limit=budget,
        validation_strategy={"validation_type": "kfold", "k_folds": 10,
                             "shuffle": True, "stratify": True} if explicit else "auto",
        start_random_models=1, hill_climbing_steps=0, stack_models=False if explicit else "auto",
        train_ensemble=False, golden_features=False, feature_discovery=False,
        features_selection=False, kmeans_features=False, mix_encoding=False,
        boost_on_errors=not explicit, explain_level=0, n_jobs=1, verbose=0,
    )
    model.fit(X, y, sample_weight=weights)
    params = json.loads((root / "params.json").read_text())
    assert params["validation_strategy"]["k_folds"] == folds
    assert model.tuner._validation_strategy["k_folds"] == folds
    repeats = 1 if explicit else 3
    assert params["validation_strategy"].get("repeats", 1) == repeats
    assert all(len(m.learners) == folds * repeats for m in model._models)
    policy = params["auto_cv_policy"]
    if explicit:
        assert policy is None
    else:
        assert policy["training_rows"] == 600
        assert policy["smallest_class_count"] == 150
        assert policy["max_folds"] == 5
        assert policy["repeats"] == 3
        assert not model._stack_models and not model._boost_on_errors
        assert not model.tuner._stack_models and not model.tuner._boost_on_errors
        assert "stack" not in model._time_ctrl._steps
        assert "boost_on_errors" not in model._time_ctrl._steps
        indices = [np.load(root / f"folds/fold_0_repeat_{r}_validation_indices.npy") for r in range(3)]
        assert not np.array_equal(indices[0], indices[1])
        assert not np.array_equal(indices[1], indices[2])
        if budget is not None:
            assert policy["time_budget_folds"] in (5, 10)
    reloaded = AutoML(results_path=str(root), verbose=0)
    np.testing.assert_allclose(model.predict_proba(X.iloc[:5]), reloaded.predict_proba(X.iloc[:5]))
    assert reloaded._auto_cv_policy == policy
    assert reloaded._validation_strategy["k_folds"] == folds
    assert reloaded._validation_strategy.get("repeats", 1) == repeats
    # Exercise the resume path with the saved candidates already completed.
    progress = json.loads((root / "progress.json").read_text())
    progress["fit_level"] = "not_so_random"
    (root / "progress.json").write_text(json.dumps(progress))
    fold_files = {p.name: p.read_bytes() for p in (root / "folds").iterdir()}
    resumed = AutoML(**model.get_params())
    resumed.fit(X, y, sample_weight=weights)
    assert resumed._validation_strategy.get("repeats", 1) == repeats
    assert fold_files == {p.name: p.read_bytes() for p in (root / "folds").iterdir()}
    # Older saved runs have no policy: loading them must not adopt the new rule.
    del params["auto_cv_policy"]
    (root / "params.json").write_text(json.dumps(params))
    reloaded.load(str(root))
    assert reloaded._auto_cv_policy is None
    assert reloaded._validation_strategy["k_folds"] == folds


@pytest.mark.parametrize("mode", ["Perform", "Compete", "Optuna"])
@pytest.mark.parametrize("budget", [30, 3600, None])
def test_small_data_repeats_have_no_budget_threshold(mode, budget, capsys):
    model = AutoML(mode=mode, ml_task="binary_classification", total_time_limit=budget)
    model._mode, model._ml_task = mode, "binary_classification"
    model._configure_auto_cv(pd.Series([0, 1] * 50))
    model._validation_strategy = model._get_validation_strategy()
    model._stack_models = model._boost_on_errors = True
    model._apply_constraints_stack_models()
    assert model._validation_strategy["k_folds"] == 5
    assert model._validation_strategy["repeats"] == 3
    assert not model._stack_models and not model._boost_on_errors
    assert "using 5-fold CV with 3 repeats" in capsys.readouterr().out
    # If the separate automatic timing probe picks holdout, no repeats are added.
    strategy = model._fastest_validation()
    model._apply_auto_cv_repeats(strategy)
    assert "repeats" not in strategy
