import json

import numpy as np
import pandas as pd
import pytest

from supervised import AutoML
from supervised.algorithms.extra_trees import additional as extra_trees_settings
from supervised.algorithms.random_forest import additional as random_forest_settings
from supervised.base_automl import BaseAutoML
from supervised.ensemble import Ensemble
from supervised.model_framework import ModelFramework


def framework(name, loss):
    model = ModelFramework(
        {
            "name": name,
            "learner": {"model_type": "Decision Tree", "ml_task": "regression"},
            "validation_strategy": {},
        }
    )
    model.final_loss = loss
    model.train_time = 1.0
    return model


def load_ensemble(tmp_path, loss):
    directory = tmp_path / "Ensemble"
    directory.mkdir(exist_ok=True)
    (directory / "ensemble.json").write_text(
        json.dumps(
            {
                "name": "Ensemble",
                "optimize_metric": "rmse",
                "ml_task": "regression",
                "final_loss": loss,
                "train_time": 1.0,
                "selected_models": [],
            }
        )
    )
    return Ensemble.load(str(tmp_path), "Ensemble", {})


@pytest.mark.parametrize("loss", ["0.12345678901234567", 0.125, None])
def test_ensemble_loaded_loss_is_numeric(tmp_path, loss):
    ensemble = load_ensemble(tmp_path, loss)
    if loss is None:
        assert ensemble.get_final_loss() is None
    else:
        assert isinstance(ensemble.get_final_loss(), float)
        assert ensemble.get_final_loss() == float(loss)


def test_stacking_orders_reloaded_and_new_losses_numerically(tmp_path):
    directory = tmp_path / "old"
    directory.mkdir()
    (directory / "framework.json").write_text(
        json.dumps(
            {
                "params": framework("old", 10).params,
                "name": "old",
                "final_loss": "10.0",
                "train_time": 1.0,
                "learners": [],
                "saved": [],
                "preprocessing": [],
            }
        )
    )
    old = ModelFramework.load(str(tmp_path), "old")
    new = framework("new", 2.0)
    ensemble = load_ensemble(tmp_path, "1.0")
    automl = AutoML(verbose=0)
    automl._models = [old, new, ensemble]
    automl._eval_metric = "rmse"
    automl._perform_model_stacking()
    # Ensembles remain available for final selection, but no longer introduce
    # hidden dependencies as stacking inputs. Loaded losses still sort numerically.
    assert [m.get_name() for m in automl._stacked_models] == ["new", "old"]
    assert ensemble in automl._models
    assert pd.api.types.is_numeric_dtype(automl.get_leaderboard().metric_value)


@pytest.mark.parametrize(
    "task,classes",
    [("regression", 0), ("binary_classification", 2), ("multiclass_classification", 3)],
)
def test_optuna_resume_mixes_loaded_and_new_models_before_stacking(
    tmp_path, monkeypatch, task, classes
):
    # Keep real Optuna trials and fold training small for this resume test.
    for settings in [random_forest_settings, extra_trees_settings]:
        monkeypatch.setitem(settings, "trees_in_step", 5)
        monkeypatch.setitem(settings, "max_steps", 2)
        monkeypatch.setitem(settings, "early_stopping_rounds", 2)
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(90, 3)), columns=list("abc"))
    y = (
        np.arange(90) % classes
        if classes
        else X.a.to_numpy() + 0.1 * rng.normal(size=90)
    )
    directory = tmp_path / "optuna"
    options = dict(
        results_path=str(directory),
        mode="Optuna",
        ml_task=task,
        algorithms=[
            "Baseline",
            "Linear",
            "Decision Tree",
            "Random Forest",
            "Extra Trees",
        ],
        train_ensemble=True,
        stack_models=True,
        explain_level=0,
        validation_strategy={
            "validation_type": "kfold",
            "k_folds": 2,
            "shuffle": True,
            "stratify": bool(classes),
            "random_seed": 123,
        },
        total_time_limit=600,
        optuna_time_budget=1,
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        boost_on_errors=False,
        mix_encoding=False,
        start_random_models=1,
        hill_climbing_steps=0,
        n_jobs=1,
        verbose=0,
    )
    original = BaseAutoML.train_model

    def interrupt(self, params):
        if len(self._models) == 2:
            raise KeyboardInterrupt("interrupt before third model")
        return original(self, params)

    monkeypatch.setattr(BaseAutoML, "train_model", interrupt)
    first = AutoML(**options)
    with pytest.raises(KeyboardInterrupt, match="third model"):
        first.fit(X.copy(), y)
    initial_names = [m.get_name() for m in first._models]
    assert len(initial_names) == 2
    for file in directory.glob("*/framework.json"):
        data = json.loads(file.read_text())
        data["final_loss"] = str(data["final_loss"])
        file.write_text(json.dumps(data))
    monkeypatch.setattr(BaseAutoML, "train_model", original)
    original_stacking = BaseAutoML._perform_model_stacking
    checked = []

    def check_stacking(self):
        names = [m.get_name() for m in self._models]
        assert set(initial_names).issubset(names)
        assert len(names) > len(initial_names)
        assert all(isinstance(m.get_final_loss(), float) for m in self._models)
        assert pd.api.types.is_numeric_dtype(self.get_leaderboard().metric_value)
        checked.append(True)
        return original_stacking(self)

    monkeypatch.setattr(BaseAutoML, "_perform_model_stacking", check_stacking)
    resumed = AutoML(**options)
    resumed.fit(X.copy(), y)
    assert checked
    # The checks above apply during training. Reloading for inference may load
    # only the selected model's dependencies, excluding unused models.
    monkeypatch.setattr(BaseAutoML, "_perform_model_stacking", original_stacking)
    assert resumed._fit_level == "finished"
    assert any(m._is_stacked for m in resumed._models)
    names = [m.get_name() for m in resumed._models]
    assert len(names) == len(set(names))
    assert (directory / "optuna" / "optuna.json").exists()
    predict = "predict_proba" if classes else "predict"
    expected = getattr(resumed, predict)(X.copy())
    reloaded = AutoML(results_path=str(directory))
    np.testing.assert_array_equal(getattr(reloaded, predict)(X.copy()), expected)
