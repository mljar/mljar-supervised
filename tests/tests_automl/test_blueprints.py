import json
import numpy as np
import pytest
import pandas as pd

from supervised import AutoML


def data(rows=100):
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(rows, 3)), columns=["a", "b", "c"])
    return X, X.a.to_numpy() + rng.normal(size=rows) * 0.1


def automl(path, **kwargs):
    params = dict(
        results_path=str(path),
        ml_task="regression",
        algorithms=["Baseline"],
        explain_level=0,
        train_ensemble=False,
        stack_models=False,
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        boost_on_errors=False,
        mix_encoding=False,
        start_random_models=1,
        hill_climbing_steps=0,
        n_jobs=2,
        verbose=0,
        total_time_limit=120,
        validation_strategy={"validation_type": "split", "train_ratio": 0.8},
    )
    params.update(kwargs)
    return AutoML(**params)


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize(
    "task", ["regression", "binary_classification", "multiclass_classification"]
)
def test_blueprints_only_at_high_explanation_level(tmp_path, level, task):
    X, y = data(80)
    if task != "regression":
        y = np.arange(len(X)) % (2 if task == "binary_classification" else 3)
    model = automl(tmp_path / "fit", explain_level=level, ml_task=task)
    model.fit(X, y)
    directory = tmp_path / "fit" / model._best_model.get_name()
    assert (directory / "blueprint.svg").exists() == (level == 2)
    assert ("## Model blueprint" in (directory / "README.md").read_text()) == (
        level == 2
    )
    if level == 2:
        assert getattr(model, "_training_store", None) is None
        graph = json.loads((directory / "blueprint.json").read_text())
        assert graph["task"] == task
        assert any(n["label"] == "Baseline" for n in graph["nodes"])
        assert "Model blueprints" in (tmp_path / "fit" / "README.md").read_text()
        assert "https://mljar.com" in model._md_to_html(
            str(directory / "README.md"), "sub", str(directory), directory.name
        )
        before = (directory / "blueprint.svg").read_bytes()
        loaded = AutoML(results_path=str(tmp_path / "fit"))
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
        loaded.select_and_save_best()
        assert (directory / "blueprint.svg").read_bytes() == before


def test_stacked_models_have_real_dependency_links(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "supervised.algorithms.algorithm.BaseAlgorithm.interpret", lambda *a, **kw: None
    )
    X, y = data(120)
    model = automl(
        tmp_path / "stack",
        algorithms=[
            "Baseline",
            "Linear",
            "Decision Tree",
            "Random Forest",
            "Extra Trees",
        ],
        explain_level=2,
        stack_models=True,
        train_ensemble=True,
        total_time_limit=300,
        validation_strategy={"validation_type": "kfold", "k_folds": 2, "shuffle": True},
    )
    model.fit(X, y)
    stacked = [m for m in model._models if m._is_stacked]
    assert stacked
    for member in stacked:
        directory = tmp_path / "stack" / member.get_name()
        graph = json.loads((directory / "blueprint.json").read_text())
        uses_stacked_input = (
            any(selected["model"]._is_stacked for selected in member.selected_models)
            if hasattr(member, "selected_models")
            else True
        )
        assert (
            any(n["label"] == "Stacked features" for n in graph["nodes"])
            == uses_stacked_input
        )
        assert ("out-of-fold" in " ".join(graph["notes"])) == uses_stacked_input
        html = model._md_to_html(
            str(directory / "README.md"), "sub", str(directory), directory.name
        )
        assert "<svg" in html
        assert 'href="../' not in html.split("<svg", 1)[1].split("</svg>", 1)[0]


def test_high_explanation_includes_all_members_and_ensemble(tmp_path, monkeypatch):
    from supervised.model_framework import ModelFramework

    # Explanation computation is outside the report-generation behavior under test.
    monkeypatch.setattr(
        "supervised.algorithms.algorithm.BaseAlgorithm.interpret", lambda *a, **kw: None
    )
    X, y = data(100)
    model = automl(
        tmp_path / "ensemble",
        algorithms=["Baseline", "Decision Tree"],
        explain_level=2,
        train_ensemble=True,
        validation_strategy={
            "validation_type": "kfold",
            "k_folds": 2,
            "shuffle": True,
            "stratify": False,
        },
    )
    model.fit(X, y)
    assert any(m.get_type() == "Ensemble" for m in model._models)
    for member in model._models:
        directory = tmp_path / "ensemble" / member.get_name()
        assert (directory / "blueprint.svg").exists()
        graph = json.loads((directory / "blueprint.json").read_text())
        if isinstance(member, ModelFramework):
            assert (
                sum(len(n["learners"]) for n in graph["nodes"] if "learners" in n) == 2
            )
        html = model._md_to_html(
            str(directory / "README.md"), "sub", str(directory), directory.name
        )
        assert "<svg" in html
        assert "blueprint.svg" not in html


def test_completed_model_blueprint_survives_interrupted_fit(tmp_path, monkeypatch):
    from supervised.base_automl import BaseAutoML

    original = BaseAutoML.train_model

    def interrupt(self, params):
        if params["learner"]["model_type"] == "Decision Tree":
            raise KeyboardInterrupt("interrupt later model")
        return original(self, params)

    monkeypatch.setattr(BaseAutoML, "train_model", interrupt)
    X, y = data(80)
    model = automl(
        tmp_path / "interrupted",
        algorithms=["Baseline", "Decision Tree"],
        explain_level=2,
    )
    with pytest.raises(KeyboardInterrupt):
        model.fit(X, y)
    assert getattr(model, "_training_store", None) is None
    assert (tmp_path / "interrupted" / "1_Baseline" / "blueprint.svg").exists()
    assert (
        "## Model blueprint"
        in (tmp_path / "interrupted" / "1_Baseline" / "README.md").read_text()
    )


def test_blueprint_error_does_not_abort_training(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("diagram unavailable")

    monkeypatch.setattr("supervised.utils.blueprint.save_blueprints", unavailable)
    X, y = data(80)
    model = automl(tmp_path / "warning", explain_level=2)
    with pytest.warns(RuntimeWarning, match="Could not generate model blueprints"):
        model.fit(X, y)
    assert len(model.predict(X)) == len(X)
