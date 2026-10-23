import json
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from supervised import AutoML
from supervised.exceptions import AutoMLException
from supervised.utils.importance import PermutationImportance
from supervised.utils.report_structured import _compute_feature_importance_summary


def write_model(root, name, vectors):
    directory = root / name
    directory.mkdir()
    learners = []
    for i, vector in enumerate(vectors):
        learner = f"learner_fold_{i}"
        learners.append({"name": learner})
        if vector is not None:
            pd.DataFrame(
                list(vector.items()), columns=["feature", "mean_importance"]
            ).to_csv(directory / f"{learner}_importance.csv", index=False)
    (directory / "framework.json").write_text(json.dumps({"learners": learners}))
    return directory


def saved_automl(root, names, best=None):
    automl = AutoML()
    automl._results_path = str(root)
    automl._model_subpaths = names
    automl._best_model = SimpleNamespace(get_name=lambda: best or names[0])
    automl._models = [automl._best_model]
    return automl


def values(frame):
    return frame.set_index("feature")["importance"].to_dict()


def test_saved_selection_aggregation_normalization_and_report_agree(tmp_path):
    directory = write_model(
        tmp_path, "Tree", [{"a": 4, "b": -2}, {"b": 0, "a": 2, "c": 6}]
    )
    write_model(tmp_path, "Linear", [{"a": 10, "b": 10}])
    # Neither SHAP nor a stale learner file should enter the public result.
    for filename in ("learner_fold_0_shap_importance.csv", "stale_importance.csv"):
        pd.DataFrame({"feature": ["a"], "mean_importance": [999]}).to_csv(
            directory / filename, index=False
        )
    automl = saved_automl(tmp_path, ["Tree", "Linear"])
    with patch.object(
        PermutationImportance, "compute", side_effect=AssertionError("must reuse")
    ), patch(
        "supervised.utils.feature_importance.ModelFramework.load",
        side_effect=AssertionError("must not load"),
    ):
        best = automl.get_feature_importance()
        assert best.columns.tolist() == ["model", "feature", "importance"]
        assert best.feature.tolist() == ["c", "a", "b"]
        assert values(best) == {"c": 6, "a": 3, "b": -1}
        pd.testing.assert_frame_equal(best, automl.get_feature_importance(model="Tree"))
        all_models = automl.get_feature_importance(model="all")
        assert all_models.model.unique().tolist() == ["Tree", "Linear"]
        normalized = automl.get_feature_importance(model="all", kind="normalized")
        assert values(normalized[normalized.model == "Tree"]) == {
            "c": 1,
            "a": 4 / 7,
            "b": 0,
        }
        assert (normalized[normalized.model == "Linear"].importance == 0).all()
    (directory / "stale_importance.csv").unlink()
    summary = _compute_feature_importance_summary(str(directory))
    assert {row["feature"]: row["importance"] for row in summary["top"]} == values(best)


@pytest.mark.parametrize("vector", [{"a": 0, "b": 0}, {"a": -2, "b": -2}, {"a": 3}])
def test_constant_normalization(tmp_path, vector):
    write_model(tmp_path, "Tree", [vector])
    result = saved_automl(tmp_path, ["Tree"]).get_feature_importance(kind="normalized")
    assert (result.importance == 0).all()


def test_missing_and_partial_importance_warns(tmp_path):
    directory = write_model(tmp_path, "Partial", [{"a": 2}, None])
    write_model(tmp_path, "Missing", [None])
    # A malformed file is treated like a missing learner, not a zero value.
    (directory / "learner_fold_1_importance.csv").write_text(
        "feature,wrong_column\na,999\n"
    )
    automl = saved_automl(tmp_path, ["Partial", "Missing"])
    with pytest.warns(UserWarning, match="provide X and y") as recorded:
        result = automl.get_feature_importance(model="all")
    assert len(recorded) == 2
    assert result.to_dict("records") == [
        {"model": "Partial", "feature": "a", "importance": 2.0}
    ]
    with pytest.warns(UserWarning, match="provide X and y"):
        empty = automl.get_feature_importance(model="Missing")
    assert empty.empty
    assert empty.columns.tolist() == ["model", "feature", "importance"]
    assert empty.importance.dtype == float


def test_ensemble_weighted_summary_and_missing_member(tmp_path):
    write_model(tmp_path, "Tree", [{"a": 4, "b": -2}])
    write_model(tmp_path, "Linear", [{"a": 2, "c": 8}])
    directory = tmp_path / "Ensemble"
    directory.mkdir()
    (directory / "ensemble.json").write_text(
        json.dumps(
            {
                "selected_models": [
                    {"model": "Tree", "repeat": 3},
                    {"model": "Linear", "repeat": 1},
                ]
            }
        )
    )
    automl = saved_automl(tmp_path, ["Tree", "Linear", "Ensemble"], best="Ensemble")
    assert values(automl.get_feature_importance()) == {"a": 3.5, "b": -1.5, "c": 2}
    assert values(automl.get_feature_importance(kind="normalized")) == {
        "a": 1,
        "b": 0,
        "c": 0.7,
    }
    (tmp_path / "Tree" / "learner_fold_0_importance.csv").unlink()
    with pytest.warns(UserWarning) as recorded:
        assert automl.get_feature_importance().empty
    assert any("member's importance is unavailable" in str(w.message) for w in recorded)


def test_argument_errors_and_unfitted_does_not_create_results(tmp_path):
    automl = AutoML(results_path=str(tmp_path / "unused"))
    with pytest.raises(AutoMLException, match="not been fitted"):
        automl.get_feature_importance()
    assert not (tmp_path / "unused").exists()
    write_model(tmp_path, "Tree", [{"a": 1}])
    automl = saved_automl(tmp_path, ["Tree"])
    with pytest.raises(AutoMLException, match="Unknown model"):
        automl.get_feature_importance(model="typo")
    for arguments in ({"model": []}, {"kind": "shap"}, {"X": [[1]]}, {"y": [1]}):
        with pytest.raises(ValueError):
            automl.get_feature_importance(**arguments)


@pytest.fixture(
    params=["binary_classification", "multiclass_classification", "regression"]
)
def fitted_model(tmp_path, request):
    task = request.param
    if task == "regression":
        X, y = make_regression(
            n_samples=100, n_features=4, n_informative=2, random_state=12
        )
    else:
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            n_classes=3 if task.startswith("multi") else 2,
            random_state=12,
        )
        y = np.array([f"class_{value}" for value in y])
    X = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    # Exercise fitted categorical preprocessing as well as non-default indices.
    X["category"] = np.where(X.a > 0, "positive", "negative")
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    root = tmp_path / "results"
    automl = AutoML(
        results_path=str(root),
        ml_task=task,
        algorithms=["Decision Tree"],
        total_time_limit=60,
        explain_level=0,
        train_ensemble=False,
        stack_models=False,
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        start_random_models=1,
        hill_climbing_steps=0,
        validation_strategy={
            "validation_type": "kfold",
            "k_folds": 2,
            "shuffle": True,
            "stratify": task != "regression",
        },
        n_jobs=1,
        verbose=0,
    )
    automl.fit(X_train.copy(), y_train.copy())
    return automl, root, X_eval, y_eval


def test_compute_cache_partial_repair_and_relocated_reload(fitted_model, tmp_path):
    automl, root, X, y = fitted_model
    original = X.copy(deep=True)
    assert not list(root.rglob("*_importance.csv"))
    before = automl.predict(X.copy())
    with patch.object(
        PermutationImportance, "compute", wraps=PermutationImportance.compute
    ) as compute:
        result = automl.get_feature_importance(X=X, y=y)
        assert compute.call_count == 2
    assert not result.empty
    assert np.isfinite(result.importance).all()
    assert any("category" in feature for feature in result.feature)
    pd.testing.assert_frame_equal(X, original)
    np.testing.assert_array_equal(automl.predict(X.copy()), before)
    paths = sorted(root.rglob("*_importance.csv"))
    assert len(paths) == 2
    # Repair only the missing fold; leave the other artifact byte-for-byte intact.
    untouched = paths[0].read_bytes()
    paths[1].unlink()
    with patch.object(
        PermutationImportance, "compute", wraps=PermutationImportance.compute
    ) as compute:
        repaired = automl.get_feature_importance(X=X, y=y)
        assert compute.call_count == 1
    pd.testing.assert_frame_equal(result, repaired)
    assert paths[0].read_bytes() == untouched
    relocated = tmp_path / "relocated"
    shutil.move(str(root), relocated)
    restored = AutoML(results_path=str(relocated))
    with patch.object(
        PermutationImportance, "compute", side_effect=AssertionError("must reuse")
    ):
        pd.testing.assert_frame_equal(result, restored.get_feature_importance())
    for path in relocated.rglob("*_importance.csv"):
        path.unlink()
    recomputed = restored.get_feature_importance(X=X, y=y)
    pd.testing.assert_frame_equal(result, recomputed)
    np.testing.assert_array_equal(restored.predict(X.copy()), before)


def test_invalid_evaluation_data_raises(fitted_model):
    automl, _, X, y = fitted_model
    with pytest.raises(ValueError, match="inconsistent numbers"):
        automl.get_feature_importance(X=X, y=y[:-1])
    with pytest.raises(ValueError, match="Missing columns"):
        automl.get_feature_importance(X=X.drop(columns="a"), y=y)


def test_all_after_reload_includes_models_not_loaded_for_prediction(tmp_path):
    X, y = make_classification(n_samples=80, n_features=4, random_state=7)
    root = tmp_path / "models"
    automl = AutoML(
        results_path=str(root),
        algorithms=["Baseline", "Decision Tree"],
        total_time_limit=60,
        explain_level=0,
        train_ensemble=False,
        stack_models=False,
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        start_random_models=1,
        hill_climbing_steps=0,
        validation_strategy={
            "validation_type": "split",
            "train_ratio": 0.75,
            "shuffle": True,
            "stratify": True,
        },
        n_jobs=1,
        verbose=0,
    )
    automl.fit(X, y)
    assert len(automl._model_subpaths) == 2
    restored = AutoML(results_path=str(root))
    restored.predict(X[:2])
    assert len(restored._models) == 1
    loaded_names = [model.get_name() for model in restored._models]
    result = restored.get_feature_importance(model="all", X=X, y=y)
    assert set(result.model) == set(automl._model_subpaths)
    assert [model.get_name() for model in restored._models] == loaded_names
    baseline = result[result.model.str.contains("Baseline")]
    assert (baseline.importance == 0).all()
    with patch.object(
        PermutationImportance, "compute", side_effect=AssertionError("must reuse")
    ):
        pd.testing.assert_frame_equal(
            result, AutoML(results_path=str(root)).get_feature_importance(model="all")
        )


def test_compute_failure_is_reported_without_empty_artifact(fitted_model):
    automl, root, X, y = fitted_model
    with patch.object(
        PermutationImportance,
        "compute",
        side_effect=RuntimeError("test computation failure"),
    ), pytest.warns(UserWarning, match="test computation failure"):
        result = automl.get_feature_importance(X=X, y=y)
    assert result.empty
    assert not list(root.rglob("*_importance.csv"))


def test_knn_evaluation_limit(tmp_path, monkeypatch):
    from supervised.algorithms import knn

    X, y = make_classification(n_samples=80, n_features=4, random_state=7)
    root = tmp_path / "knn"
    automl = AutoML(
        results_path=str(root),
        algorithms=["Nearest Neighbors"],
        explain_level=0,
        train_ensemble=False,
        total_time_limit=60,
        stack_models=False,
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        start_random_models=1,
        hill_climbing_steps=0,
        n_jobs=1,
        verbose=0,
        validation_strategy={
            "validation_type": "split",
            "train_ratio": 0.75,
            "shuffle": True,
            "stratify": True,
        },
    )
    automl.fit(X, y)
    monkeypatch.setattr(knn, "KNN_INTERPRET_ROWS_LIMIT", 10)
    with pytest.warns(UserWarning, match="interpretation limit"):
        assert automl.get_feature_importance(X=X[:11], y=y[:11]).empty
    assert not list(root.rglob("*_importance.csv"))
    assert not automl.get_feature_importance(X=X[:10], y=y[:10]).empty
