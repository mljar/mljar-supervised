import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from supervised.utils.importance import PermutationImportance


class PermutationImportanceTest(unittest.TestCase):
    def test_compute_and_plot(self):
        rows = 20
        X = np.random.rand(rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = np.random.randint(0, 2, rows)

        model = DecisionTreeClassifier(max_depth=1)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            PermutationImportance.compute_and_plot(
                model,
                X_validation=X,
                y_validation=y,
                model_file_path=tmpdir,
                learner_name="learner_test",
                metric_name=None,
                ml_task="binary_classification",
            )
            self.assertTrue(
                os.path.exists(os.path.join(tmpdir, "learner_test_importance.csv"))
            )


def test_loader_aligns_features_and_ignores_shap(tmp_path):
    from supervised.utils.importance import load_model_importance

    pd.DataFrame({"feature": ["b", "NA"], "mean_importance": [-2, 4]}).to_csv(
        tmp_path / "learner_0_importance.csv", index=False
    )
    pd.DataFrame({"feature": ["NA", "c"], "mean_importance": [2, 9]}).to_csv(
        tmp_path / "learner_1_importance.csv", index=False
    )
    pd.DataFrame({"feature": ["NA"], "mean_importance": [999]}).to_csv(
        tmp_path / "learner_0_shap_importance.csv", index=False
    )
    assert load_model_importance(tmp_path).to_dict() == {"b": -2, "NA": 3, "c": 9}


def test_invalid_importance_artifacts(tmp_path):
    from supervised.utils.importance import load_importance_file, load_model_importance

    assert load_model_importance(tmp_path) is None
    assert load_importance_file(tmp_path / "missing.csv") is None
    for text in (
        "",
        "feature,wrong\na,1\n",
        "feature,mean_importance\na,nan\n",
        "feature,mean_importance\na,inf\n",
        "feature,mean_importance\na,1\na,2\n",
        "feature,mean_importance\na,not-a-number\n",
        "feature,mean_importance\n",
    ):
        path = tmp_path / "learner_importance.csv"
        path.write_text(text)
        assert load_importance_file(path) is None


def test_evaluation_can_contain_only_one_training_class():
    X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5]})
    y = np.array([0, 0, 0, 1, 1, 1])
    model = DecisionTreeClassifier(max_depth=1, random_state=12).fit(X, y)
    importance = PermutationImportance.compute(
        model, X.iloc[:3], y[:3], ml_task="binary_classification", n_jobs=1
    )
    assert importance.mean_importance.tolist() == [0.0]


def test_subsample_is_reproducible(monkeypatch):
    from supervised.utils import importance

    X = pd.DataFrame(np.zeros((4000, 51)))
    y = pd.Series(np.arange(4000))
    indices = []

    def capture_sample(model, X_eval, y_eval, **kwargs):
        indices.append(X_eval.index.to_numpy())
        np.testing.assert_array_equal(X_eval.index, y_eval)
        return {"importances_mean": np.zeros(X_eval.shape[1])}

    monkeypatch.setattr(importance, "permutation_importance", capture_sample)
    for _ in range(2):
        PermutationImportance.compute(None, X, y, ml_task="regression", n_jobs=1)
    assert len(indices[0]) == 1000
    np.testing.assert_array_equal(indices[0], indices[1])
