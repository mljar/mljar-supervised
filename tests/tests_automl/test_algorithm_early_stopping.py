import json

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from supervised import AutoML
from supervised.model_framework import ModelFramework


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_generated_data_fit_skips_queued_trials_and_preserves_prediction(tmp_path, monkeypatch, capsys, task):
    # Train real estimators, but inject a controlled validation history so this
    # test verifies scheduling without relying on a lucky dataset/metric gap.
    original_loss = ModelFramework.get_final_loss
    def loss(self):
        if self.params.get("data_type", "original") == "original":
            if self.get_type() == "Nearest Neighbors":
                return 6.0
            if self.get_type() == "Decision Tree":
                return 10.0 if self.get_name() == "1_DecisionTree" else 1.0
        return original_loss(self)
    monkeypatch.setattr(ModelFramework, "get_final_loss", loss)
    factory = make_classification if task == "classification" else make_regression
    X, y = factory(n_samples=100, n_features=5, random_state=13)
    root = tmp_path / task
    options = dict(
        results_path=str(root), mode="Compete", algorithms=["Decision Tree", "Nearest Neighbors"],
        eval_metric="logloss" if task == "classification" else "rmse",
        total_time_limit=120, validation_strategy={"validation_type": "kfold", "k_folds": 2,
                                                  "shuffle": True, "stratify": task == "classification"},
        start_random_models=10, hill_climbing_steps=1, train_ensemble=True, stack_models=False,
        golden_features=False, feature_discovery=False, features_selection=False,
        kmeans_features=False, mix_encoding=False, boost_on_errors=False,
        explain_level=0, n_jobs=1, verbose=1,
    )
    automl = AutoML(**options)
    if task == "regression":
        save = AutoML.save_progress
        def interrupt(self, step=None, generated_params=None):
            save(self, step, generated_params)
            if self._algorithm_early_stopping.skipped:
                raise KeyboardInterrupt("after first algorithm pause")
        monkeypatch.setattr(AutoML, "save_progress", interrupt)
        with pytest.raises(KeyboardInterrupt, match="first algorithm pause"):
            automl.fit(X, y)
        reference = automl._algorithm_early_stopping.reference.copy()
        already_trained = {m.get_name() for m in automl._models}
        monkeypatch.setattr(AutoML, "save_progress", save)
        train = AutoML.train_model
        def check_no_retraining(self, params):
            assert params["name"] not in already_trained
            return train(self, params)
        monkeypatch.setattr(AutoML, "train_model", check_no_retraining)
        automl = AutoML(**options)
        automl.fit(X, y)
        assert automl._algorithm_early_stopping.reference == reference
    else:
        automl.fit(X, y)
    state = json.loads((root / "progress.json").read_text())
    policy = state["algorithm_early_stopping"]
    assert policy["reference"] == {"model": "1_DecisionTree", "loss": 10.0}
    assert len(policy["algorithms"]["Nearest Neighbors"]["configurations"]) == 3
    skipped = [p for ps in state["all_params"].values() for p in ps if "algorithm_early_stopping" in p]
    assert skipped and all(p["status"] == "skipped" and p["final_loss"] is None for p in skipped)
    assert all(not (root / p["name"]).exists() for p in skipped)
    assert any(m.get_type() == "Nearest Neighbors" for m in automl._models)
    assert any(m.get_type() == "Ensemble" for m in automl._models)
    assert "algorithm_early_stopping.json" in (root / "README.md").read_text()
    assert json.loads((root / "algorithm_early_stopping.json").read_text()) == policy
    assert capsys.readouterr().out.count("Algorithm early stopping: Nearest Neighbors") == 1
    expected = automl.predict(X[:7])
    loaded = AutoML(results_path=str(root))
    np.testing.assert_allclose(loaded.predict(X[:7]), expected)


def test_legacy_progress_does_not_enable_new_policy(tmp_path):
    (tmp_path / "progress.json").write_text(json.dumps(dict(fit_level="not_so_random", all_params={})))
    automl = AutoML(mode="Compete")
    automl._results_path = str(tmp_path)
    automl._eval_metric = "auc"
    automl.load_progress()
    assert not automl._algorithm_early_stopping.enabled
