import json
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.datasets import make_classification

from supervised import AutoML
from supervised.algorithms.xgboost import XgbAlgorithm
from supervised.exceptions import CandidateValidationError


def algorithm(**kwargs):
    return XgbAlgorithm(dict(objective="binary:logistic", eval_metric="auc",
                             ml_task="binary_classification", n_jobs=1,
                             max_rounds=20, **kwargs))


def test_impossible_fold_skipped_before_dmatrix(monkeypatch):
    m = algorithm(min_child_weight=13)
    monkeypatch.setattr("supervised.algorithms.xgboost.xgb.DMatrix",
                        lambda *a, **k: pytest.fail("Preflight must precede allocation"))
    with pytest.raises(CandidateValidationError) as exc:
        m.fit(np.zeros((100, 2)), np.tile([0, 1], 50))
    assert exc.value.details["phase"] == "before_fit"
    assert exc.value.details["hessian_upper_bound"] == 25


def test_boundary_and_subsampling_expectation_are_not_hard_limits():
    m = algorithm(min_child_weight=12.5, subsample=0.1)
    m._validate_split_capacity(np.tile([0, 1], 50), None)
    # Initial imbalance does not provide a bound on future boosting rounds.
    m._validate_split_capacity(np.r_[np.zeros(99), 1], None)


def test_sample_and_class_weights_are_respected():
    m = algorithm(min_child_weight=20)
    y = np.tile([0, 1], 50)
    with pytest.raises(CandidateValidationError):
        m._validate_split_capacity(y, np.ones(100))
    m._validate_split_capacity(y, np.full(100, 2.0))
    m.learner_params["scale_pos_weight"] = 3
    m._validate_split_capacity(y, np.ones(100))


@pytest.mark.parametrize("objective", ["reg:squarederror", "multi:softprob"])
def test_other_objectives_are_not_subject_to_logistic_bound(objective):
    m = algorithm(min_child_weight=1000)
    m.learner_params["objective"] = objective
    m._validate_split_capacity(np.arange(10), None)
    m._validate_trained_split_capacity()


def test_observed_no_split_case_is_skipped_after_training():
    # Full-fold bound is 250, but actual logistic Hessians are near 90,
    # and subsampling reduces them further: reproduces Good-customer's issue.
    rng = np.random.default_rng(17)
    X = rng.normal(size=(1000, 4))
    y = np.r_[np.ones(100), np.zeros(900)]
    m = algorithm(min_child_weight=50, subsample=0.9, seed=1239)
    m.early_stopping_rounds = 5
    with pytest.raises(CandidateValidationError) as exc:
        m.fit(X, y, X_validation=X[:200], y_validation=y[:200])
    assert exc.value.details["phase"] == "after_fit"
    assert exc.value.details["max_observed_root_hessian"] < 100
    assert exc.value.details["trained_trees"] == 6
    assert m.model is None


def test_a_constant_fit_with_sufficient_hessian_is_not_rejected():
    m = algorithm(min_child_weight=1)
    X = np.zeros((100, 2))
    m.fit(X, np.tile([0, 1], 50))
    assert m.model is not None


def test_later_splits_are_not_rejected_for_a_constant_best_prefix():
    m = algorithm(min_child_weight=50)
    m.model = Mock()
    m.model.get_score.return_value = {"f0": 1}
    m._validate_trained_split_capacity()
    m.model.get_dump.assert_not_called()


def test_usable_model_still_trains_and_predicts():
    X, y = make_classification(n_samples=100, n_features=5, random_state=3)
    m = algorithm(min_child_weight=1, max_depth=3)
    m.fit(X, y)
    assert m.model.get_score()
    assert np.unique(m.predict(X)).size > 1


def test_automl_records_candidate_skip_and_continues(tmp_path, monkeypatch):
    original = XgbAlgorithm.fit

    def impossible(self, *args, **kwargs):
        self.learner_params["min_child_weight"] = 1000
        return original(self, *args, **kwargs)

    monkeypatch.setattr(XgbAlgorithm, "fit", impossible)
    X, y = make_classification(n_samples=100, n_features=5, random_state=2)
    root = tmp_path / "automl"
    automl = AutoML(results_path=str(root), mode="Explain",
                    algorithms=["Decision Tree", "Xgboost"],
                    total_time_limit=30, train_ensemble=False, explain_level=0,
                    n_jobs=1, verbose=0)
    automl.fit(X, y)
    assert len(automl.predict(X[:3])) == 3
    records = json.loads((root / "candidate_skips.json").read_text())
    assert records and all(r["phase"] == "before_fit" for r in records)
    assert "candidate_skips.json" in (root / "README.md").read_text()
    assert not (root / "errors.md").exists()
    assert not (root / "resource_skips.json").exists()
    progress = json.loads((root / "progress.json").read_text())
    rejected = [p for ps in progress["all_params"].values() for p in ps
                if "candidate_skip" in p]
    assert rejected and all(p["status"] == "skipped" for p in rejected)
    assert all(m.get_type() != "Xgboost" for m in automl._models)
