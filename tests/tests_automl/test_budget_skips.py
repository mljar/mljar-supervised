import json

import pytest
from sklearn.datasets import make_classification

from supervised import AutoML
from supervised.exceptions import BudgetExceededError


def test_budget_stop_is_not_an_error_and_attempt_is_recorded(tmp_path, monkeypatch, capsys):
    train = AutoML.train_model
    def stop(self, params):
        if params['learner']['model_type'] == 'Random Forest':
            raise BudgetExceededError('deadline during validation', phase='validation', allowance=1)
        return train(self, params)
    monkeypatch.setattr(AutoML, 'train_model', stop)
    X, y = make_classification(n_samples=80, n_features=4, random_state=17)
    root = tmp_path / 'model'
    model = AutoML(results_path=str(root), mode='Explain', algorithms=['Decision Tree', 'Random Forest'],
                   total_time_limit=60, train_ensemble=False, explain_level=0, n_jobs=1)
    model.fit(X, y)
    state = json.loads((root/'progress.json').read_text())
    skips = json.loads((root/'budget_skips.json').read_text())
    assert skips and skips[0]['phase'] == 'validation'
    assert skips[0]['allowance'] == 1
    assert not (root/'errors.md').exists()
    assert 'budget_skips.json' in (root/'README.md').read_text()
    failed = [s for s in state['time_controller']['spend'] if s.get('outcome') == 'budget_exhausted']
    assert failed and failed[0]['train_time'] > 0
    assert 'There was an error' not in capsys.readouterr().out
    assert len(AutoML(results_path=str(root)).predict(X[:3])) == 3


def test_retry_identity_separates_stacking_and_validation():
    from copy import deepcopy
    p = dict(learner={'model_type': 'RealMLP', 'seed': 1}, preprocessing={}, validation_strategy={'k_folds': 5})
    model = AutoML()
    key = model._trial_configuration(p)
    q = deepcopy(p); q['learner']['seed'] = 2
    assert model._trial_configuration(q) == key
    q['is_stacked'] = True
    assert model._trial_configuration(q) != key
    q = deepcopy(p); q['validation_strategy']['k_folds'] = 10
    assert model._trial_configuration(q) != key


def test_candidate_stops_between_folds_without_publishing_partial_model(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace
    import supervised.model_framework as framework
    from supervised.algorithms.random_forest import RandomForestAlgorithm

    clock = [0.0]
    monkeypatch.setattr(framework, 'time', SimpleNamespace(time=time.time, monotonic=lambda: clock[0]))
    original_train = framework.ModelFramework.train
    original_fit = RandomForestAlgorithm.fit
    def train(self, *args):
        if self.get_type() == 'Random Forest':
            from supervised.tuner.fold_budget import FoldBudget
            self.training_budget = FoldBudget(10.0)
        return original_train(self, *args)
    def fit(self, *args, **kwargs):
        result = original_fit(self, *args, **kwargs)
        clock[0] += 6.0
        return result
    monkeypatch.setattr(framework.ModelFramework, 'train', train)
    monkeypatch.setattr(RandomForestAlgorithm, 'fit', fit)
    X, y = make_classification(n_samples=80, n_features=4, random_state=17)
    root = tmp_path/'folds'
    model = AutoML(results_path=str(root), mode='Explain', algorithms=['Decision Tree', 'Random Forest'],
                   total_time_limit=120, train_ensemble=False, explain_level=0, n_jobs=1,
                   validation_strategy={'validation_type':'kfold','k_folds':3,'shuffle':True,'stratify':True})
    model.fit(X, y)
    records = json.loads((root/'budget_skips.json').read_text())
    assert any(s.get('phase') == 'between_folds' and s['completed_folds'] == 1 for s in records)
    assert all(m.get_type() != 'Random Forest' for m in model._models)
    assert not (root/'errors.md').exists()



def test_pilot_deferral_happens_before_model_directory(tmp_path):
    from supervised.tuner.time_controller import TimeController
    model = AutoML()
    model._results_path = str(tmp_path / "automl")
    model._fit_level = "not_so_random"
    model._time_ctrl = TimeController(0, 1000, None, ["not_so_random"], ["RealMLP"])
    for i in range(2):
        model._time_ctrl.record_attempt(str(i), "RealMLP", "not_so_random", 10,
            "budget_exhausted", str(i), dict(phase="between_folds", completed_folds=1))
    with pytest.raises(BudgetExceededError, match="Two unaffordable") as exc:
        model.train_model(dict(name="third", learner=dict(model_type="RealMLP")))
    assert exc.value.details["phase"] == "pilot_admission"
    assert not (tmp_path / "automl").exists()


def test_two_failed_pilots_defer_changed_candidates_and_save_status(tmp_path, monkeypatch):
    from supervised.model_framework import ModelFramework
    original = ModelFramework.train
    attempts = []

    def train(self, *args):
        if self.get_type() == "Random Forest":
            attempts.append(self.get_name())
            raise BudgetExceededError("Pilot cannot finish", phase="between_folds",
                                      completed_folds=1, expected_folds=3)
        return original(self, *args)

    monkeypatch.setattr(ModelFramework, "train", train)
    X, y = make_classification(n_samples=80, n_features=4, random_state=17)
    root = tmp_path / "pilots"
    automl = AutoML(results_path=str(root), mode="Compete",
                   algorithms=["Decision Tree", "Random Forest"], total_time_limit=120,
                   start_random_models=4, hill_climbing_steps=0, train_ensemble=False,
                   stack_models=False, explain_level=0, n_jobs=1,
                   golden_features=False, feature_discovery=False, kmeans_features=False,
                   features_selection=False, mix_encoding=False, boost_on_errors=False,
                   validation_strategy=dict(validation_type="kfold", k_folds=3,
                                            stratify=True, shuffle=True))
    automl.fit(X, y)
    assert len(attempts) == 3  # One default pilot; two in not_so_random.
    records = json.loads((root / "budget_skips.json").read_text())
    assert any(r["phase"] == "pilot_admission" for r in records)
    state = json.loads((root / "progress.json").read_text())
    assert state["time_controller"]["pilot_failures"]["not_so_random"]["Random Forest"] == 2
    assert all(m.get_type() != "Random Forest" for m in automl._models)
    assert '"status": "not_trained"' in (root / "progress.json").read_text()
    assert '"status": "not_started"' in (root / "progress.json").read_text()


@pytest.mark.parametrize("algorithm,shared", [("RealMLP", True), ("RealMLP", False), ("TabM", False)])
def test_neural_timeout_checkpoint_is_not_published(tmp_path, monkeypatch, algorithm, shared):
    from supervised.algorithms import realmlp, tabm
    from supervised.algorithms.realmlp import RealMLPAlgorithm
    from supervised.algorithms.tabm import TabMAlgorithm

    # Fit is simulated here; exercise exclusion without requiring optional backends.
    monkeypatch.setattr(realmlp, "check_dependencies", lambda: None)
    monkeypatch.setattr(tabm, "check_dependencies", lambda: None)
    attempts = []

    def interrupted(self, *args, **kwargs):
        assert args[-1] is None
        attempts.append(self)
        self.training_info = dict(stop_reason="time_limit", best_epoch=2)

    cls = RealMLPAlgorithm if algorithm == "RealMLP" else TabMAlgorithm
    monkeypatch.setattr(cls, "fit", interrupted)
    X, y = make_classification(n_samples=80, n_features=4, random_state=17)
    root = tmp_path / algorithm
    automl = AutoML(results_path=str(root), mode="Compete",
                   algorithms=["Decision Tree", algorithm], total_time_limit=120,
                   custom_params=({"realmlp": {"shared_checkpoint_selection": False}}
                                  if algorithm == "RealMLP" and not shared else {}),
                   train_ensemble=True, explain_level=0, n_jobs=1,
                   start_random_models=1, hill_climbing_steps=0, stack_models=False,
                   golden_features=False, feature_discovery=False, kmeans_features=False,
                   features_selection=False, mix_encoding=False, boost_on_errors=False,
                   validation_strategy=dict(validation_type="kfold", k_folds=2,
                                            stratify=True, shuffle=True))
    automl.fit(X, y)
    assert attempts
    assert all(m.get_type() != algorithm for m in automl._models)
    records = json.loads((root / "budget_skips.json").read_text())
    assert any(r.get("stop_reason") == "time_limit" for r in records)
    assert not (root / "errors.md").exists()
    assert not list(root.rglob(".realmlp-*"))
