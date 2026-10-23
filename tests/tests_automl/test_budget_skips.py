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
            self.params['candidate_time_limit'] = 10.0
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


def test_stacking_cost_admission_happens_before_model_directory(tmp_path, monkeypatch):
    from supervised.tuner.algorithm_early_stopping import AlgorithmEarlyStopping
    from supervised.tuner.time_controller import TimeController
    monkeypatch.setattr('supervised.tuner.time_controller.time.time', lambda: 0)
    model = AutoML()
    model._results_path = str(tmp_path/'automl')
    model._fit_level = 'stack'
    model._total_time_limit = 100
    model._model_time_limit = None
    model._algorithm_early_stopping = AlgorithmEarlyStopping('auc')
    model._time_ctrl = TimeController(0, 100, None, ['stack'], ['RealMLP'])
    p = dict(name='second', learner={'model_type':'RealMLP','hidden_width':64,'n_hidden_layers':2},
             preprocessing={}, validation_strategy={'k_folds':10}, is_stacked=True)
    model._time_ctrl.stacking_cost.observe('first', model._stacking_descriptor(p), 12, 1, 10)
    with pytest.raises(BudgetExceededError) as exc:
        model.train_model(p)
    assert exc.value.details['source_attempt'] == 'first'
    assert exc.value.details['estimated_total'] == 120
    assert not (tmp_path/'automl').exists()
    assert len(model._time_ctrl.stacking_cost.observations) == 1


def test_failed_stacking_probe_informs_next_trial_and_is_saved(tmp_path, monkeypatch):
    from copy import deepcopy
    from supervised.model_framework import ModelFramework
    from supervised.tuner.mljar_tuner import MljarTuner
    train = ModelFramework.train
    generate = MljarTuner.get_params_stack_models
    probes = []
    def candidates(self, models):
        generated = generate(self, models)
        assert generated
        first = deepcopy(generated[0]); first['name'] = '1001_StackProbe'
        second = deepcopy(first); second['name'] = '1002_StackProbe'
        return [first, second]
    def fail_stacked(self, *args):
        if self.params.get('is_stacked'):
            probes.append(self.get_name())
            raise BudgetExceededError('Remaining folds cannot finish', phase='between_folds',
                                      elapsed=100, completed_folds=1, expected_folds=3,
                                      dimensions={'train_rows':60,'validation_rows':30,'features':9})
        return train(self, *args)
    monkeypatch.setattr(MljarTuner, 'get_params_stack_models', candidates)
    monkeypatch.setattr(ModelFramework, 'train', fail_stacked)
    X, y = make_classification(n_samples=90, n_features=4, random_state=17)
    root = tmp_path/'stacking'
    model = AutoML(results_path=str(root), mode='Compete',
                   algorithms=['Decision Tree','Random Forest','Extra Trees'],
                   total_time_limit=120, start_random_models=3, hill_climbing_steps=0,
                   train_ensemble=False, stack_models=True, explain_level=0, n_jobs=1,
                   golden_features=False, feature_discovery=False, kmeans_features=False,
                   features_selection=False, mix_encoding=False, boost_on_errors=False,
                   validation_strategy={'validation_type':'kfold','k_folds':3,'shuffle':True,'stratify':True})
    model.fit(X, y)
    assert probes == ['1001_StackProbe']
    state = json.loads((root/'progress.json').read_text())
    observations = state['time_controller']['stacking_cost_observations']
    assert len(observations) == 1 and observations[0]['estimated_total'] == 300
    skips = json.loads((root/'budget_skips.json').read_text())
    second = next(s for s in skips if s['name'] == '1002_StackProbe')
    assert second['source_attempt'] == '1001_StackProbe'
    assert second['phase'] == 'stacking_admission'
    assert not (root/'1002_StackProbe').exists()
    assert all(not m._is_stacked for m in model._models)
    assert not (root/'errors.md').exists()
