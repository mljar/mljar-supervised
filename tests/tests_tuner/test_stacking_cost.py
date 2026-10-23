import copy

import pytest

from supervised.tuner.stacking_cost import StackingCostHistory
from supervised.tuner.time_controller import TimeController


def params(depth=2, width=64, **settings):
    return dict(learner=dict(model_type='RealMLP', n_hidden_layers=depth,
                            hidden_width=width, device='cuda', batch_size=256,
                            **settings), preprocessing={}, is_stacked=True,
                validation_strategy={'k_folds': 10, 'shuffle': True})


def descriptor(p=None, inputs=None):
    return StackingCostHistory.descriptor(p or params(), inputs or {'rows': 1000, 'cols': 20})


def test_probe_rejects_equivalent_and_larger_but_admits_cheaper_or_more_time():
    history = StackingCostHistory()
    history.observe('first', descriptor(), 12, 1, 10, {'features': 20})
    assert history.decision(descriptor(), 90)['estimated_total'] == 120
    assert history.decision(descriptor(params(3, 128)), 90)['source_attempt'] == 'first'
    assert history.decision(descriptor(params(2, 32)), 90) is None
    assert history.decision(descriptor(), 120) is None
    assert history.decision(descriptor(params(3, 32)), 90) is None  # incomparable
    history.observe('cheaper', descriptor(params(2, 32)), 10, 1, 10)
    assert history.decision(descriptor(params(3, 32)), 80)['source_attempt'] == 'cheaper'


@pytest.mark.parametrize('change', ['device', 'batch_size', 'algorithm', 'validation', 'preprocessing', 'inputs'])
def test_incompatible_observations_are_not_reused(change):
    history = StackingCostHistory()
    history.observe('first', descriptor(), 12, 1, 10)
    p = params(); inputs = {'rows': 1000, 'cols': 20}
    if change == 'device': p['learner']['device'] = 'cpu'
    if change == 'batch_size': p['learner']['batch_size'] = 512
    if change == 'algorithm': p['learner']['model_type'] = 'TabM'
    if change == 'validation': p['validation_strategy']['k_folds'] = 5
    if change == 'preprocessing': p['preprocessing']['drop_features'] = ['a']
    if change == 'inputs': inputs['cols'] = 21
    assert history.decision(descriptor(p, inputs), 1) is None


def test_hyperparameter_only_changes_reuse_cost_and_success_can_improve_estimate():
    history = StackingCostHistory()
    history.observe('slow', descriptor(), 12, 1, 10)
    assert history.decision(descriptor(params(lr=.1, p_drop=.3, wd=.01, seed=9)), 90)
    history.observe('fast', descriptor(), 50, 10, 10)
    assert history.decision(descriptor(), 90) is None
    assert history.decision(descriptor(), 40)['source_attempt'] == 'fast'


def test_history_roundtrip_and_no_observation_for_untrained_or_admission_skip():
    tc = TimeController(0, 100, None, ['stack'], ['RealMLP'])
    tc.stacking_cost.observe('first', descriptor(), 12, 1, 10)
    tc.stacking_cost.observe('admission', descriptor(), .01, 0, 10)
    tc.stacking_cost.observe('invalid', descriptor(), float('nan'), 1, 10)
    restored = TimeController.from_json(tc.to_json())
    assert len(restored.stacking_cost.observations) == 1
    assert restored.stacking_cost.decision(descriptor(), 90)['source_attempt'] == 'first'
    legacy = tc.to_json(); legacy.pop('stacking_cost_observations')
    assert TimeController.from_json(legacy).stacking_cost.observations == []


def test_other_algorithms_require_equivalent_settings():
    history = StackingCostHistory()
    p = params(); p['learner']['model_type'] = 'TabM'
    history.observe('tabm', descriptor(p), 12, 1, 10)
    assert history.decision(descriptor(p), 90)
    q = copy.deepcopy(p); q['learner']['hidden_width'] = 128
    assert history.decision(descriptor(q), 90) is None
