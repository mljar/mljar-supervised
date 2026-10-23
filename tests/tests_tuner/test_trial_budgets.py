import pytest

from supervised.tuner.time_controller import TimeController


def controller(monkeypatch):
    monkeypatch.setattr('supervised.tuner.time_controller.time.time', lambda: 0)
    tc = TimeController(0, 1000, None, ['default_algorithms', 'not_so_random'], ['RealMLP', 'CatBoost'])
    tc.set_active_algorithms('default_algorithms', {'RealMLP', 'CatBoost'})
    return tc


def test_initial_budget_is_for_whole_model_and_spending_is_cumulative(monkeypatch):
    tc = controller(monkeypatch)
    budget = tc.candidate_budget('RealMLP', 'default_algorithms', 'a')
    assert budget == pytest.approx(1000 * .3 / .65 / 2)
    tc.record_attempt('model', 'RealMLP', 'default_algorithms', budget, 'budget_exhausted', 'a')
    assert tc.candidate_budget('RealMLP', 'default_algorithms', 'b') == 0
    assert tc.candidate_budget('CatBoost', 'default_algorithms', 'c') == pytest.approx(budget)


def test_retry_needs_larger_budget_but_different_configuration_can_run(monkeypatch):
    tc = controller(monkeypatch)
    tc.record_attempt('failed', 'RealMLP', 'default_algorithms', 200, 'budget_exhausted', 'a')
    assert tc.candidate_budget('RealMLP', 'default_algorithms', 'a') == 0
    assert tc.candidate_budget('RealMLP', 'default_algorithms', 'smaller') > 0
    tc.set_active_algorithms('not_so_random', {'RealMLP', 'CatBoost'})
    assert tc.candidate_budget('RealMLP', 'not_so_random', 'a') > 200
    restored = TimeController.from_json(tc.to_json())
    restored.set_active_algorithms('default_algorithms', {'RealMLP', 'CatBoost'})
    assert restored.candidate_budget('RealMLP', 'default_algorithms', 'a') == 0


def test_success_accounted_once_and_failed_work_preserved_on_resume(monkeypatch):
    tc = controller(monkeypatch)
    tc.log_time('ok', 'RealMLP', 'default_algorithms', 10)
    tc.record_attempt('ok', 'RealMLP', 'default_algorithms', 12, 'trained', 'a')
    tc.record_attempt('bad', 'RealMLP', 'default_algorithms', 20, 'error', 'b')
    assert tc.already_spend() == 32
    assert len(tc._spend) == 2
    restored = TimeController.from_json(tc.to_json())
    assert restored.already_spend() == 32
    assert restored._start_time == -32


def test_remaining_wall_time_caps_budget_and_explicit_limit_wins(monkeypatch):
    tc = controller(monkeypatch)
    tc.candidate_budget('RealMLP', 'default_algorithms', 'a')
    monkeypatch.setattr('supervised.tuner.time_controller.time.time', lambda: 450)
    assert tc.candidate_budget('RealMLP', 'default_algorithms', 'a') == pytest.approx(1000*.3/.65-450)
    tc._model_time_limit = 42
    assert tc.candidate_budget('RealMLP', 'default_algorithms', 'a') == 42
    tc._model_time_limit = None
    tc._total_time_limit = None
    assert tc.candidate_budget('RealMLP', 'default_algorithms', 'a') is None


def test_unused_time_carries_into_next_stage(monkeypatch):
    tc = controller(monkeypatch)
    monkeypatch.setattr('supervised.tuner.time_controller.time.time', lambda: 100)
    tc.set_active_algorithms('not_so_random', {'RealMLP', 'CatBoost'})
    assert tc.candidate_budget('RealMLP', 'not_so_random', 'a') == 450
    assert tc._stage_budgets['not_so_random'] == 900


def test_stacking_budget_does_not_blacklist_all_configurations(monkeypatch):
    monkeypatch.setattr('supervised.tuner.time_controller.time.time', lambda: 0)
    tc = TimeController(0, 100, None, ['stack'], ['RealMLP'])
    tc.set_active_algorithms('stack', {'RealMLP'})
    assert tc.candidate_budget('RealMLP', 'stack', 'first') == 100
    tc.record_attempt('first', 'RealMLP', 'stack', 60, 'trained', 'first')
    # Compatibility-aware admission is separate: a cheaper network may fit.
    assert tc.candidate_budget('RealMLP', 'stack', 'second') == 40
