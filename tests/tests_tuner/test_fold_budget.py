import pytest

from supervised.exceptions import BudgetExceededError
from supervised.tuner.fold_budget import FoldBudget
from supervised.tuner.time_controller import TimeController


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("supervised.tuner.fold_budget.time.monotonic", lambda: now[0])
    monkeypatch.setattr("supervised.tuner.time_controller.time.time", lambda: now[0])
    return now


def test_first_completed_fold_can_extend_candidate_allocation(clock):
    budget = FoldBudget(900, allowance=120)
    clock[0] = 10
    budget.record("fold_0", 10, "patience")
    budget.check(15)
    assert budget.decisions[-1]["decision"] == "extend"
    assert budget.allowance == 188


def test_first_fold_too_expensive_is_not_trained(clock):
    budget = FoldBudget(100)
    budget.status = "not_trained"
    clock[0] = 20
    budget.record("fold_0", 20, "max_epochs")
    with pytest.raises(BudgetExceededError) as exc:
        budget.check(15)
    assert exc.value.details["completed_folds"] == 1
    assert budget.status == "not_trained"


def test_setup_is_not_multiplied_and_small_candidate_allowance_is_not_deadline(clock):
    budget = FoldBudget(100, allowance=1)
    clock[0] = 30  # 20 seconds of setup, then a 10-second fold
    budget.record("fold_0", 10, "patience")
    budget.check(3)
    assert budget.decisions[-1]["estimated_total"] == 64


def test_eighty_percent_can_use_global_finishing_reserve(clock):
    budget = FoldBudget(100, 110)
    for i in range(12):
        budget.record(str(i), 5, "patience")
    clock[0] = 85
    budget.check(15)  # 23 seconds remaining: exceeds normal 15, fits reserve 25
    assert budget.decisions[-1]["finishing_reserve"]
    clock[0] = 90
    with pytest.raises(BudgetExceededError):
        budget.check(15)


def test_reserve_is_unavailable_early_and_not_reset_per_candidate(clock):
    controller = TimeController(0, 100, None, ["default_algorithms", "ensemble"], ["RealMLP"])
    clock[0] = 80
    first = controller.fold_budget()
    assert first.available == 15 and first.finishing_available == 25
    first.record("fold_0", 5, "max_epochs")
    with pytest.raises(BudgetExceededError):
        first.check(5)  # 29 seconds required, not a fresh allowance
    clock[0] = 90
    second = controller.fold_budget()
    assert second.available == 5 and second.finishing_available == 15


def test_finished_folds_are_not_discarded_at_budget_boundary(clock):
    budget = FoldBudget(1)
    clock[0] = 10
    budget.record("fold_0", 10, "max_rounds")
    budget.check(1)


def test_pilot_deferral_survives_resume_and_is_stage_specific(clock):
    controller = TimeController(0, 1000, None, ["default_algorithms", "not_so_random"], ["RealMLP"])
    details = dict(phase="between_folds", completed_folds=1)
    for i in range(2):
        controller.record_attempt(str(i), "RealMLP", "default_algorithms", 10,
                                  "budget_exhausted", str(i), details)
    restored = TimeController.from_json(controller.to_json())
    assert restored.pilots_deferred("RealMLP", "default_algorithms")
    assert not restored.enough_time("RealMLP", "default_algorithms")
    assert not restored.pilots_deferred("RealMLP", "not_so_random")
    restored.record_attempt("success", "RealMLP", "default_algorithms", 10, "trained", "c")
    assert not restored.pilots_deferred("RealMLP", "default_algorithms")


def test_changed_configuration_is_not_rejected_using_algorithm_mean(clock):
    controller = TimeController(0, 100, None, ["default_algorithms"], ["LightGBM"])
    controller.record_attempt("expensive", "LightGBM", "default_algorithms", 1000, "trained", "a")
    assert controller.enough_time("LightGBM", "default_algorithms")


def test_only_consecutive_first_fold_rejections_count(clock):
    controller = TimeController(0, 1000, None, ["default_algorithms"], ["RealMLP"])
    for i, completed in enumerate([1, 3, 1]):
        controller.record_attempt(str(i), "RealMLP", "default_algorithms", 10,
            "budget_exhausted", str(i), dict(phase="between_folds", completed_folds=completed))
    assert not controller.pilots_deferred("RealMLP", "default_algorithms")


def test_resume_preserves_unlogged_wall_time_and_finishing_reserve(clock):
    controller = TimeController(0, 100, None, ["default_algorithms", "ensemble"], ["RealMLP"])
    clock[0] = 90
    saved = controller.to_json()
    clock[0] = 1000
    restored = TimeController.from_json(saved)
    budget = restored.fold_budget()
    assert budget.available == 5 and budget.finishing_available == 15
