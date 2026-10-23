from types import SimpleNamespace

from supervised import AutoML
from supervised.prediction import prediction_dependencies
from supervised.ensemble import Ensemble
from supervised.tuner.stacking import select_stacking_models


def model(name, kind="Kumo", loss=-0.8, stacked=False, fair=True, valid=True, cost=1.0):
    return SimpleNamespace(
        get_name=lambda: name, get_type=lambda: kind, get_final_loss=lambda: loss,
        _is_stacked=stacked, is_fair=lambda: fair, is_valid=lambda: valid,
        _single_prediction_time=cost,
    )


def test_select_five_per_family_excluding_hidden_dependencies_and_invalid_models():
    candidates = [model(f"k{i}", loss=-0.9 + i * 0.01) for i in range(8)]
    candidates += [model("tree", "LightGBM", loss=-0.85),
                   model("ensemble", "Ensemble", loss=-0.99),
                   model("baseline", "Baseline"), model("x_RandomFeature"),
                   model("stack", stacked=True), model("bad", valid=False)]
    expected = ["k0", "k1", "k2", "k3", "k4", "tree"]
    for ordered in (candidates, candidates[::-1]):
        assert [m.get_name() for m in select_stacking_models(ordered)] == expected
    assert len(candidates) == 14


def test_fairness_and_ties_at_the_stacking_limit():
    candidates = [model(name) for name in ["z", "e", "d", "c", "b", "a"]]
    candidates.append(model("unfair", loss=-1, fair=False))
    selected = select_stacking_models(candidates, require_fair=True)
    assert [m.get_name() for m in selected] == ["a", "b", "c", "d", "e"]


def test_saved_stack_inputs_are_not_reselected():
    automl = AutoML()
    old_inputs = [model(f"k{i}") for i in range(10)]
    automl._stacked_models = old_inputs
    automl._models = []
    automl._perform_model_stacking()
    assert automl._stacked_models is old_inputs


def test_dependency_cost_counts_shared_inputs_and_repeated_weights_once():
    a, b = model("a", cost=2), model("b", cost=3)
    stack = model("stack", stacked=True, cost=6)
    base = Ensemble()
    base.selected_models = [{"model": a, "repeat": 3}, {"model": b, "repeat": 1}]
    final = Ensemble(is_stacked=True)
    final.selected_models = [{"model": base, "repeat": 1}, {"model": stack, "repeat": 1}]
    assert set(prediction_dependencies(final, [a, b])) == {"a", "b", "stack"}
    final._prediction_dependencies = {"a": {"a"}, "Ensemble": {"a", "b"},
                                      "stack": {"a", "b", "stack"}}
    final._dependency_times = {"a": 2, "b": 3, "stack": 1}
    assert final._prediction_cost(["a", "a"]) == 2
    assert final._prediction_cost(["Ensemble", "stack", "stack"]) == 6


def test_repeated_ensemble_member_does_not_consume_another_time_budget(monkeypatch):
    import numpy as np
    import pandas as pd

    ensemble = Ensemble(optimize_metric="mse", ml_task="regression",
                        max_single_prediction_time=2.1)
    ensemble.model_prediction_time = {"zero": 1., "one": 1., "expensive": 10.}
    ensemble.models_map = {n: model(n) for n in ensemble.model_prediction_time}
    monkeypatch.setattr(ensemble, "get_additional_metrics", lambda: {})
    oofs = {n: pd.DataFrame({"prediction": [v, v, v]})
            for n, v in [("zero", 0.), ("one", 1.), ("expensive", 4.)]}
    ensemble.fit(oofs, np.array([0.3, 0.3, 0.3]))
    assert ensemble.best_algs.count("zero") == 7
    assert ensemble.best_algs.count("one") == 3
    assert "expensive" not in ensemble.best_algs
    np.testing.assert_allclose(ensemble.total_best_sum, 0.3)


def test_oof_costs_use_exclusive_stack_timings_after_reload():
    import pandas as pd

    a, b, stacked = model("a", cost=2), model("b", cost=3), model("stack", stacked=True, cost=6)
    stacked._prediction_time_by_model = {"a": 2, "b": 3, "stack": 1}
    for candidate in [a, b, stacked]:
        candidate.is_fast_enough = lambda limit: True
        candidate.get_out_of_folds = lambda: pd.DataFrame({"prediction": [0., 1.], "target": [0., 1.]})
    ensemble = Ensemble(ml_task="regression", max_single_prediction_time=10)
    ensemble.get_oof_matrix([a, b, stacked], [a, b])
    assert ensemble._prediction_cost(["a", "b", "stack"]) == 6
