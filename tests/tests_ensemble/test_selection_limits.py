from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from supervised.ensemble import Ensemble


def fit_ensemble(monkeypatch, oofs, y, is_stacked):
    ensemble = Ensemble(
        optimize_metric="mse", ml_task="regression", is_stacked=is_stacked
    )
    ensemble.models_map = {
        name: SimpleNamespace(get_name=lambda name=name: name) for name in oofs
    }
    monkeypatch.setattr(ensemble, "get_additional_metrics", lambda: {})
    ensemble.fit(oofs, y)
    return ensemble


@pytest.mark.parametrize("is_stacked", [False, True])
def test_selection_stops_at_thirty_rounds_with_more_candidates(monkeypatch, is_stacked):
    # Every additional distinct model improves the mean. Without the round
    # limit, all 31 models would be selected and the ensemble would be exact.
    predictions = np.eye(31)
    oofs = {
        f"model_{i}": pd.DataFrame({"prediction": prediction})
        for i, prediction in enumerate(predictions)
    }
    y = np.full(31, 1.0 / 31)
    ensemble = fit_ensemble(monkeypatch, oofs, y, is_stacked)

    assert len(ensemble.best_algs) == 30
    assert len(ensemble.selected_models) == 30
    assert len(ensemble.models_map) == len(oofs) == 31
    assert ensemble.best_loss > 0
    expected = np.mean([oofs[name].values for name in ensemble.best_algs], axis=0)
    np.testing.assert_allclose(ensemble.total_best_sum, expected)


@pytest.mark.parametrize("is_stacked", [False, True])
def test_thirty_rounds_refine_weights_with_only_two_candidates(monkeypatch, is_stacked):
    oofs = {
        "zero": pd.DataFrame({"prediction": np.zeros(4)}),
        "one": pd.DataFrame({"prediction": np.ones(4)}),
    }
    y = np.full(4, 1.0 / 30)
    ensemble = fit_ensemble(monkeypatch, oofs, y, is_stacked)

    repeats = {
        selected["model"].get_name(): selected["repeat"]
        for selected in ensemble.selected_models
    }
    assert repeats == {"zero": 29, "one": 1}
    np.testing.assert_allclose(ensemble.total_best_sum.values.ravel(), y)
    assert ensemble.best_loss == pytest.approx(0)


@pytest.mark.parametrize("is_stacked", [False, True])
def test_selection_keeps_the_earliest_best_combination(monkeypatch, is_stacked):
    oofs = {
        "zero": pd.DataFrame({"prediction": np.zeros(4)}),
        "one": pd.DataFrame({"prediction": np.ones(4)}),
    }
    ensemble = fit_ensemble(monkeypatch, oofs, np.zeros(4), is_stacked)

    assert ensemble.best_algs == ["zero"]
    assert ensemble.selected_models[0]["repeat"] == 1
    np.testing.assert_array_equal(ensemble.total_best_sum, np.zeros((4, 1)))
