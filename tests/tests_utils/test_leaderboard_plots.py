import io
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from supervised import AutoML
from supervised.utils.leaderboard_plots import LeaderboardPlots, markers


def plot_snapshots(ldb, tmp_path, monkeypatch, fairness_threshold=None):
    snapshots = {}
    savefig = plt.savefig

    def capture(path, *args, **kwargs):
        ax = plt.gca()
        snapshots[Path(path).name] = {
            "series": {
                line.get_label(): {
                    "x": list(line.get_xdata()),
                    "y": list(line.get_ydata()),
                    "marker": line.get_marker(),
                    "color": line.get_color(),
                }
                for line in ax.lines
                if not line.get_label().startswith("_")
            },
            "categories": [tick.get_text() for tick in ax.get_xticklabels()],
        }
        savefig(path, *args, **kwargs)

    monkeypatch.setattr(plt, "savefig", capture)
    report = io.StringIO()
    before = ldb.copy(deep=True)
    LeaderboardPlots.compute(ldb, tmp_path, report, fairness_threshold)
    pd.testing.assert_frame_equal(ldb, before)
    for filename in snapshots:
        assert (tmp_path / filename).stat().st_size > 0
        assert filename in report.getvalue()
    assert plt.get_fignums() == []
    return snapshots


def test_all_algorithms_and_ensembles_separate_stacked_variants(tmp_path, monkeypatch):
    types = list(markers)
    ldb = pd.DataFrame(
        {
            "model_type": types * 2,
            "is_stacked": [False] * len(types) + [True] * len(types),
            "metric_type": "logloss",
            "metric_value": np.arange(2 * len(types)) / 100,
            "fairness_metric": "demographic_parity_ratio",
            "fairness_group": 0.9,
        },
        # Plot iterations must use row positions even with duplicate index labels.
        index=[7] * (2 * len(types)),
    )
    plots = plot_snapshots(ldb, tmp_path, monkeypatch, fairness_threshold=0.8)
    labels = set(types) | {f"{name} (stacked)" for name in types}
    performance = plots[LeaderboardPlots.performance_fname]["series"]
    fairness = plots["performance_vs_fairness_group.png"]["series"]
    assert set(performance) == set(fairness) == labels
    assert (
        set(plots[LeaderboardPlots.performance_boxplot_fname]["categories"]) == labels
    )
    for i, name in enumerate(types):
        base, stacked = performance[name], performance[f"{name} (stacked)"]
        assert base["color"] == stacked["color"] == markers[name]["color"]
        assert base["marker"] != stacked["marker"]
        assert base["x"] == [i]
        assert stacked["x"] == [i + len(types)]
        assert base["y"] == [ldb.metric_value.iloc[i]]
        assert stacked["y"] == [ldb.metric_value.iloc[i + len(types)]]
        assert fairness[f"{name} (stacked)"]["marker"] == stacked["marker"]


@pytest.mark.parametrize("is_stacked", [None, False, True])
def test_legacy_and_single_status_leaderboards(tmp_path, monkeypatch, is_stacked):
    ldb = pd.DataFrame(
        {
            "model_type": ["CatBoost", "CatBoost"],
            "metric_type": "auc",
            "metric_value": [0.7, 0.8],
        }
    )
    if is_stacked is not None:
        ldb["is_stacked"] = is_stacked
    plots = plot_snapshots(ldb, tmp_path, monkeypatch)
    label = "CatBoost (stacked)" if is_stacked else "CatBoost"
    assert set(plots[LeaderboardPlots.performance_fname]["series"]) == {label}
    assert plots[LeaderboardPlots.performance_boxplot_fname]["categories"] == [label]


def test_leaderboard_exports_stacking_metadata_without_changing_types():
    automl = AutoML()
    automl._eval_metric = "logloss"
    automl._models = [
        SimpleNamespace(
            _is_stacked=stacked,
            get_name=lambda name=name: name,
            get_type=lambda model_type=model_type: model_type,
            get_final_loss=lambda: 0.3,
            get_train_time=lambda: 1.0,
        )
        for name, model_type, stacked in [
            ("1_CatBoost", "CatBoost", False),
            ("2_CatBoost_Stacked", "CatBoost", True),
            ("Ensemble_Stacked", "Ensemble", True),
            ("3_RandomFeature", "Xgboost", True),
        ]
    ]
    ldb = automl.get_leaderboard(filter_random_feature=True)
    assert ldb.model_type.tolist() == ["CatBoost", "CatBoost", "Ensemble"]
    assert ldb.is_stacked.tolist() == [False, True, True]
    assert ldb.metric_value.tolist() == [0.3, 0.3, 0.3]
