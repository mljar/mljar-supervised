import time

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from supervised import AutoML
from supervised.exceptions import AutoMLException
from supervised.tuner.time_controller import TimeController


def test_exhausted_budget_skips_kmeans_without_key_error(tmp_path, monkeypatch, capsys):
    original_init = TimeController.__init__

    def expired_preparation(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Reproduce preparation consuming the budget without a slow real dataset.
        self._start_time = time.time() - self._total_time_limit - 1

    monkeypatch.setattr(TimeController, "__init__", expired_preparation)
    X, y = make_classification(n_samples=80, n_features=4, random_state=42)
    X = pd.DataFrame(X, columns=list("abcd"))
    model = AutoML(
        results_path=str(tmp_path / "run"),
        mode="Compete",
        ml_task="binary_classification",
        algorithms=["Nearest Neighbors"],
        total_time_limit=60,
        explain_level=0,
        n_jobs=1,
    )

    with pytest.raises(AutoMLException, match="No models produced"):
        model.fit(X, y)

    output = capsys.readouterr().out
    assert "Skip adjust_validation because of the time limit." in output
    assert "Skip default_algorithms because of the time limit." in output
    assert "Skip kmeans_features because no parameters were generated." in output
    assert "KeyError" not in output
    assert len(model._models) == 0
