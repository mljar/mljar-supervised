import copy
import json

import pytest
from sklearn.datasets import make_classification

from supervised import AutoML
from supervised.exceptions import AutoMLException, ResourceAdmissionError
from supervised.algorithms.realmlp_resources import RealMLPAdmissionError, RealMLPResourceError
from supervised.algorithms.tabm_resources import TabMAdmissionError, TabMResourceError


@pytest.mark.parametrize("admission,error", [(RealMLPAdmissionError, RealMLPResourceError),
                                             (TabMAdmissionError, TabMResourceError)])
def test_only_admission_failures_are_skips(admission, error):
    exc = admission("too large", resources={"estimated_ram_bytes": 100})
    assert isinstance(exc, (ResourceAdmissionError, error))
    assert exc.resources["estimated_ram_bytes"] == 100
    assert not isinstance(error("actual GPU allocation failed"), ResourceAdmissionError)


def test_preflight_rechecks_pressure_and_does_not_blacklist_smaller_models(monkeypatch):
    from supervised.algorithms import realmlp_resources

    params = dict(name="candidate", preprocessing={}, validation_strategy={},
                  learner={"model_type": "Kumo", "max_context_size": 4096})
    previous = copy.deepcopy(params)
    previous["resource_skip"] = dict(resources=dict(device="cuda:0",
                                                   estimated_vram_bytes=100))
    automl = AutoML()
    automl._all_params = {"default": [previous]}
    free = [100]
    monkeypatch.setattr(realmlp_resources, "available_gpu_memory", lambda device: (free[0], 1000))
    with pytest.raises(ResourceAdmissionError) as exc:
        automl._resource_preflight(params)
    assert exc.value.resources["vram_budget_bytes"] == 70
    free[0] = 1000
    automl._resource_preflight(params)
    free[0] = 0
    params["learner"]["max_context_size"] = 1024
    automl._resource_preflight(params)


@pytest.mark.parametrize("resource_skip", [True, False])
def test_fit_continues_and_reports_resource_skips_separately(tmp_path, monkeypatch, capsys, resource_skip):
    original = AutoML.train_model

    def train(self, params):
        if params["learner"]["model_type"] == "Random Forest":
            if resource_skip:
                raise ResourceAdmissionError("estimated GPU memory exceeds budget",
                                             {"estimated_vram_bytes": 100, "vram_budget_bytes": 70})
            raise AutoMLException("actual allocation failure")
        return original(self, params)

    monkeypatch.setattr(AutoML, "train_model", train)
    X, y = make_classification(n_samples=100, n_features=5, random_state=2)
    automl = AutoML(results_path=str(tmp_path / "automl"), mode="Explain",
                    algorithms=["Decision Tree", "Random Forest"],
                    total_time_limit=30, train_ensemble=False, explain_level=0,
                    n_jobs=1, verbose=1)
    automl.fit(X, y)
    assert len(automl.predict(X[:3])) == 3
    console = capsys.readouterr().out
    root = tmp_path / "automl"
    if resource_skip:
        records = json.loads((root / "resource_skips.json").read_text())
        assert records and records[0]["resources"]["estimated_vram_bytes"] == 100
        assert "resource_skips.json" in (root / "README.md").read_text()
        assert not (root / "errors.md").exists()
        assert "There was an error" not in console
    else:
        assert "actual allocation failure" in (root / "errors.md").read_text()
        assert not (root / "resource_skips.json").exists()


def test_failed_framework_releases_partial_learner(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from supervised.model_framework import ModelFramework

    learners = []
    def fail(self, *args):
        learner = SimpleNamespace(model=object())
        learner.unload = lambda: setattr(learner, "model", None)
        self.learners = [learner]
        learners.append(learner)
        raise ResourceAdmissionError("budget rejected")

    monkeypatch.setattr(ModelFramework, "train", fail)
    X, y = make_classification(n_samples=80, n_features=4, random_state=3)
    automl = AutoML(results_path=str(tmp_path / "automl"), algorithms=["Decision Tree"],
                    train_ensemble=False, explain_level=0, n_jobs=1, verbose=0)
    with pytest.raises(AutoMLException, match="No models produced"):
        automl.fit(X, y)
    assert learners and all(learner.model is None for learner in learners)
