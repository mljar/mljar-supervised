import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_friedman1

from supervised import AutoML
from supervised.apps.dependencies import runtime_environment
from supervised.apps.generator import _create_automl_zip
from supervised.exceptions import AutoMLException


def framework(root, name, kind, *, preprocessing=None, checkpoint=None, size="small"):
    folder = root / name
    folder.mkdir(parents=True)
    learner = dict(params=dict(model_type=kind, size=size))
    if checkpoint:
        learner["training_info"] = dict(checkpoint=dict(revision=checkpoint, filename="classifier.pt"))
    description = dict(params=dict(learner=dict(model_type=kind)), learners=[learner],
                       preprocessing=preprocessing or [], joblib_version="1.6.0")
    (folder / "framework.json").write_text(json.dumps(description))
    return folder


@pytest.mark.parametrize("kind,extra", [("Decision Tree", None), ("RealMLP", "realmlp"),
                                      ("TabM", "tabm"), ("Kumo", "tfm"), ("TabICLv2", "tfm")])
def test_requirements_follow_selected_models(tmp_path, kind, extra):
    framework(tmp_path, "selected", kind)
    framework(tmp_path, "unused", "Kumo" if kind != "Kumo" else "RealMLP")
    result = runtime_environment(tmp_path, ["selected"])
    assert result["extras"] == sorted(["apps"] + ([extra] if extra else []))
    assert result["algorithms"] == [kind]
    assert "joblib==1.6.0" in result["requirements"]
    assert "mercury>=3.2.12,<4" in result["requirements"]


def test_requirements_include_ensemble_and_stacking_inputs(tmp_path):
    framework(tmp_path, "stacked", "TabM")
    framework(tmp_path, "base", "Kumo")
    ensemble = tmp_path / "Ensemble_Stacked"
    ensemble.mkdir()
    (ensemble / "ensemble.json").write_text("{}")
    result = runtime_environment(tmp_path, ["base", "stacked", "Ensemble_Stacked"])
    assert result["extras"] == ["apps", "tabm", "tfm"]
    assert result["algorithms"] == ["Kumo", "TabM"]


def test_conflicting_joblib_versions_are_not_silently_exported(tmp_path):
    framework(tmp_path, "one", "Linear")
    folder = framework(tmp_path, "two", "RealMLP")
    path = folder / "framework.json"
    payload = json.loads(path.read_text())
    payload["joblib_version"] = "1.5.0"
    path.write_text(json.dumps(payload))
    with pytest.raises(AutoMLException, match="different joblib"):
        runtime_environment(tmp_path, ["one", "two"])


def test_invalid_saved_version_cannot_add_another_requirement(tmp_path):
    folder = framework(tmp_path, "model", "Linear")
    path = folder / "framework.json"
    payload = json.loads(path.read_text())
    payload["joblib_version"] = "1.6.0\nunexpected-package"
    path.write_text(json.dumps(payload))
    with pytest.raises(AutoMLException, match="invalid saved joblib"):
        runtime_environment(tmp_path, ["model"])


@pytest.mark.parametrize("layout", ["shared", "model-local"])
def test_zip_preserves_selected_shared_artifacts_and_licenses(tmp_path, layout):
    root, app = tmp_path / "run", tmp_path / "app"
    root.mkdir()
    app.mkdir()
    names = ["model"]
    (root / "params.json").write_text(json.dumps(dict(saved=names, best_model="model")))
    framework(root, "model", "Kumo", checkpoint="revision", preprocessing=[
        dict(feature_discovery=dict(info=dict(directory="feature_discovery/fold")))])
    discovery = root / "feature_discovery/fold"
    discovery.mkdir(parents=True)
    (discovery / "generator.joblib").write_bytes(b"frozen")
    (discovery / "training.joblib").write_bytes(b"training-only")
    prefix = "" if layout == "shared" else "model/"
    weights = root / (prefix + "sdm_weights/Kumo-revision/small")
    weights.mkdir(parents=True)
    (weights / "classifier.pt").write_bytes(b"weights")
    (weights / "0-LICENSE.md").write_text("license")
    unused = root / "sdm_weights/Kumo-revision/large"
    unused.mkdir(parents=True, exist_ok=True)
    (unused / "classifier.pt").write_bytes(b"unused")
    _create_automl_zip(str(app), str(root), names)
    with zipfile.ZipFile(app / "automl.zip") as archive:
        files = archive.namelist()
        assert "automl/feature_discovery/fold/generator.joblib" in files
        assert "automl/" + prefix + "sdm_weights/Kumo-revision/small/classifier.pt" in files
        assert "automl/" + prefix + "sdm_weights/Kumo-revision/small/0-LICENSE.md" in files
        assert not any("training.joblib" in p or "/large/" in p for p in files)
    (discovery / "generator.joblib").unlink()
    with pytest.raises(AutoMLException, match="missing discovery generator"):
        _create_automl_zip(str(app), str(root), names)
    (discovery / "generator.joblib").write_bytes(b"frozen")
    (weights / "classifier.pt").unlink()
    with pytest.raises(AutoMLException, match="missing checkpoint file"):
        _create_automl_zip(str(app), str(root), names)


def test_shared_artifacts_cannot_escape_the_results_directory(tmp_path):
    root, app = tmp_path / "run", tmp_path / "app"
    root.mkdir()
    app.mkdir()
    (root / "params.json").write_text(json.dumps(dict(saved=["model"])))
    framework(root, "model", "Linear", preprocessing=[
        dict(feature_discovery=dict(info=dict(directory="../outside")))])
    with pytest.raises(AutoMLException, match="outside the results directory"):
        _create_automl_zip(str(app), str(root), ["model"])


def test_discovery_export_predicts_without_original_run(tmp_path):
    X, y = make_friedman1(n_samples=250, n_features=6, random_state=42)
    X = pd.DataFrame(X, columns=list("abcdef"))
    root = tmp_path / "run"
    automl = AutoML(results_path=str(root), mode="Compete", algorithms=["Decision Tree"],
                    total_time_limit=120, n_jobs=1, verbose=0, explain_level=0,
                    golden_features=False, feature_discovery=True, features_selection=False,
                    kmeans_features=False, mix_encoding=False, boost_on_errors=False,
                    start_random_models=1, hill_climbing_steps=0, train_ensemble=False,
                    stack_models=False, validation_strategy=dict(validation_type="split", train_ratio=0.8))
    automl.fit(X, y)
    chosen = next(model for model in automl._models if "FeatureDiscovery" in model.get_name())
    automl._best_model = chosen
    params_path = root / "params.json"
    params = json.loads(params_path.read_text())
    params["best_model"] = chosen.get_name()
    params_path.write_text(json.dumps(params))
    expected = automl.predict(X.iloc[:4])
    app = Path(automl.app(path=str(tmp_path / "app"), verbose=False))
    shutil.move(root, tmp_path / "original-not-available")
    spec = importlib.util.spec_from_file_location("exported_app_support", app / "app_support.py")
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    single = support.predict_single(X.iloc[:1])
    assert single["prediction"] == pytest.approx(expected[0])
    batch, error = support.batch_predict(SimpleNamespace(value=X.iloc[:4].to_csv(index=False).encode()))
    assert error is None
    np.testing.assert_allclose(batch["prediction"], expected)
    manifest = json.loads((app / "mljar_app.json").read_text())
    assert manifest["environment"]["extras"] == ["apps"]
    assert "[apps]==" in (app / "requirements.txt").read_text()
