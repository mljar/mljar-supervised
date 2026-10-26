"""Verify wheel dependency selection and the installed XGBoost evaluation path."""

import importlib.metadata

import numpy as np
import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement


def xgboost_requirements():
    return [r for value in importlib.metadata.requires("mljar-supervised")
            if (r := Requirement(value)).name in ("xgboost", "xgboost-cpu")]


@pytest.mark.parametrize("python_version", ["3.11", "3.12"])
@pytest.mark.parametrize("system,machine,expected", [
    ("linux", "x86_64", "xgboost-cpu"),
    ("linux", "aarch64", "xgboost-cpu"),
    ("win32", "AMD64", "xgboost-cpu"),
    ("win32", "x86_64", "xgboost-cpu"),
    ("darwin", "x86_64", "xgboost"),
    ("darwin", "arm64", "xgboost"),
    ("win32", "ARM64", "xgboost"),
    ("linux", "ppc64le", "xgboost"),
])
def test_exactly_one_xgboost_provider_per_platform(system, machine, expected, python_version):
    environment = {**default_environment(), "sys_platform": system,
                   "platform_machine": machine, "python_version": python_version}
    selected = [r.name for r in xgboost_requirements() if r.marker.evaluate(environment)]
    assert selected == [expected]


def test_installed_xgboost_multiclass_evaluation_and_prediction():
    import xgboost as xgb

    expected = next(r.name for r in xgboost_requirements() if r.marker.evaluate())
    providers = []
    for name in ("xgboost", "xgboost-cpu", "xgboost-cu12"):
        try:
            importlib.metadata.version(name)
            providers.append(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    assert providers == [expected], "Overlapping XGBoost packages share import files"
    if expected == "xgboost-cpu":
        assert xgb.build_info()["USE_CUDA"] is False
    data = xgb.DMatrix(np.arange(72, dtype=np.float32).reshape(24, 3), label=np.arange(24) % 3)
    scores = {}
    model = xgb.train(
        dict(device="cpu", tree_method="hist", objective="multi:softprob", num_class=3,
             eval_metric="mlogloss", nthread=1, seed=42),
        data, num_boost_round=2, evals=[(data, "validation")],
        evals_result=scores, verbose_eval=False,
    )
    assert len(scores["validation"]["mlogloss"]) == 2
    predictions = model.predict(data)
    assert predictions.shape == (24, 3)
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions.sum(axis=1), 1, rtol=1e-6)
