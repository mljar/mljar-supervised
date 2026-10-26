"""Validate the installed distribution's base and optional dependency contract."""

import importlib.metadata
import importlib.util
import os

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


def selected_requirements(extra=""):
    requirements = map(Requirement, importlib.metadata.requires("mljar-supervised"))
    return {r.name: r for r in requirements
            if r.marker is None or r.marker.evaluate({"extra": extra})}


def test_python_minimum_and_classifiers():
    metadata = importlib.metadata.metadata("mljar-supervised")
    supported = SpecifierSet(metadata["Requires-Python"])
    assert "3.9" not in supported and "3.10" not in supported
    assert "3.11" in supported and "3.12" in supported
    classifiers = metadata.get_all("Classifier")
    assert "Programming Language :: Python :: 3.9" not in classifiers
    assert "Programming Language :: Python :: 3.10" not in classifiers


def test_discovery_is_unconditional_and_alias_adds_nothing():
    base = selected_requirements()
    discovery = base["features-goldmine"]
    assert discovery.marker is None
    assert discovery.specifier == SpecifierSet(">=1.1,<1.2")
    assert selected_requirements("feature-discovery") == base
    assert "feature-discovery" in importlib.metadata.metadata("mljar-supervised").get_all("Provides-Extra")


@pytest.mark.parametrize("extra,backends", [
    ("", set()), ("feature-discovery", set()),
    ("realmlp", {"torch", "pytabkit"}), ("tabm", {"torch", "pytabkit"}),
    ("tfm", {"torch", "structured-data-models", "huggingface_hub"}),
    ("kumo", {"torch", "structured-data-models", "huggingface_hub"}),
    ("tabiclv2", {"torch", "structured-data-models", "huggingface_hub"}),
    ("all", {"torch", "pytabkit", "structured-data-models", "huggingface_hub"}),
])
def test_optional_backends_remain_opt_in(extra, backends):
    selected = selected_requirements(extra)
    optional = {"torch", "pytabkit", "structured-data-models", "huggingface_hub", "mercury"}
    assert selected.keys() & optional == backends
    assert "features-goldmine" in selected


def test_installed_discovery_and_activation_policy():
    from supervised import AutoML
    from supervised.preprocessing.feature_discovery_transformer import check_dependencies

    assert check_dependencies().__name__ == "GoldenFeatures"
    for mode in ["Explain", "Perform", "Compete", "Optuna"]:
        automl = AutoML(mode=mode)
        automl._validation_strategy = {"validation_type": "split", "shuffle": True}
        assert automl._get_feature_discovery() is (mode == "Compete")
        automl.feature_discovery = False
        assert automl._get_feature_discovery() is False


@pytest.mark.skipif(os.environ.get("MLJAR_TEST_BASE_INSTALL") != "1", reason="base-only environment check")
def test_fresh_base_install_has_no_optional_backends():
    for module in ["torch", "pytabkit", "sdm", "mercury"]:
        assert importlib.util.find_spec(module) is None, module


@pytest.mark.parametrize("version", [None, "1.0.0", "1.2.0"])
def test_discovery_dependency_errors_do_not_request_an_extra(monkeypatch, version):
    from supervised.preprocessing import feature_discovery_transformer as discovery
    from supervised.exceptions import AutoMLException

    def installed(name):
        if version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return version

    monkeypatch.setattr(discovery.importlib.metadata, "version", installed)
    with pytest.raises(AutoMLException) as exc:
        discovery.check_dependencies()
    assert "standard MLJAR installation" in str(exc.value)
    assert "[feature-discovery]" not in str(exc.value)
