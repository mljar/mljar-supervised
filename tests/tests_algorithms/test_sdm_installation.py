"""The installed wheel must supply the supported PyPI backend through extras."""

import importlib.metadata
import importlib.util
import re

import pytest
from packaging.requirements import Requirement

from supervised.algorithms import sdm
from supervised.exceptions import AutoMLException


@pytest.mark.parametrize("extra", ["tfm", "kumo", "tabiclv2", "all"])
def test_tfm_extras_install_the_supported_pypi_backend(extra):
    requirements = [
        Requirement(value) for value in importlib.metadata.requires("mljar-supervised")
    ]
    selected = [
        r for r in requirements if r.marker is None or r.marker.evaluate({"extra": extra})
    ]
    backend = [r for r in selected if r.name == "structured-data-models"]
    assert len(backend) == 1
    assert str(backend[0].specifier) == f"=={sdm.SDM_VERSION}"
    assert backend[0].url is None
    assert all(r.url is None for r in selected)


@pytest.mark.parametrize("extra", ["", "realmlp", "tabm", "feature-discovery"])
def test_other_installs_do_not_require_sdm(extra):
    for value in importlib.metadata.requires("mljar-supervised"):
        requirement = Requirement(value)
        if requirement.name == "structured-data-models":
            assert requirement.marker is not None
            assert not requirement.marker.evaluate({"extra": extra})


@pytest.mark.parametrize("installed", ["0.0.0a0", "0.2.0rc2.dev18+gb97cb914e", "0.2.0", "0.3.0"])
def test_unsupported_backend_has_a_pypi_repair_command(monkeypatch, installed):
    monkeypatch.setattr(sdm, "dependencies_available", lambda: True)
    monkeypatch.setattr(sdm, "version", lambda name: installed)
    with pytest.raises(AutoMLException, match=re.escape(f"found {installed}")) as error:
        sdm.check_dependencies()
    message = str(error.value)
    assert f"structured-data-models=={sdm.SDM_VERSION}" in message
    assert "mljar-supervised[tfm]" in message
    assert "git+" not in message


def test_missing_backend_has_a_pypi_install_command(monkeypatch):
    monkeypatch.setattr(sdm, "dependencies_available", lambda: False)
    with pytest.raises(AutoMLException, match="pip install") as error:
        sdm.check_dependencies()
    assert "git+" not in str(error.value)


@pytest.mark.skipif(importlib.util.find_spec("sdm") is None, reason="optional SDM backend")
def test_installed_backend_version_and_adapter_import():
    sdm.check_dependencies()
    from supervised.algorithms import _sdm_backend

    assert _sdm_backend.sdm.models.KumoTabular
    assert _sdm_backend.sdm.models.TabICLv2
