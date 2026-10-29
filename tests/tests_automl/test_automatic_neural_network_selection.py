import pytest

from supervised import AutoML
from supervised.algorithms import realmlp, sdm, tabm
from supervised.utils import native_compatibility as native


@pytest.fixture
def available_networks(monkeypatch):
    monkeypatch.setattr(realmlp, "dependencies_available", lambda: True)
    monkeypatch.setattr(tabm, "dependencies_available", lambda: True)
    monkeypatch.setattr(sdm, "automatic_available", lambda: False)
    monkeypatch.setattr(native, "validate_dependencies", lambda algorithms: None)
    monkeypatch.setattr(native, "torch_loaded", lambda: False)
    monkeypatch.setattr(
        native, "check_native_compatibility",
        lambda needs_torch: dict(status="not_needed", reason="platform"),
    )


@pytest.mark.parametrize("status", ["not_needed", "passed"])
@pytest.mark.parametrize("has_realmlp,has_tabm", [(True, True), (True, False), (False, True), (False, False)])
def test_auto_prefers_available_modern_networks(
    available_networks, monkeypatch, capsys, status, has_realmlp, has_tabm
):
    monkeypatch.setattr(realmlp, "dependencies_available", lambda: has_realmlp)
    monkeypatch.setattr(tabm, "dependencies_available", lambda: has_tabm)
    monkeypatch.setattr(
        native, "check_native_compatibility",
        lambda needs_torch: dict(status=status, reason="test"),
    )
    automl = AutoML(mode="Compete")
    candidates = automl._candidate_algorithms()
    effective = automl._get_algorithms()
    prefer_modern = has_realmlp or has_tabm
    assert ("Neural Network" in effective) == (not prefer_modern)
    assert ("RealMLP" in effective) == has_realmlp
    assert ("TabM" in effective) == has_tabm
    assert effective == [name for name in candidates if not (prefer_modern and name == "Neural Network")]
    assert automl.algorithms == automl._requested_algorithms == "auto"
    assert ("Neural Network skipped" in capsys.readouterr().out) == prefer_modern
    if status == "passed":
        assert automl._native_compatibility["effective_algorithms"] == effective
        assert automl._native_compatibility["skipped_algorithms"] == []


@pytest.mark.parametrize("modern", [["RealMLP"], ["TabM"], ["RealMLP", "TabM"]])
def test_explicit_lists_keep_neural_network(available_networks, modern):
    requested = ["Neural Network", *modern]
    automl = AutoML(mode="Compete", algorithms=requested)
    automl._ml_task = "binary_classification"
    assert automl._get_algorithms() == requested
    assert automl.algorithms == automl._requested_algorithms == requested
    assert requested == ["Neural Network", *modern]


def test_weights_keep_neural_network_fallback(available_networks):
    automl = AutoML(mode="Compete")
    with pytest.warns(UserWarning, match="sample weights"):
        effective = automl._get_algorithms(sample_weight=[1.0, 1.0])
    assert "Neural Network" in effective
    assert "RealMLP" not in effective
    assert "TabM" not in effective


def test_native_failure_keeps_neural_network_fallback(available_networks, monkeypatch):
    monkeypatch.setattr(
        native, "check_native_compatibility",
        lambda needs_torch: dict(status="inconclusive", reason="timeout"),
    )
    automl = AutoML(mode="Compete")
    effective = automl._get_algorithms()
    assert "Neural Network" in effective
    assert "RealMLP" not in effective
    assert "TabM" not in effective
    assert automl._native_compatibility["effective_algorithms"] == effective
    assert automl._native_compatibility["skipped_algorithms"] == ["RealMLP", "TabM"]


def test_foundation_models_alone_keep_neural_network(available_networks, monkeypatch):
    monkeypatch.setattr(realmlp, "dependencies_available", lambda: False)
    monkeypatch.setattr(tabm, "dependencies_available", lambda: False)
    monkeypatch.setattr(sdm, "automatic_available", lambda: True)
    effective = AutoML(mode="Compete")._get_algorithms()
    assert {"Neural Network", "Kumo", "TabICLv2"}.issubset(effective)


@pytest.mark.parametrize("mode", ["Explain", "Perform", "Optuna"])
def test_other_modes_keep_neural_network(available_networks, mode):
    assert "Neural Network" in AutoML(mode=mode)._get_algorithms()


def test_quiet_selection(available_networks, capsys):
    assert "Neural Network" not in AutoML(mode="Compete", verbose=0)._get_algorithms()
    assert capsys.readouterr().out == ""
