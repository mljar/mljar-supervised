import json
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import psutil
import pytest

from supervised import AutoML
from supervised.exceptions import NativeCompatibilityError
from supervised.utils import native_compatibility as native


@pytest.fixture
def mac(monkeypatch):
    native._cache.clear()
    monkeypatch.setattr(native, "is_macos", lambda: True)
    monkeypatch.setattr(native, "torch_loaded", lambda: False)
    monkeypatch.setattr(native.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(native, "_inspect", lambda: ("environment-a", {"torch": "test"}))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=2 * 1024**3))
    yield
    native._cache.clear()


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_other_platforms_do_not_probe(monkeypatch, platform):
    monkeypatch.setattr(native, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(native, "_inspect", lambda: pytest.fail("inspected on an unaffected platform"))
    assert native.check_native_compatibility(True)["reason"] == "platform"


def test_unused_or_absent_torch_does_not_probe(mac, monkeypatch):
    monkeypatch.setattr(native, "_run_child", lambda *a: pytest.fail("unexpected probe"))
    assert native.check_native_compatibility()["reason"] == "torch_unused"
    monkeypatch.setattr(native.importlib.util, "find_spec", lambda name: None)
    assert native.check_native_compatibility(True)["reason"] == "torch_absent"


@pytest.mark.parametrize("status", ["passed", "blocked"])
def test_cache_coalesces_and_invalidates(mac, monkeypatch, status):
    calls = []

    def run(order, deadline):
        calls.append(order)
        time.sleep(0.02)
        return dict(status=status, reason="test")

    monkeypatch.setattr(native, "_run_child", run)
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda _: native.check_native_compatibility(True), range(3)))
    expected = 2 if status == "passed" else 1
    assert len(calls) == expected
    results[0]["versions"]["torch"] = "mutated"
    assert native.check_native_compatibility(True)["versions"]["torch"] == "test"
    monkeypatch.setattr(native, "_inspect", lambda: ("environment-b", {}))
    native.check_native_compatibility(True)
    assert len(calls) == expected * 2


def test_memory_admission_is_retriable(mac, monkeypatch):
    calls = []
    monkeypatch.setattr(native, "_run_child", lambda *args: calls.append(args) or dict(status="passed", reason="ok"))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=1))
    assert native.check_native_compatibility(True)["reason"] == "insufficient_probe_memory"
    assert calls == []
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=2 * 1024**3))
    assert native.check_native_compatibility(True)["status"] == "passed"
    assert len(calls) == 2


def child_script(tmp_path, body):
    path = tmp_path / "probe.py"
    path.write_text(body)
    return path


@pytest.mark.parametrize("body,reason", [
    ('import json, sys; print(json.dumps(dict(status="passed", order=sys.argv[1])))', "probe_passed"),
    ('print("invalid")', "invalid_response"),
    ('raise RuntimeError("ordinary Python error")', "probe_error"),
    ('print(\'{"reason": "import_error"}\'); raise ImportError("missing")', "import_error"),
    ('print("x" * 1000000)', "invalid_response"),
])
def test_child_responses_and_bounded_output(tmp_path, body, reason):
    result = native._run_child("boosting-first", time.monotonic() + 3,
                               script=child_script(tmp_path, body))
    assert result["reason"] == reason
    assert len(result.get("diagnostic", "")) <= native.MAX_OUTPUT_BYTES


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal semantics")
def test_native_crash_is_contained(tmp_path):
    # Linux ignores RLIMIT_CORE for piped crash reporters (for example Apport).
    # Disable dumps in this deliberately crashing child so reporter latency is
    # not confused with the probe timeout. Keep the production deadline intact.
    script = child_script(tmp_path, """import ctypes, os, resource, signal, sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
if sys.platform.startswith("linux"):
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    if prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
        raise OSError(ctypes.get_errno(), "Could not disable test crash dumps")
os.kill(os.getpid(), signal.SIGSEGV)
""")
    result = native._run_child("boosting-first", time.monotonic() + 3, script=script)
    assert result["status"] == "blocked", result
    assert result["reason"] == "native_crash"
    assert result["signal"] == signal.SIGSEGV


def test_hang_has_one_three_second_deadline_and_is_reaped(mac, monkeypatch, tmp_path):
    pid_path = tmp_path / "pid"
    script = child_script(tmp_path, f"import os, time\nopen({str(pid_path)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n")
    original = native._run_child
    calls = []

    def run(order, deadline):
        calls.append(order)
        return original(order, deadline, script=script)

    monkeypatch.setattr(native, "_run_child", run)
    started = time.monotonic()
    result = native.check_native_compatibility(True)
    elapsed = time.monotonic() - started
    assert result["reason"] == "timeout"
    assert calls == ["boosting-first"]
    assert 2.5 < elapsed < 3.25  # Small OS scheduling tolerance, not a longer watchdog.
    assert not psutil.pid_exists(int(pid_path.read_text()))


def test_second_case_uses_remaining_budget(mac, monkeypatch, tmp_path):
    script = child_script(tmp_path, 'import time, json, sys\ntime.sleep(1.7)\nprint(json.dumps(dict(status="passed", order=sys.argv[1])))\n')
    original = native._run_child
    deadlines = []

    def run(order, deadline):
        deadlines.append(deadline)
        return original(order, deadline, script=script)

    monkeypatch.setattr(native, "_run_child", run)
    started = time.monotonic()
    assert native.check_native_compatibility(True)["reason"] == "timeout"
    assert time.monotonic() - started < 3.25
    assert len(deadlines) == 2 and deadlines[0] == deadlines[1]


def test_memory_excess_kills_child(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "MAX_CHILD_RSS", 1)
    script = child_script(tmp_path, "import time; time.sleep(60)")
    assert native._run_child("boosting-first", time.monotonic() + 3, script=script)["reason"] == "probe_memory_limit"


def test_launch_failure(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("cannot start")

    monkeypatch.setattr(subprocess, "Popen", fail)
    assert native._run_child("boosting-first", time.monotonic() + 3)["reason"] == "launch_failed"


def test_cancellation_reaps_child(tmp_path, monkeypatch):
    processes = []
    original = subprocess.Popen

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    def cancel(delay):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(native, "time", SimpleNamespace(monotonic=time.monotonic, sleep=cancel))
    script = child_script(tmp_path, "import time; time.sleep(60)")
    with pytest.raises(KeyboardInterrupt):
        native._run_child("boosting-first", time.monotonic() + 3, script=script)
    assert len(processes) == 1
    assert processes[0].poll() is not None


@pytest.fixture
def rejected(monkeypatch):
    monkeypatch.setattr(native, "validate_dependencies", lambda algorithms: None)
    monkeypatch.setattr(native, "torch_loaded", lambda: False)
    result = dict(status="inconclusive", reason="timeout", diagnostic="bounded details")
    monkeypatch.setattr(native, "check_native_compatibility", lambda needs_torch=False:
                        result.copy() if needs_torch else dict(status="not_needed", reason="torch_unused"))
    return result


def test_automatic_selection_preserves_request(rejected, monkeypatch, capsys):
    automl = AutoML(mode="Compete")
    monkeypatch.setattr(automl, "_candidate_algorithms", lambda: ["LightGBM", "RealMLP", "TabM"])
    assert automl._get_algorithms() == ["LightGBM"]
    assert automl._get_algorithms() == ["LightGBM"]
    assert automl.algorithms == "auto"
    assert capsys.readouterr().out.count("skipped:") == 1
    assert automl._native_compatibility["skipped_algorithms"] == ["RealMLP", "TabM"]
    assert "diagnostic" not in automl._native_compatibility


@pytest.mark.parametrize("algorithm", sorted(native.TORCH_ALGORITHMS))
def test_explicit_models_raise_before_training(rejected, algorithm):
    automl = AutoML(mode="Compete", algorithms=[algorithm])
    automl._ml_task = "binary_classification"
    with pytest.raises(NativeCompatibilityError, match=algorithm):
        automl._get_algorithms()
    assert automl.algorithms == [algorithm]


def test_torch_loaded_prevents_classical_fallback(rejected, monkeypatch):
    monkeypatch.setattr(native, "torch_loaded", lambda: True)
    automl = AutoML(mode="Compete")
    monkeypatch.setattr(automl, "_candidate_algorithms", lambda: ["LightGBM", "RealMLP"])
    with pytest.raises(NativeCompatibilityError):
        automl._get_algorithms()


def test_factory_and_framework_block_before_artifact_loading(rejected, tmp_path):
    from supervised.algorithms.factory import AlgorithmFactory
    from supervised.model_framework import ModelFramework

    with pytest.raises(NativeCompatibilityError):
        AlgorithmFactory.load({"params": {"model_type": "Kumo"}}, "missing.model", True)
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "framework.json").write_text(json.dumps({"params": {"learner": {"model_type": "Kumo"}}}))
    with pytest.raises(NativeCompatibilityError):
        ModelFramework.load(str(tmp_path), "model")


def test_direct_learner_blocks_fit_predict_and_load(rejected):
    from supervised.algorithms.realmlp import RealMLPAlgorithm

    learner = RealMLPAlgorithm({"ml_task": "binary_classification"})
    for method, args in [("fit", (None, None)), ("predict", (None,)), ("load", ("missing.model",))]:
        with pytest.raises(NativeCompatibilityError):
            getattr(learner, method)(*args)


def test_fallback_report_and_reload(rejected, monkeypatch, tmp_path, capsys):
    from sklearn.datasets import make_classification

    monkeypatch.setattr(AutoML, "_candidate_algorithms", lambda self: ["Decision Tree", "RealMLP"])
    root = tmp_path / "run"
    automl = AutoML(results_path=str(root), mode="Compete", train_ensemble=False,
                    stack_models=False, explain_level=0, golden_features=False,
                    feature_discovery=False, features_selection=False, kmeans_features=False,
                    start_random_models=1, hill_climbing_steps=0, boost_on_errors=False,
                    validation_strategy={"validation_type": "split", "train_ratio": 0.8},
                    total_time_limit=15, n_jobs=1, verbose=0)
    X, y = make_classification(n_samples=80, n_features=4, random_state=42)
    automl.fit(X, y)
    assert "skipped:" not in capsys.readouterr().out
    params = json.loads((root / "params.json").read_text())
    assert params["requested_algorithms"] == "auto"
    assert params["algorithms"] == ["Decision Tree"]
    assert params["native_compatibility"]["reason"] == "timeout"
    assert "RealMLP" in (root / "README.md").read_text()
    assert (root / "native_compatibility.log").read_text() == "bounded details"
    assert automl.report_structured(format="dict")["native_compatibility"]["skipped_algorithms"] == ["RealMLP"]
    assert "RealMLP" in automl.report_structured(format="markdown")
    assert json.loads((root / "report_structured.json").read_text())["run_summary"]["native_compatibility"]["reason"] == "timeout"
    loaded = AutoML(results_path=str(root), verbose=0)
    assert (automl.predict(X[:4]) == loaded.predict(X[:4])).all()
    assert loaded._native_compatibility == automl._native_compatibility


def test_existing_missing_dependency_error_precedes_probe(monkeypatch):
    from supervised.algorithms import realmlp
    from supervised.exceptions import AutoMLException

    monkeypatch.setattr(native, "is_macos", lambda: True)
    monkeypatch.setattr(realmlp, "dependencies_available", lambda: False)
    monkeypatch.setattr(native, "_run_child", lambda *args: pytest.fail("unexpected probe"))
    automl = AutoML(mode="Compete", algorithms=["RealMLP"])
    automl._ml_task = "binary_classification"
    with pytest.raises(AutoMLException, match="install") as exc:
        automl._get_algorithms()
    assert not isinstance(exc.value, NativeCompatibilityError)


def test_saved_ensemble_and_resume_refuse_without_mutating_artifacts(rejected, tmp_path):
    from sklearn.datasets import make_classification

    root = tmp_path / "saved"
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    automl = AutoML(results_path=str(root), algorithms=["Decision Tree", "Linear"],
                    train_ensemble=True, explain_level=0, n_jobs=1, verbose=0)
    automl.fit(X, y)
    assert any(model.get_type() == "Ensemble" for model in automl._models)
    params_path = root / "params.json"
    params = json.loads(params_path.read_text())
    params["fit_level"] = "default"
    params_path.write_text(json.dumps(params))
    progress_path = root / "progress.json"
    progress = json.loads(progress_path.read_text())
    progress["fit_level"] = "default"
    progress_path.write_text(json.dumps(progress))
    # Simulate a saved run from another machine that requires a blocked backend.
    framework_path = next(root.glob("*/framework.json"))
    framework = json.loads(framework_path.read_text())
    framework["params"]["learner"]["model_type"] = "Kumo"
    framework_path.write_text(json.dumps(framework))
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    for operation in ("load", "predict", "fit"):
        loaded = AutoML(results_path=str(root), algorithms=["Decision Tree"], verbose=0)
        with pytest.raises(NativeCompatibilityError):
            if operation == "load":
                loaded.load(str(root))
            elif operation == "predict":
                loaded.predict(X[:3])
            else:
                loaded.fit(X, y)
        assert loaded._models == []
    after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after


def test_sdm_discovery_on_mac_never_imports_torch(monkeypatch):
    from supervised.algorithms import sdm

    monkeypatch.setattr(sdm, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(sdm, "check_dependencies", lambda: pytest.fail("unnecessary runtime discovery"))
    assert sdm.automatic_available() is False
