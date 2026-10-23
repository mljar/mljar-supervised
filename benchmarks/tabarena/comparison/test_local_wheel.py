from types import SimpleNamespace
from zipfile import ZipFile

import pytest

import setup_env
from common import read_json, sha256
from run import parse_args
from worker import prediction_graph


def make_wheel(path, version="1.4.0rc6", name="mljar-supervised", content="first"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        archive.writestr("mljar_supervised.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n")
        archive.writestr("supervised/__init__.py", content)
    return path


@pytest.mark.parametrize("version", ["1.4.0rc6", "1.4.0rc7"])
def test_local_wheel_metadata_and_separate_run_paths(tmp_path, version):
    wheel = make_wheel(tmp_path / "space in path" / f"mljar_supervised-{version}-py3-none-any.whl", version=version)
    args = parse_args(["--version", version, "--local-wheel", str(wheel), "--wait-for-resources"])
    assert args.wait_for_resources
    assert args.local_wheel == wheel
    assert args.run.name.endswith("-local-" + sha256(wheel)[:12])
    assert args.run != parse_args(["--version", version]).run
    for wrong_version in ["1.3.2", "1.4.0rc5"]:
        with pytest.raises(SystemExit):
            parse_args(["--version", wrong_version, "--local-wheel", str(wheel)])
    make_wheel(wheel, name="unrelated-package", version=version)
    with pytest.raises(ValueError, match="must contain mljar-supervised"):
        setup_env.local_wheel_info(wheel, version)


def test_local_environment_installs_retained_wheel_and_changes_path_for_new_build(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_env, "BASE", tmp_path / "benchmarks")
    monkeypatch.setattr(setup_env.shutil, "which", lambda name: "/usr/bin/uv")
    commands = []

    def command(args, **kwargs):
        commands.append([str(arg) for arg in args])
        if args[1] == "venv":
            python = args[-1] / "bin/python"
            python.parent.mkdir(parents=True)
            python.touch()

    monkeypatch.setattr(setup_env, "command", command)
    monkeypatch.setattr(setup_env.subprocess, "check_output", lambda *args, **kwargs: "mljar-supervised==1.4.0rc6\n")
    wheel = make_wheel(tmp_path / "mljar_supervised-1.4.0rc6-py3-none-any.whl")
    original_hash = sha256(wheel)
    first = setup_env.ensure_environment("1.4.0rc6", wheel)
    root = first.parent.parent
    info = read_json(root / "benchmark-environment.json")
    assert info["local_wheel"]["sha256"] == original_hash
    assert info["features_goldmine"] == "1.1.0"
    retained = root / "artifacts" / wheel.name
    assert sha256(retained) == original_hash
    assert any(f"mljar-supervised[all] @ {retained.as_uri()}" in args for args in commands)
    count = len(commands)
    assert setup_env.ensure_environment("1.4.0rc6", wheel) == first
    assert len(commands) == count
    make_wheel(wheel, content="second build")
    second = setup_env.ensure_environment("1.4.0rc6", wheel)
    assert second != first
    assert sha256(retained) == original_hash
    (second.parent.parent / "artifacts" / wheel.name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Retained local wheel changed"):
        setup_env.ensure_environment("1.4.0rc6", wheel)


def test_prediction_graph_counts_shared_models_and_fold_learners_once():
    def model(name, kind="LightGBM", stacked=False, members=()):
        return SimpleNamespace(get_name=lambda: name, get_type=lambda: kind,
                               _is_stacked=stacked, learners=[None] * 3,
                               selected_models=[dict(model=m, repeat=5) for m in members])

    base = model("base")
    stacked = model("stacked", stacked=True)
    inner = model("inner", "Ensemble", members=[base])
    final = model("final", "Ensemble", stacked=True, members=[inner, base, stacked])
    graph = prediction_graph(SimpleNamespace(_best_model=final, _stacked_models=[base]))
    assert graph["n_prediction_models"] == 2
    assert graph["n_prediction_learners"] == 6
    assert graph["n_stacking_inputs"] == 1
    assert graph["n_ensemble_members"] == 3
