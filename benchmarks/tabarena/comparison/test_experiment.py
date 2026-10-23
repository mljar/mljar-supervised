from pathlib import Path
import sys

import pytest

import experiment
from common import read_json, write_json
from reevaluate import tree_hashes


def test_experiment_freezes_source_and_does_not_modify_old_results(tmp_path, monkeypatch):
    repo, original, run = [tmp_path / x for x in ("repo", "original", "experiment")]
    (repo / "supervised").mkdir(parents=True)
    (repo / "supervised/__init__.py").write_text("__version__ = '1.4.0rc5'\n")
    (repo / "supervised/new_fix.py").write_text("value = 1\n")
    monkeypatch.setattr(experiment, "ROOT", repo)
    attempt = original / "credit-g/r0f0/attempt-001"
    write_json(attempt / "case.json", dict(dataset="credit-g", version="1.4.0rc5"))
    write_json(attempt.parent / "status.json", dict(status="error", attempt=str(attempt)))
    write_json(original / "environment.json", dict(python=sys.executable))
    run.mkdir()
    before = tree_hashes(original)
    manifest = experiment.freeze(run, original, ["credit-g"], 3600, [1, 2])
    (repo / "supervised/new_fix.py").write_text("value = 2\n")
    assert experiment.freeze(run, original, ["credit-g"], 3600, [1, 2]) == manifest
    assert (run / "source/supervised/new_fix.py").read_text() == "value = 1\n"
    assert before == tree_hashes(original)
    with pytest.raises(ValueError, match="settings changed"):
        experiment.freeze(run, original, ["credit-g"], 10, [1, 2])
    (run / "source/supervised/new_fix.py").write_text("changed\n")
    with pytest.raises(RuntimeError, match="Files changed"):
        experiment.freeze(run, original, ["credit-g"], 3600, [1, 2])


def test_worker_memory_snapshots_do_not_require_torch(tmp_path, monkeypatch):
    from worker import record_memory

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    record_memory(tmp_path, "before_fit")
    record_memory(tmp_path, "after_fit")
    import json
    snapshots = [json.loads(line) for line in (tmp_path / "memory.jsonl").read_text().splitlines()]
    assert [s["phase"] for s in snapshots] == ["before_fit", "after_fit"]
    assert all(s["rss_bytes"] > 0 and "cuda" not in s for s in snapshots)
