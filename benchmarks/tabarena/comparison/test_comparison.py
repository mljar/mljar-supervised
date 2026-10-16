"""Checks for label alignment, process handling, and comparable result retention."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from common import cpu_affinity, read_json, write_json
from report import build_report
from run import execute, freeze_run, parse_args
from worker import probability_matrix, score_predictions


def test_multiclass_probabilities_follow_encoded_labels_not_column_sort():
    frame = pd.DataFrame({f"prediction_{i}": [float(i == 10)] for i in sorted(range(12), key=str)})
    frame["label"] = 10
    values = probability_matrix(frame, 12)
    assert values.argmax(axis=1).tolist() == [10]
    score, _ = score_predictions(frame, np.array([10]), "multiclass", 12)
    assert score < 1e-10


def test_metrics_have_correct_direction_and_validate_probabilities():
    frame = pd.DataFrame({"prediction_1": [.1, .9], "prediction_0": [.9, .1]})
    assert score_predictions(frame, [0, 1], "binary", 2) == (1, 0)
    frame.loc[0, "prediction_0"] = .8
    with pytest.raises(ValueError, match="Invalid class probabilities"):
        probability_matrix(frame, 2)
    assert score_predictions(pd.DataFrame({"prediction": [1., 4.]}), [1., 2.], "regression")[0] == pytest.approx(2 ** .5)


def test_cpu_partitions_never_share_physical_cores():
    def physical(cpus):
        return {(Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id").read_text(),
                 Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/core_id").read_text()) for cpu in cpus}
    first, second = cpu_affinity("1.3.2", 1), cpu_affinity("1.4.0rc1", 1)
    assert not physical(first) & physical(second)
    assert cpu_affinity("1.4.0rc2", 1) == second
    assert cpu_affinity("1.4.0rc3", 1) == second
    assert cpu_affinity("1.4.0rc4", 1) == second


def test_rc4_defaults_preserve_older_candidate_paths():
    stable = parse_args(["--version", "1.3.2"])
    rc2 = parse_args(["--version", "1.4.0rc2"])
    rc1 = parse_args(["--version", "1.4.0rc1"])
    legacy_stable = parse_args(["--version", "1.3.2", "--candidate-version", "1.4.0rc1"])
    rc3 = parse_args(["--version", "1.4.0rc3"])
    rc4 = parse_args(["--version", "1.4.0rc4"])
    assert stable.run == rc4.run != rc3.run
    assert "vs-1.4.0rc4-small-3600s-15threads" in rc4.run.name
    assert not rc4.evaluate_all_models
    assert parse_args(["--version", "1.4.0rc4", "--evaluate-all-models"]).evaluate_all_models
    assert rc3.run != rc2.run
    assert "vs-1.4.0rc3-small-3600s-15threads" in rc3.run.name
    assert "vs-1.4.0rc2-small-3600s-15threads" in rc2.run.name
    assert legacy_stable.run == rc1.run != rc2.run
    assert stable.device == "cpu"
    assert rc1.device == rc2.device == rc3.device == rc4.device == "cuda"
    assert parse_args(["--version", "1.4.0rc2", "--device", "cpu"]).device == "cpu"
    with pytest.raises(SystemExit):
        parse_args(["--version", "1.4.0rc2", "--candidate-version", "1.4.0rc1"])


def test_streams_stderr_and_stdout_and_skips_completed_case(tmp_path, capsys):
    script = tmp_path / "fake_worker.py"
    script.write_text("import sys\nprint('model-start', flush=True)\nprint('candidate-warning', file=sys.stderr, flush=True)\n")
    case = tmp_path / "case"
    spec = dict(version="1.3.2", dataset="credit-g", n_jobs=1, device="cpu", seconds=1,
                affinity=[min(os.sched_getaffinity(0))])
    result = execute(sys.executable, script, case, spec, grace=5, evaluation_seconds=5, memory_gib=2)
    assert result["status"] == "success"
    logged = (Path(result["attempt"]) / "console.log").read_text()
    assert "model-start" in logged and "candidate-warning" in logged
    assert "candidate-warning" in capsys.readouterr().out
    again = execute(sys.executable, script, case, spec, grace=5, evaluation_seconds=5, memory_gib=2)
    assert again == result
    assert len(list(case.glob("attempt-*"))) == 1


def test_timeout_retains_partial_output_and_interrupted_retry(tmp_path):
    script = tmp_path / "fake_worker.py"
    script.write_text("import time\nprint('partial model', flush=True)\ntime.sleep(30)\n")
    case = tmp_path / "case"
    spec = dict(version="1.3.2", dataset="credit-g", n_jobs=1, device="cpu", seconds=1,
                affinity=[min(os.sched_getaffinity(0))])
    result = execute(sys.executable, script, case, spec, grace=0, evaluation_seconds=1, memory_gib=2)
    assert result["status"] == "timeout"
    old_log = Path(result["attempt"]) / "console.log"
    assert "partial model" in old_log.read_text()
    write_json(case / "status.json", dict(result, status="interrupted"))
    script.write_text("print('retry', flush=True)\n")
    retry = execute(sys.executable, script, case, spec, grace=5, evaluation_seconds=5, memory_gib=2)
    assert retry["status"] == "success"
    assert retry["attempt"] != result["attempt"]
    assert "partial model" in old_log.read_text()


def test_run_rejects_changed_budget(tmp_path):
    freeze_run(tmp_path, {"seconds": 60})
    with pytest.raises(ValueError, match="Configuration"):
        freeze_run(tmp_path, {"seconds": 3600})


@pytest.mark.parametrize("candidate", ["1.4.0rc1", "1.4.0rc2", "1.4.0rc3", "1.4.0rc4"])
def test_report_compares_only_matching_successful_splits(tmp_path, candidate):
    config = {"datasets": ["credit-g"]}
    if candidate != "1.4.0rc1":
        config["comparison_versions"] = ["1.3.2", candidate]
    write_json(tmp_path / "manifest.json", {"config": config})
    for version, score in [("1.3.2", .7), (candidate, .8)]:
        case = tmp_path / version / "credit-g/r0f0"
        attempt = case / "attempt-001"
        write_json(case / "status.json", dict(dataset="credit-g", version=version, status="success", attempt=str(attempt)))
        write_json(attempt / "metrics.json", dict(dataset_sha256="same", metric="roc_auc", split="r0f0",
                   seed=1234, seconds=60, n_jobs=1, test_score=score, metric_error=1-score))
    build_report(tmp_path)
    with (tmp_path / "comparison/paired.csv").open() as stream:
        row = list(csv.DictReader(stream))[0]
        assert row["winner"] == candidate
        assert float(row[f"score_{candidate.replace('.', '_')}"]) == .8
    assert f"{candidate} wins: 1" in (tmp_path / "comparison/report.md").read_text()
    metrics_path = tmp_path / candidate / "credit-g/r0f0/attempt-001/metrics.json"
    write_json(metrics_path, dict(read_json(metrics_path), dataset_sha256="different"))
    with pytest.raises(ValueError, match="dataset_sha256"):
        build_report(tmp_path)


def test_report_can_read_old_baseline_without_changing_it(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    config = dict(datasets=["credit-g"], seconds=3600, n_jobs=15, algorithms=None)
    write_json(old / "manifest.json", {"config": config})
    write_json(new / "manifest.json", {"config": dict(config, comparison_versions=["1.3.2", "1.4.0rc2"])})
    for root, version, score in [(old, "1.3.2", .7), (new, "1.4.0rc2", .8)]:
        case = root / version / "credit-g/r0f0"
        attempt = case / "attempt-001"
        write_json(case / "status.json", dict(dataset="credit-g", version=version, status="success", attempt=str(attempt)))
        write_json(attempt / "metrics.json", dict(dataset_sha256="same", metric="roc_auc", split="r0f0",
                   seed=1234, seconds=3600, n_jobs=15, test_score=score, metric_error=1-score))
    previous = {p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    build_report(new, baseline_run=old)
    with (new / "comparison/paired.csv").open() as stream:
        assert list(csv.DictReader(stream))[0]["winner"] == "1.4.0rc2"
    assert previous == {p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    write_json(new / "manifest.json", {"config": dict(config, algorithms=["Linear"])})
    with pytest.raises(ValueError, match="algorithms"):
        build_report(new, baseline_run=old)


def test_report_retains_multiple_versions_and_excludes_incomplete_scores(tmp_path):
    config = dict(datasets=["credit-g"], seconds=3600, n_jobs=15, seed=1234)
    roots = {v: tmp_path / v for v in ["1.3.2", "1.4.0rc2", "1.4.0rc3", "1.4.0rc4"]}
    for version, root in roots.items():
        write_json(root / "manifest.json", {"config": dict(config, comparison_versions=["1.3.2", version if version != "1.3.2" else "1.4.0rc1"])})
        case = root / version / "credit-g/r0f0"
        attempt = case / "attempt-001"
        write_json(case / "status.json", dict(dataset="credit-g", version=version,
                   status="timeout" if version == "1.4.0rc3" else "success", attempt=str(attempt)))
        write_json(attempt / "metrics.json", dict(dataset_sha256="same", metric="roc_auc", split="r0f0",
                   seed=1234, seconds=3600, n_jobs=15, test_score=.8, metric_error=.2))
    current = roots["1.4.0rc4"]
    manifest = read_json(current / "manifest.json")
    manifest["config"].update(baseline_run=str(roots["1.3.2"]), reference_runs=[str(roots[v]) for v in ["1.4.0rc2", "1.4.0rc3"]])
    write_json(current / "manifest.json", manifest)
    previous = {str(p): p.read_bytes() for v, root in roots.items() if v != "1.4.0rc4" for p in root.rglob("*") if p.is_file()}
    build_report(current)
    with (current / "comparison/datasets.csv").open() as stream:
        assert {r["version"] for r in csv.DictReader(stream)} == set(roots)
    with (current / "comparison/paired.csv").open() as stream:
        assert {r["candidate_version"] for r in csv.DictReader(stream)} == {"1.4.0rc2", "1.4.0rc4"}
    assert all(Path(p).read_bytes() == content for p, content in previous.items())
    old_manifest = roots["1.4.0rc3"] / "manifest.json"
    changed = read_json(old_manifest)
    changed["config"]["seconds"] = 60
    write_json(old_manifest, changed)
    with pytest.raises(ValueError, match="seconds"):
        build_report(current)
