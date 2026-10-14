"""Safeguards for evaluation without retraining or replacing original artifacts."""

import os
from pathlib import Path
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from common import read_json, write_json
import reevaluate
from reevaluate import (
    freeze,
    input_attempt,
    paired_scores,
    parse_args,
    tree_hashes,
    verify_hashes,
)
from reevaluate_worker import assert_unloaded, encode_labels
from run import execute
from worker import score_predictions


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    root, source, baseline, output = [
        tmp_path / name for name in ("repo", "rc2", "stable", "new")
    ]
    (root / "supervised").mkdir(parents=True)
    (root / "supervised/__init__.py").write_text("__version__ = '1.4.0rc2'\n")
    (root / "supervised/uncommitted.py").write_text("fixed = True\n")
    for dataset in reevaluate.SMALL:
        attempt = source / dataset / "r0f0/attempt-001"
        write_json(attempt / "case.json", dict(version="1.4.0rc2", dataset=dataset))
        write_json(
            attempt / "automl/params.json",
            dict(fit_level="finished", best_model="Ensemble"),
        )
        write_json(
            attempt.parent / "status.json", dict(status="error", attempt=str(attempt))
        )
    write_json(source / "environment.json", dict(python=sys.executable, packages={}))
    (source / "requirements-frozen.txt").write_text("fixture\n")
    monkeypatch.setattr(reevaluate, "ROOT", root)
    monkeypatch.setattr(
        reevaluate.subprocess,
        "check_output",
        lambda *a, **kw: "head" if kw.get("text") else b"diff",
    )
    return root, source, baseline, output


def test_snapshot_includes_uncommitted_files_and_resumes_frozen_code(saved_run):
    root, source, baseline, output = saved_run
    before = tree_hashes(source)
    manifest = freeze(output, source, baseline, 7200, 96)
    assert "supervised/uncommitted.py" in manifest["source_sha256"]
    (root / "supervised/uncommitted.py").write_text("fixed = 'later edit'\n")
    assert freeze(output, source, baseline, 7200, 96) == manifest
    assert (output / "source/supervised/uncommitted.py").read_text() == "fixed = True\n"
    assert tree_hashes(source) == before
    with pytest.raises(ValueError, match="configuration changed"):
        freeze(output, source, baseline, 1, 96)
    (output / "source/supervised/uncommitted.py").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="Files changed"):
        freeze(output, source, baseline, 7200, 96)


def test_unfinished_source_is_rejected(saved_run):
    _, source, _, _ = saved_run
    path, _ = input_attempt(source, "anneal")
    write_json(
        path / "automl/params.json", dict(fit_level="stack", best_model="Ensemble")
    )
    with pytest.raises(ValueError, match="finished RC2 fit"):
        input_attempt(source, "anneal")


def test_hash_verification_detects_new_and_modified_artifacts(tmp_path):
    (tmp_path / "model").write_bytes(b"original")
    expected = tree_hashes(tmp_path)
    (tmp_path / "extra").write_bytes(b"added")
    with pytest.raises(RuntimeError, match="extra"):
        verify_hashes(tmp_path, expected)
    (tmp_path / "extra").unlink()
    (tmp_path / "model").write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="model"):
        verify_hashes(tmp_path, expected)


def test_saved_class_order_is_used_and_unknown_labels_are_rejected():
    assert encode_labels(pd.Series(["10", "2", "1"]), ["1", "10", "2"]).tolist() == [
        1,
        2,
        0,
    ]
    assert encode_labels(pd.Series([2, 0]), [0, 1, 2]).tolist() == [2, 0]
    with pytest.raises(ValueError, match="absent"):
        encode_labels(pd.Series(["new"]), ["old"])


def test_float32_probability_rounding_does_not_warn_or_change_logloss():
    from sklearn.metrics import log_loss

    values = np.array([[0.3, 0.3, 0.4], [0.6, 0.3, 0.1]], dtype=np.float32)
    frame = pd.DataFrame(values, columns=[f"prediction_{i}" for i in range(3)])
    normalized = values.astype(float)
    normalized /= normalized.sum(axis=1, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        score, _ = score_predictions(frame, [2, 0], "multiclass", 3)
    assert score == log_loss([2, 0], normalized, labels=[0, 1, 2])


def test_unfitted_placeholders_are_allowed_but_predicted_models_must_release():
    from supervised.algorithms.decision_tree import DecisionTreeAlgorithm

    learner = DecisionTreeAlgorithm({"ml_task": "binary_classification"})
    assert learner.model is not None and not learner.is_fitted()
    model = SimpleNamespace(learners=[learner], get_name=lambda: "tree")
    automl = SimpleNamespace(_models=[model])
    assert_unloaded(automl)
    with pytest.raises(RuntimeError, match="resident"):
        assert_unloaded(automl, {"tree"})
    learner.model.fit([[0], [1]], [0, 1])
    with pytest.raises(RuntimeError, match="resident"):
        assert_unloaded(automl)
    learner.model = None
    assert_unloaded(automl, {"tree"})


def test_pairs_require_matching_protocol_and_flag_partial_baseline():
    metrics = dict(
        dataset="example",
        task_id=1,
        split="r0f0",
        metric="roc_auc",
        dataset_sha256="data",
        seed=1234,
        seconds=3600,
        n_jobs=15,
        metric_error=0.1,
        reload_verified=True,
    )
    baseline = dict(metrics, metric_error=0.2)
    state = dict(status="success")
    assert paired_scores(metrics, state, baseline, state)["winner"] == "patched"
    assert (
        paired_scores(metrics, state, baseline, dict(status="timeout"))["comparison"]
        == "provisional_baseline"
    )
    assert (
        paired_scores(metrics, state, dict(baseline, dataset_sha256="other"), state)[
            "comparison"
        ]
        == "unpaired"
    )
    assert (
        paired_scores(dict(metrics, reload_verified=False), state, baseline, state)[
            "comparison"
        ]
        == "unpaired"
    )


def test_output_cannot_overlap_original_models(tmp_path):
    source = tmp_path / "models"
    with pytest.raises(SystemExit):
        parse_args(["--source", str(source), "--run", str(source / "new")])
    with pytest.raises(SystemExit):
        parse_args(["--source", str(source), "--run", str(tmp_path)])


def test_retry_retains_failed_attempt_and_never_repeats_success(tmp_path):
    script = tmp_path / "fake_worker.py"
    script.write_text("raise RuntimeError('prediction failed')\n")
    case = tmp_path / "case"
    spec = dict(
        version="patched",
        dataset="example",
        n_jobs=1,
        device="cpu",
        seconds=1,
        affinity=[min(os.sched_getaffinity(0))],
    )
    options = dict(grace=5, evaluation_seconds=5, memory_gib=2)
    first = execute(sys.executable, script, case, spec, **options)
    assert first["status"] == "error"
    before = tree_hashes(first["attempt"])
    script.write_text("print('recovered', flush=True)\n")
    assert execute(sys.executable, script, case, spec, **options) == first
    second = execute(sys.executable, script, case, spec, retry_failed=True, **options)
    assert second["status"] == "success" and second["attempt"] != first["attempt"]
    assert tree_hashes(first["attempt"]) == before
    assert (
        execute(sys.executable, script, case, spec, retry_failed=True, **options)
        == second
    )


def test_report_keeps_pending_and_failed_datasets_visible(saved_run):
    _, source, baseline, output = saved_run
    freeze(output, source, baseline, 7200, 96)
    attempt = output / "results/anneal/r0f0/attempt-001"
    write_json(
        attempt.parent / "status.json", dict(status="error", attempt=str(attempt))
    )
    rows = reevaluate.build_report(output)
    assert len(rows) == 12
    assert sum(row["status"] == "pending" for row in rows) == 11
    assert (
        next(row for row in rows if row["dataset"] == "anneal")["comparison"]
        == "unpaired"
    )
    assert "0/12" in (output / "comparison/report.md").read_text()
