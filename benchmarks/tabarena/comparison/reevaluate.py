"""Evaluate saved release models with a frozen snapshot of local inference code; never fit."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess

from common import BASE, ROOT, SMALL, lock, now, read_json, sha256, write_json
from run import execute

HERE = Path(__file__).resolve().parent
VERSION = "1.4.0rc2"
LABEL = "RC2 models + patched inference"
RUNNER_FILES = (
    "common.py",
    "run.py",
    "setup_env.py",
    "worker.py",
    "reevaluate.py",
    "reevaluate_worker.py",
)


def tree_hashes(root):
    root = Path(root)
    return {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


def verify_hashes(root, expected):
    actual = tree_hashes(root)
    if actual != expected:
        changed = sorted(
            k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k)
        )
        raise RuntimeError(f"Files changed under {root}: {changed[:10]}")


def input_attempt(root, dataset):
    state = read_json(root / dataset / "r0f0/status.json")
    if state["status"] in {"running", "interrupted"}:
        raise ValueError(f"Source run is not terminal: {dataset} ({state['status']})")
    attempt = Path(state["attempt"]).resolve()
    if not attempt.is_relative_to(root.resolve()):
        raise ValueError(f"Source attempt is outside {root}: {attempt}")
    spec = read_json(attempt / "case.json")
    params = read_json(attempt / "automl/params.json")
    if (
        spec["version"] != VERSION
        or spec["dataset"] != dataset
        or params.get("fit_level") != "finished"
        or not params.get("best_model")
    ):
        raise ValueError(f"Expected a finished {VERSION} fit for {dataset}")
    return attempt, spec


def freeze(run, source, baseline, evaluation_seconds, memory_gib):
    """Resume the stored source snapshot even if the working tree changes later."""
    config = dict(
        source=str(source),
        baseline=str(baseline),
        datasets=SMALL,
        evaluation_seconds=evaluation_seconds,
        memory_gib=memory_gib,
    )
    run.mkdir(parents=True, exist_ok=True)
    manifest_file = run / "manifest.json"
    if manifest_file.exists():
        manifest = read_json(manifest_file)
        if manifest["config"] != config:
            raise ValueError(
                "Reevaluation configuration changed; use a new --run directory"
            )
        verify_hashes(run / "source", manifest["source_sha256"])
        verify_hashes(run / "code", manifest["runner_sha256"])
        current_runner = {name: sha256(HERE / name) for name in RUNNER_FILES}
        if current_runner != manifest["runner_sha256"]:
            raise ValueError(
                "Runner changed; resume with the saved code/reevaluate.py or use a new --run"
            )
        return manifest
    if any(run.iterdir()):
        # run.lock is held by main; an interrupted freeze must not reuse partial code.
        if any(p.name != "run.lock" for p in run.iterdir()):
            raise ValueError("Incomplete snapshot; use a new --run directory")
    cases = {}
    for dataset in SMALL:
        attempt, spec = input_attempt(source, dataset)
        cases[dataset] = dict(
            source_attempt=str(attempt),
            original_spec=spec,
            case_sha256=sha256(attempt / "case.json"),
            params_sha256=sha256(attempt / "automl/params.json"),
        )
    env = read_json(source / "environment.json")
    if not Path(env["python"]).is_file():
        raise ValueError(f"Original environment is unavailable: {env['python']}")
    shutil.copytree(
        ROOT / "supervised",
        run / "source/supervised",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (run / "code").mkdir()
    for name in RUNNER_FILES:
        shutil.copy2(HERE / name, run / "code" / name)
    for name in ("environment.json", "requirements-frozen.txt"):
        shutil.copy2(source / name, run / name)
    source_hashes = tree_hashes(run / "source")
    digest = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode()
    ).hexdigest()
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    (run / "working-tree.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--", "supervised"], cwd=ROOT)
    )
    manifest = dict(
        label=LABEL,
        created=now(),
        config=config,
        cases=cases,
        source_sha256=source_hashes,
        source_digest=digest,
        runner_sha256=tree_hashes(run / "code"),
        git_head=git_head,
        environment_sha256=sha256(run / "environment.json"),
        requirements_sha256=sha256(run / "requirements-frozen.txt"),
    )
    write_json(manifest_file, manifest)
    return manifest


def paired_scores(metrics, state, baseline_metrics, baseline_state):
    """A completed selected prediction can survive a later all-model timeout."""
    keys = (
        "dataset",
        "task_id",
        "split",
        "metric",
        "dataset_sha256",
        "seed",
        "seconds",
        "n_jobs",
    )
    matching = bool(baseline_metrics) and all(
        key in metrics and metrics[key] == baseline_metrics.get(key) for key in keys
    )
    ready = (
        state.get("status") == "success"
        and metrics.get("reload_verified")
        and matching
        and "metric_error" in metrics
        and "metric_error" in baseline_metrics
    )
    if not ready:
        return dict(comparison="unpaired", error_delta=None, winner=None)
    delta = metrics["metric_error"] - baseline_metrics["metric_error"]
    strict = baseline_state.get("status") == "success" and baseline_metrics.get(
        "reload_verified"
    )
    return dict(
        comparison="verified" if strict else "provisional_baseline",
        error_delta=delta,
        winner="patched" if delta < 0 else "1.3.2" if delta > 0 else "tie",
    )


def build_report(run):
    manifest = read_json(run / "manifest.json")
    baseline = Path(manifest["config"]["baseline"])
    rows = []
    for dataset in manifest["config"]["datasets"]:
        state_path = run / "results" / dataset / "r0f0/status.json"
        state = read_json(state_path) if state_path.exists() else {"status": "pending"}
        attempt = Path(state["attempt"]) if "attempt" in state else None
        metrics = (
            read_json(attempt / "metrics.json")
            if attempt and (attempt / "metrics.json").exists()
            else {}
        )
        baseline_path = baseline / dataset / "r0f0/status.json"
        base_state = read_json(baseline_path) if baseline_path.exists() else {}
        base_attempt = Path(base_state["attempt"]) if "attempt" in base_state else None
        base_metrics = (
            read_json(base_attempt / "metrics.json")
            if base_attempt and (base_attempt / "metrics.json").exists()
            else {}
        )
        pair = paired_scores(metrics, state, base_metrics, base_state)
        rows.append(
            dict(
                dataset=dataset,
                status=state["status"],
                metric=metrics.get("metric", base_metrics.get("metric")),
                baseline_score=base_metrics.get("test_score"),
                patched_score=metrics.get("test_score"),
                baseline_status=base_state.get("status"),
                **pair,
                prediction_seconds=metrics.get("time_infer_s"),
                peak_rss_gib=state.get("peak_rss_gib"),
                peak_gpu_allocated_gib=metrics.get("gpu", {}).get("peak_allocated_gib"),
                reload_verified=metrics.get("reload_verified", False),
                selected_model=metrics.get("best_model"),
                result_path=str(attempt) if attempt else "",
                baseline_path=str(base_attempt) if base_attempt else "",
            )
        )
    output = run / "comparison"
    output.mkdir(exist_ok=True)
    temporary = output / "datasets.csv.tmp"
    with temporary.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "datasets.csv")
    successes = sum(r["status"] == "success" for r in rows)
    lines = [
        f"# 1.3.2 vs {LABEL}",
        "",
        f"Selected models reevaluated successfully: {successes}/{len(rows)}.",
        f"Frozen inference source: `{manifest['source_digest']}`.",
        "",
        "These are saved RC2 fits evaluated with patched code, not published RC2 or newly trained models.",
        "Scores use the original full held-out r0f0 splits; lower error delta favors patched inference.",
        "Provisional baseline means its selected score exists but the original evaluation did not finish successfully.",
        "Inference timings use different loading/cache conditions from the original baseline; they are diagnostic, not a speedup claim.",
        "",
        "| Dataset | Status | Metric | 1.3.2 | Patched | Error delta | Pair |",
        "|---|---|---|---:|---:|---:|---|",
    ]

    def value(x):
        return "—" if x is None else f"{x:.8g}"

    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['status']} | {row['metric'] or '—'} | "
            f"{value(row['baseline_score'])} | {value(row['patched_score'])} | "
            f"{value(row['error_delta'])} | {row['comparison']} |"
        )
    for kind in ("verified", "provisional_baseline"):
        pairs = [r for r in rows if r["comparison"] == kind]
        lines += [
            "",
            f"{kind}: {len(pairs)} pairs; patched wins {sum(r['winner'] == 'patched' for r in pairs)}, "
            f"1.3.2 wins {sum(r['winner'] == '1.3.2' for r in pairs)}, ties {sum(r['winner'] == 'tie' for r in pairs)}.",
        ]
    lines += [
        "",
        f"Saved models: `{manifest['config']['source']}`",
        f"Baseline: `{baseline}`",
        "",
    ]
    (output / "report.md.tmp").write_text("\n".join(lines))
    (output / "report.md.tmp").replace(output / "report.md")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-version", default=VERSION,
                        help="Published version that trained the saved models")
    parser.add_argument(
        "--source",
        type=Path,
        help="Saved version directory (defaults to the selected training version)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASE / "comparison-1.3.2-vs-1.4.0rc1-small-3600s-15threads/1.3.2",
    )
    parser.add_argument(
        "--run", type=Path
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SMALL,
        help="Subset to execute in the same saved run",
    )
    parser.add_argument(
        "--evaluation-seconds",
        type=int,
        default=7200,
        help="Per-dataset limit including reload verification",
    )
    parser.add_argument("--memory-gib", type=float, default=96)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Create new attempts for failed evaluations",
    )
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    args.source = args.source or BASE / f"comparison-1.3.2-vs-{args.training_version}-small-3600s-15threads/{args.training_version}"
    args.run = args.run or BASE / f"reevaluation-{args.training_version}-prediction-fix"
    args.datasets = args.datasets or SMALL
    args.run, args.source, args.baseline = (
        args.run.resolve(),
        args.source.resolve(),
        args.baseline.resolve(),
    )
    if min(args.evaluation_seconds, args.memory_gib) <= 0 or len(
        set(args.datasets)
    ) != len(args.datasets):
        parser.error("Budgets must be positive and datasets unique")
    protected = (
        args.source,
        args.baseline,
        ROOT / "supervised",
        BASE / "comparison-data",
        BASE / "comparison-environments",
    )
    if any(args.run.is_relative_to(p) or p.is_relative_to(args.run) for p in protected):
        parser.error(
            "Output must be separate from model, data, environment and source directories"
        )
    return args


def main():
    global VERSION, LABEL
    args = parse_args()
    VERSION = args.training_version
    LABEL = f"{VERSION} models + patched inference"
    if args.report_only:
        build_report(args.run)
        return 0

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    with contextlib.ExitStack() as locks:
        for path in (
            args.run / "run.lock",
            BASE / f"comparison-environments/{VERSION}.run.lock",
            BASE / "comparison-environments/candidate.run.lock",
            BASE / "comparison-environments/gpu.lock",
        ):
            locks.enter_context(lock(path, wait=False))
        manifest = freeze(
            args.run,
            args.source,
            args.baseline,
            args.evaluation_seconds,
            args.memory_gib,
        )
        verify_hashes(args.run / "source", manifest["source_sha256"])
        if sha256(args.run / "environment.json") != manifest["environment_sha256"]:
            raise RuntimeError("Saved environment manifest changed")
        env = read_json(args.run / "environment.json")
        print(
            f"{LABEL}\nSource snapshot: {manifest['source_digest']}\nOutputs: {args.run}\n"
            f"{len(args.datasets)} datasets; original devices/thread counts; selected models only; no training.",
            flush=True,
        )
        write_json(
            args.run / "run_status.json",
            dict(
                status="running", pid=os.getpid(), started=now(), datasets=args.datasets
            ),
        )
        states, status = [], "interrupted"
        try:
            for dataset in args.datasets:
                source_case = manifest["cases"][dataset]
                attempt = Path(source_case["source_attempt"])
                if (
                    sha256(attempt / "case.json") != source_case["case_sha256"]
                    or sha256(attempt / "automl/params.json")
                    != source_case["params_sha256"]
                ):
                    raise RuntimeError(f"Source fit metadata changed for {dataset}")
                spec = dict(
                    source_case["original_spec"],
                    version=LABEL,
                    training_version=VERSION,
                    source_attempt=str(attempt),
                    snapshot=str(args.run),
                    label=LABEL,
                )
                states.append(
                    execute(
                        env["python"],
                        args.run / "code/reevaluate_worker.py",
                        args.run / "results" / dataset / "r0f0",
                        spec,
                        grace=0,
                        evaluation_seconds=args.evaluation_seconds,
                        memory_gib=args.memory_gib,
                        retry_failed=args.retry_failed,
                    )
                )
                build_report(args.run)
            status = (
                "success"
                if all(s["status"] == "success" for s in states)
                else "completed_with_failures"
            )
        finally:
            build_report(args.run)
            write_json(
                args.run / "run_status.json",
                dict(
                    status=status,
                    pid=os.getpid(),
                    finished=now(),
                    datasets=args.datasets,
                ),
            )
        print(
            f"Reevaluation {status}. Report: {args.run / 'comparison/report.md'}",
            flush=True,
        )
        return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
