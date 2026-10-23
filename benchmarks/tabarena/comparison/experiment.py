"""Train controlled stacking variants from frozen local code in an existing release environment.

Each variant gets its own outputs. Original release models and environments are
read only. This is an experiment, not a published-version benchmark result.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import shutil

from common import BASE, ROOT, SMALL, lock, now, read_json, sha256, write_json
from reevaluate import tree_hashes, verify_hashes
from run import execute

HERE = Path(__file__).resolve().parent
FOCUSED = ["credit-g", "anneal", "polish_companies_bankruptcy", "Amazon_employee_access"]


def freeze(run, source_results, datasets, seconds, limits):
    config = dict(source_results=str(source_results), datasets=datasets,
                  seconds=seconds, stacking_limits=limits)
    manifest_file = run / "manifest.json"
    if manifest_file.exists():
        manifest = read_json(manifest_file)
        if manifest["config"] != config:
            raise ValueError("Experiment settings changed; use another output directory")
        verify_hashes(run / "source/supervised", manifest["source_sha256"])
        verify_hashes(run / "code", manifest["runner_sha256"])
        return manifest
    if any(p.name != "run.lock" for p in run.iterdir()):
        raise ValueError("Output contains an incomplete snapshot; use another directory")
    cases = {}
    for dataset in datasets:
        state = read_json(source_results / dataset / "r0f0/status.json")
        if state["status"] not in {"success", "error", "timeout", "memory_limit"}:
            raise ValueError(f"Source benchmark is not terminal: {dataset}")
        case = Path(state["attempt"]) / "case.json"
        if not case.resolve().is_relative_to(source_results.resolve()):
            raise ValueError("Source attempt is outside source results")
        spec = read_json(case)
        cases[dataset] = dict(spec=spec, case_sha256=sha256(case))
    shutil.copytree(ROOT / "supervised", run / "source/supervised",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (run / "code").mkdir()
    for path in HERE.glob("*.py"):
        shutil.copy2(path, run / "code" / path.name)
    environment = read_json(source_results / "environment.json")
    if not Path(environment["python"]).is_file():
        raise ValueError("Original Python environment is unavailable")
    manifest = dict(config=config, cases=cases, environment=environment,
                    created=now(), source_sha256=tree_hashes(run / "source/supervised"),
                    runner_sha256=tree_hashes(run / "code"))
    write_json(manifest_file, manifest)
    return manifest


def report(run, manifest):
    rows = []
    for limit in manifest["config"]["stacking_limits"]:
        for dataset in manifest["config"]["datasets"]:
            state_file = run / f"top-{limit}" / dataset / "r0f0/status.json"
            if not state_file.exists():
                continue
            state = read_json(state_file)
            attempt = Path(state["attempt"])
            row = dict(dataset=dataset, stacking_limit=limit, status=state["status"],
                       attempt=str(attempt))
            if (attempt / "metrics.json").exists():
                row.update(read_json(attempt / "metrics.json"))
            if (attempt / "automl/params.json").exists():
                params = read_json(attempt / "automl/params.json")
                row["stack_inputs"] = len(params.get("stacked", []))
                best = params.get("best_model")
                if best:
                    # Internal minimized validation loss only. Held-out scores
                    # are reported but must not select the stacking policy.
                    framework = attempt / "automl" / best / "framework.json"
                    ensemble = attempt / "automl" / best / "ensemble.json"
                    if framework.exists():
                        row["validation_loss"] = read_json(framework).get("final_loss")
                    elif ensemble.exists():
                        row["validation_loss"] = read_json(ensemble).get("final_loss")
            rows.append(row)
    write_json(run / "results.json", rows)
    text = ["# Stacking experiment", "",
            "Frozen local source; original release environment and official r0f0 splits. "
            "All trained models are retained. These are experimental results.", "",
            "| Dataset | Inputs/type | Status | Stack inputs | Validation loss | Test score | Prediction seconds |",
            "|---|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        text.append("| " + " | ".join(str(row.get(key, "—")) for key in (
            "dataset", "stacking_limit", "status", "stack_inputs", "validation_loss",
            "test_score", "time_infer_s")) + " |")
    (run / "report.md").write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=SMALL, default=FOCUSED)
    parser.add_argument("--stacking-limits", nargs="+", type=int, choices=[1, 2], default=[1, 2])
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--evaluation-seconds", type=int, default=1800)
    parser.add_argument("--memory-gib", type=float, default=96)
    args = parser.parse_args()
    args.run, args.source_results = args.run.resolve(), args.source_results.resolve()
    if min(args.seconds, args.evaluation_seconds, args.memory_gib) <= 0:
        parser.error("Budgets must be positive")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.stacking_limits)) != len(args.stacking_limits):
        parser.error("Datasets and stacking limits must be unique")
    for protected in (args.source_results, ROOT / "supervised", BASE / "comparison-environments",
                      BASE / "comparison-data"):
        if args.run.is_relative_to(protected) or protected.is_relative_to(args.run):
            parser.error("Output must be separate from existing artifacts/source/environments")
    with ExitStack() as locks:
        locks.enter_context(lock(args.run / "run.lock", wait=False))
        manifest = freeze(args.run, args.source_results, args.datasets, args.seconds, args.stacking_limits)
        print(f"Frozen source: {args.run / 'source'}; waiting for benchmark resources", flush=True)
        for path in (BASE / "comparison-environments/candidate.run.lock",
                     BASE / "comparison-environments/gpu.lock"):
            locks.enter_context(lock(path, wait=True))
        states = []
        try:
            for limit in args.stacking_limits:
                for dataset in args.datasets:
                    spec = dict(manifest["cases"][dataset]["spec"], seconds=args.seconds,
                                evaluate_all_models=False, stacking_limit=limit,
                                source_snapshot=str(args.run / "source"),
                                source_sha256=manifest["source_sha256"],
                                experiment=f"local fixes / top-{limit}")
                    states.append(execute(
                        manifest["environment"]["python"], args.run / "code/worker.py",
                        args.run / f"top-{limit}" / dataset / "r0f0", spec,
                        grace=900, evaluation_seconds=args.evaluation_seconds,
                        memory_gib=args.memory_gib,
                    ))
                    report(args.run, manifest)
        finally:
            report(args.run, manifest)
        return 0 if all(state["status"] == "success" for state in states) else 1


if __name__ == "__main__":
    raise SystemExit(main())
