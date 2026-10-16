"""Build a local paired comparison without loading models or imputing failures."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from common import lock, read_json


def write_csv(path, rows):
    if not rows:
        path.unlink(missing_ok=True)
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_reference(manifest, reference):
    for key in ["tabarena_revision", "protocol"]:
        if manifest.get(key) != reference.get(key):
            raise ValueError(f"Cannot reuse results: differing {key}")
    for key in ["seconds", "n_jobs", "seed", "algorithms", "grace_seconds", "evaluation_seconds", "memory_gib"]:
        if manifest["config"].get(key) != reference["config"].get(key):
            raise ValueError(f"Cannot reuse results: differing {key}")


def build_report(run, baseline_run=None, reference_runs=None):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    # Older frozen runs predate explicit comparison_versions; their pair was RC1.
    versions = list(manifest["config"].get("comparison_versions", ["1.3.2", "1.4.0rc1"]))
    stable, candidate = versions
    sources = {version: run for version in versions}
    baseline_run = baseline_run or manifest["config"].get("baseline_run")
    if reference_runs is None:
        reference_runs = manifest["config"].get("reference_runs", [])
    if baseline_run is not None:
        baseline_run = Path(baseline_run).resolve()
        baseline = read_json(baseline_run / "manifest.json")
        validate_reference(manifest, baseline)
        if not (baseline_run / stable).is_dir():
            raise ValueError(f"No {stable} results in baseline run: {baseline_run}")
        sources[stable] = baseline_run
    for reference_run in reference_runs:
        reference_run = Path(reference_run).resolve()
        reference = read_json(reference_run / "manifest.json")
        validate_reference(manifest, reference)
        reference_version = reference["config"].get("comparison_versions", [stable, "1.4.0rc1"])[1]
        if reference_version in sources:
            if sources[reference_version].resolve() != reference_run:
                raise ValueError(f"Multiple result sources for {reference_version}")
            continue
        if not (reference_run / reference_version).is_dir():
            raise ValueError(f"No {reference_version} results in reference run: {reference_run}")
        sources[reference_version] = reference_run
    versions = [stable, *sorted(v for v in sources if v != stable)]
    out = run / "comparison"
    out.mkdir(exist_ok=True)
    with lock(out / "report.lock"):
        cases, models = [], []
        for version in versions:
            for status_file in sorted((sources[version] / version).glob("*/r0f0/status.json")):
                status = read_json(status_file)
                if status["dataset"] not in manifest["config"]["datasets"]:
                    continue
                attempt = Path(status["attempt"])
                metrics = read_json(attempt / "metrics.json") if (attempt / "metrics.json").exists() else {}
                row = dict(status, **{k: v for k, v in metrics.items() if k not in status})
                cases.append(row)
                if (attempt / "model_inventory.csv").exists():
                    with (attempt / "model_inventory.csv").open() as stream:
                        models.extend(dict(record, dataset=status["dataset"], version=version)
                                      for record in csv.DictReader(stream))
        paired = []
        for dataset in manifest["config"]["datasets"]:
            matches = {r["version"]: r for r in cases if r["dataset"] == dataset and r["status"] == "success"}
            if stable not in matches:
                continue
            for compared in versions[1:]:
                if compared not in matches:
                    continue
                old, new = matches[stable], matches[compared]
                for key in ["dataset_sha256", "metric", "split", "seed", "seconds", "n_jobs"]:
                    if old[key] != new[key]:
                        raise ValueError(f"Cannot compare {dataset}: differing {key}")
                delta = new["metric_error"] - old["metric_error"]
                winner = "tie" if math.isclose(old["metric_error"], new["metric_error"], rel_tol=1e-7, abs_tol=1e-9) else (compared if delta < 0 else stable)
                error_column = f"error_change_rc{compared.rsplit('rc', 1)[-1]}_minus_stable"
                paired.append(dict(dataset=dataset, candidate_version=compared, metric=old["metric"], score_1_3_2=old["test_score"],
                                   **{f"score_{compared.replace('.', '_')}": new["test_score"], error_column: delta}, winner=winner))
        write_csv(out / "datasets.csv", cases)
        write_csv(out / "models.csv", models)
        write_csv(out / "paired.csv", paired)
        text = ["# Published MLJAR version comparison", "",
                "Official TabArena-v0.1 r0f0 splits; MLJAR owns internal validation and model selection.",
                "Test metrics are separate from the internal leaderboard. No missing/failed results are imputed.",
                "Scores from incomplete cases are provisional and excluded from paired win counts. Evaluation mode can differ; all model artifacts are retained.",
                "GPU allocations differ between versions; timings and scores reflect those resource choices.",
                "This subset is not the full TabArena leaderboard. External comparisons must match dataset, split and metric and disclose hardware/budgets.", "",
                *[f"{version} results: `{sources[version]}`." for version in versions], "",
                "| Version | Dataset | Status | Test metric | Test score | Selected model |",
                "|---|---|---|---|---:|---|"]
        for row in cases:
            text.append(f"| {row['version']} | {row['dataset']} | {row['status']} | {row.get('metric', '')} | {row.get('test_score', '')} | {row.get('best_model', '')} |")
        for compared in versions[1:]:
            matching = [r for r in paired if r["candidate_version"] == compared]
            text += ["", f"{compared}: matched successful datasets: {len(matching)}.",
                     f"{compared} wins: {sum(r['winner'] == compared for r in matching)}; "
                     f"stable wins: {sum(r['winner'] == stable for r in matching)}; "
                     f"ties: {sum(r['winner'] == 'tie' for r in matching)}."]
        text += ["", "See datasets.csv, models.csv and paired.csv. Model inventory includes all saved models, including random-feature candidates."]
        temporary = out / "report.md.tmp"
        temporary.write_text("\n".join(text) + "\n")
        temporary.replace(out / "report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path,
                        help="Read 1.3.2 results from an earlier run without copying or changing its files")
    parser.add_argument("--reference-run", type=Path, action="append",
                        help="Read another candidate run; repeat for multiple versions")
    args = parser.parse_args()
    build_report(args.run, args.baseline_run, args.reference_run)
