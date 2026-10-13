"""Live, resumable comparison of published MLJAR releases on official TabArena splits."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

from common import (BASE, CANDIDATE_VERSIONS, DEFAULT_CANDIDATE, ROOT, SMALL,
                    SMOKE, STABLE_VERSION, TABARENA_REVISION, VERSIONS,
                    clean_environment, cpu_affinity, lock, now, read_json, sha256, write_json)
from setup_env import command, ensure_environment, ensure_tabarena

HERE = Path(__file__).resolve().parent
TERMINAL = {"success", "error", "timeout", "memory_limit"}


def stop(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def rss_bytes(pid):
    """Sample process-tree RSS on Linux (shared pages can be counted twice)."""
    total = 0
    pending = [pid]
    seen = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            total += int(Path(f"/proc/{current}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            pending.extend(map(int, Path(f"/proc/{current}/task/{current}/children").read_text().split()))
        except (OSError, IndexError, ValueError):
            pass
    return total


def execute(python, worker, case, spec, *, grace, evaluation_seconds, memory_gib):
    """Stream both output channels live, retaining every attempt and child process."""
    case.mkdir(parents=True, exist_ok=True)
    status_file = case / "status.json"
    if status_file.exists() and read_json(status_file)["status"] in TERMINAL:
        state = read_json(status_file)
        print(f"SKIP {spec['dataset']}: retained {state['status']} result at {case}", flush=True)
        return state
    attempts = sorted(case.glob("attempt-*"))
    attempt = case / f"attempt-{len(attempts) + 1:03d}"
    attempt.mkdir()
    spec = dict(spec, attempt=str(attempt))
    write_json(attempt / "case.json", spec)
    state = dict(dataset=spec["dataset"], version=spec["version"], status="running",
                 started=now(), attempt=str(attempt))
    write_json(status_file, state)
    started = time.monotonic()
    env = clean_environment(spec["n_jobs"], spec["device"])
    argv = ["taskset", "-c", ",".join(map(str, spec["affinity"])),
            str(python), "-u", str(worker), "--case", str(attempt / "case.json")]
    print(f"\nSTART {spec['version']} / {spec['dataset']} -> {attempt}\n", flush=True)
    peak, reason = 0, None
    with (attempt / "console.log").open("wb") as log, (attempt / "resources.jsonl").open("w") as resources:
        process = subprocess.Popen(argv, cwd=attempt, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        write_json(attempt / "pid.json", dict(pid=process.pid, started=now()))

        def tee():
            # Read bytes rather than lines: progress bars without newlines remain live.
            import codecs
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            while chunk := os.read(process.stdout.fileno(), 65536):
                log.write(chunk)
                log.flush()
                sys.stdout.write(decoder.decode(chunk))
                sys.stdout.flush()
            sys.stdout.write(decoder.decode(b"", final=True))
            sys.stdout.flush()

        reader = threading.Thread(target=tee, daemon=True)
        reader.start()
        last_heartbeat = -30
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                rss = rss_bytes(process.pid)
                peak = max(peak, rss)
                phase = read_json(attempt / "phase.json") if (attempt / "phase.json").exists() else {}
                phase_elapsed = time.monotonic() - phase.get("started_monotonic", started)
                limit = evaluation_seconds if phase.get("phase") == "evaluation" else spec["seconds"] + grace
                if phase_elapsed > limit:
                    reason = "timeout"
                if rss > memory_gib * 1024**3:
                    reason = "memory_limit"
                resources.write(json.dumps(dict(elapsed_s=elapsed, rss_bytes=rss, phase=phase.get("phase", "startup"))) + "\n")
                resources.flush()
                if elapsed - last_heartbeat >= 30:
                    print(f"[benchmark {spec['version']}] {spec['dataset']} | {phase.get('phase', 'startup')} | "
                          f"{elapsed:.0f}s elapsed | RSS {rss / 1024**3:.2f} GiB", flush=True)
                    last_heartbeat = elapsed
                if reason:
                    print(f"Stopping worker: {reason}; all partial files remain at {attempt}", flush=True)
                    stop(process)
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            reason = "interrupted"
            stop(process)
            raise
        finally:
            stop(process)
            reader.join(timeout=10)
            process.stdout.close()
            state.update(status=reason or ("success" if process.returncode == 0 else "error"),
                         finished=now(), exit_code=process.returncode,
                         wall_seconds=time.monotonic() - started, peak_rss_gib=peak / 1024**3)
            write_json(status_file, state)
            write_json(attempt / "status.json", state)
    print(f"END {state['status']}: {spec['dataset']} ({state['wall_seconds']:.1f}s)\n", flush=True)
    return state


def freeze_run(run, config):
    run.mkdir(parents=True, exist_ok=True)
    hashes = {p.name: sha256(p) for p in HERE.iterdir() if p.suffix in [".py", ".txt"]}
    with lock(run / "manifest.lock"):
        manifest_file = run / "manifest.json"
        if manifest_file.exists():
            manifest = read_json(manifest_file)
            if manifest["config"] != config or manifest["code_sha256"] != hashes:
                raise ValueError("Configuration or runner code differs from this saved run. Use a new --run directory.")
        else:
            code = run / "code"
            code.mkdir(exist_ok=True)
            for name in hashes:
                shutil.copy2(HERE / name, code / name)
            write_json(manifest_file, dict(created=now(), config=config, code_sha256=hashes,
                                          tabarena_revision=TABARENA_REVISION,
                                          protocol="TabArena-v0.1 system; official r0f0; automatic inner validation"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, choices=VERSIONS)
    parser.add_argument("--candidate-version", choices=CANDIDATE_VERSIONS,
                        help="Comparison partner for 1.3.2 (default: 1.4.0rc2)")
    parser.add_argument("--suite", choices=["small", "smoke"], default="small")
    parser.add_argument("--datasets", nargs="+", choices=SMALL, help="Explicit subset; changes the saved run configuration")
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--n-jobs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Default: CPU for 1.3.2, CUDA for release candidates")
    parser.add_argument("--run", type=Path, help="Shared output directory for both versions; existing results resume")
    parser.add_argument("--setup-only", action="store_true", help="Install and verify environments/weights; do not train")
    parser.add_argument("--prepare-only", action="store_true", help="Also prepare all selected datasets; do not train")
    parser.add_argument("--algorithms", nargs="+", help="Override automatic selection for diagnostic runs only")
    parser.add_argument("--grace-seconds", type=int, default=900)
    parser.add_argument("--evaluation-seconds", type=int, default=1800)
    parser.add_argument("--memory-gib", type=float, default=96)
    args = parser.parse_args(argv)
    if min(args.seconds, args.n_jobs, args.evaluation_seconds, args.memory_gib) <= 0 or args.grace_seconds < 0:
        parser.error("Budgets and n-jobs must be positive; grace-seconds must be nonnegative")
    candidate = args.candidate_version or (args.version if args.version in CANDIDATE_VERSIONS else DEFAULT_CANDIDATE)
    if args.version != STABLE_VERSION and args.version != candidate:
        parser.error("--candidate-version must match --version when running a release candidate")
    args.candidate_version = candidate
    args.device = args.device or ("cuda" if args.version in CANDIDATE_VERSIONS else "cpu")
    if args.version == STABLE_VERSION and args.device != "cpu":
        parser.error("This comparison uses the published 1.3.2 CPU configuration")
    args.datasets = args.datasets or (SMALL if args.suite == "small" else SMOKE)
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Datasets must be unique")
    args.run = (args.run or BASE / f"comparison-{STABLE_VERSION}-vs-{candidate}-{args.suite}-{args.seconds}s-{args.n_jobs}threads").resolve()
    return args


def main():
    args = parse_args()
    datasets, device, run = args.datasets, args.device, args.run
    affinity = cpu_affinity(args.version, args.n_jobs)
    print(f"MLJAR {args.version}; {len(datasets)} datasets; {args.seconds}s per fit; "
          f"{args.n_jobs} threads; {device}; CPU affinity {affinity}\nOutputs: {run}", flush=True)
    # Candidates share one CPU partition, even when using different environments.
    # Retain the version lock as well for compatibility with already-running jobs.
    with contextlib.ExitStack() as locks:
        locks.enter_context(lock(BASE / f"comparison-environments/{args.version}.run.lock", wait=False))
        if args.version in CANDIDATE_VERSIONS:
            locks.enter_context(lock(BASE / "comparison-environments/candidate.run.lock", wait=False))
        python = ensure_environment(args.version)
        environment_dir = python.parent.parent
        with (lock(BASE / "comparison-environments/gpu.lock", wait=False) if device == "cuda" else contextlib.nullcontext()):
            preflight_output = environment_dir / "preflight.json"
            command([python, "-u", HERE / "preflight.py", "--version", args.version,
                     "--device", device, "--output", preflight_output], env=clean_environment(args.n_jobs, device))
            if args.setup_only:
                return
            tabarena_python = ensure_tabarena()
            command([tabarena_python, "-u", HERE / "prepare.py", *datasets])
            if args.prepare_only:
                return
            config = dict(datasets=datasets, seconds=args.seconds, n_jobs=args.n_jobs, seed=args.seed,
                          comparison_versions=[STABLE_VERSION, args.candidate_version],
                          algorithms=args.algorithms, grace_seconds=args.grace_seconds,
                          evaluation_seconds=args.evaluation_seconds, memory_gib=args.memory_gib)
            freeze_run(run, config)
            version_root = run / args.version
            version_root.mkdir(exist_ok=True)
            env_spec = dict(device=device, affinity=affinity, python=str(python),
                            packages=read_json(preflight_output)["packages"],
                            freeze_sha256=sha256(environment_dir / "requirements-frozen.txt"))
            saved_env = version_root / "environment.json"
            if saved_env.exists() and read_json(saved_env) != env_spec:
                raise ValueError("Version environment/resources changed; use a new --run directory")
            write_json(saved_env, env_spec)
            shutil.copy2(preflight_output, version_root / "preflight.json")
            shutil.copy2(environment_dir / "requirements-frozen.txt", version_root / "requirements-frozen.txt")
            results = []
            for index, dataset in enumerate(datasets, 1):
                data = BASE / "comparison-data" / TABARENA_REVISION / dataset / "r0f0"
                print(f"\n{'=' * 78}\nDATASET {index}/{len(datasets)}: {dataset}\n{'=' * 78}", flush=True)
                spec = dict(config, version=args.version, dataset=dataset, data=str(data),
                            device=device, affinity=affinity, dataset_sha256=sha256(data / "dataset.json"))
                case = version_root / dataset / "r0f0"
                try:
                    result = execute(python, run / "code/worker.py", case, spec,
                                     grace=args.grace_seconds, evaluation_seconds=args.evaluation_seconds,
                                     memory_gib=args.memory_gib)
                    results.append(result)
                finally:
                    from report import build_report
                    build_report(run)
                write_json(version_root / "progress.json", dict(updated=now(), total=len(datasets),
                           completed=len(results), successful=sum(r["status"] == "success" for r in results)))
            print(f"Finished. Comparison: {run / 'comparison/report.md'}", flush=True)
            if any(result["status"] != "success" for result in results):
                raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. All outputs retained; rerun the same command to resume.", flush=True)
        raise SystemExit(130)
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
