"""Profile saved models without training or changing their artifacts."""

from __future__ import annotations

import argparse
import os
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
import sys
import time
import resource
from unittest.mock import patch

from common import BASE, lock, read_json, sha256, write_json


def validate_reference(manifest, reference):
    if reference.get("status") != "success":
        raise ValueError("Reference profile did not finish successfully")
    for key in (
        "artifact_sha256",
        "dataset_sha256",
        "device",
        "n_jobs",
        "cpu_affinity",
        "packages",
        "rows",
        "repeats",
    ):
        if manifest[key] != reference[key]:
            raise ValueError(f"Reference profile differs: {key}")


class PredictionProfiler:
    """Synchronized, inclusive phase timings; nested times must not be summed."""

    def __init__(self, synchronize=lambda: None, memory_snapshot=lambda: None):
        self.synchronize = synchronize
        self.memory_snapshot = memory_snapshot
        self.events = []
        self.model = None
        self.stack = ExitStack()

    def instrument(self, owner, name, phase):
        original = getattr(owner, name)

        @wraps(original)
        def measured(*args, **kwargs):
            previous = self.model
            if phase == "model":
                self.model = args[0].get_name()
                print(f"Predict {self.model}", flush=True)
            self.synchronize()
            memory_before = self.memory_snapshot()
            started = time.perf_counter()
            success = False
            hit = None
            try:
                result = original(*args, **kwargs)
                success = True
                if phase in ("prediction_lookup", "network_lookup"):
                    hit = result is not None
                return result
            finally:
                self.synchronize()
                self.events.append(
                    dict(
                        phase=phase,
                        model=self.model,
                        seconds=time.perf_counter() - started,
                        success=success,
                        hit=hit,
                        memory_before=memory_before,
                        memory_after=self.memory_snapshot(),
                        cleanup=(
                            kwargs.get("cleanup", True)
                            if phase == "gpu_admission_cleanup"
                            else None
                        ),
                    )
                )
                self.model = previous

        self.stack.enter_context(patch.object(owner, name, measured))

    def __enter__(self):
        from supervised.algorithms.sdm import SDMAlgorithm
        from supervised.algorithms import realmlp_resources
        from supervised.model_framework import ModelFramework
        from supervised.preprocessing.preprocessing import Preprocessing
        from supervised.prediction import PredictionContext
        import sdm

        self.instrument(ModelFramework, "predict", "model")
        self.instrument(ModelFramework, "release_learners", "release_learners")
        self.instrument(Preprocessing, "transform", "preprocessing")
        for method, phase in [
            ("_load_network", "checkpoint_load"),
            ("_preprocess", "sdm_preprocessing"),
            ("_check_resources", "admission"),
            ("_load_state", "fold_metadata"),
            ("_prediction_key", "equivalence_fingerprint"),
        ]:
            if hasattr(SDMAlgorithm, method):  # Also supports published RC4.
                self.instrument(SDMAlgorithm, method, phase)
        self.instrument(
            realmlp_resources, "available_gpu_memory", "gpu_admission_cleanup"
        )
        for method, phase in [
            ("sdm_prediction", "prediction_lookup"),
            ("network_for", "network_lookup"),
        ]:
            if hasattr(PredictionContext, method):
                self.instrument(PredictionContext, method, phase)
        for cls in (sdm.models.KumoTabular, sdm.models.TabICLv2):
            for method, phase in [
                ("fit", "context_encode"),
                ("predict", "query"),
                ("clear", "context_clear"),
            ]:
                self.instrument(cls, method, phase)
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


def main(args):
    if args.source_root:
        if not (args.source_root / "supervised/__init__.py").is_file():
            raise ValueError("source-root must contain supervised/__init__.py")
        sys.path.insert(0, str(args.source_root.resolve()))
    import importlib.metadata
    import pandas as pd
    import psutil
    import torch
    import supervised
    from supervised import AutoML
    from reevaluate import tree_hashes, verify_hashes
    from reevaluate_worker import assert_unloaded, gpu_stats

    attempt = args.attempt.resolve()
    spec = read_json(attempt / "case.json")
    if read_json(attempt / "automl/params.json").get("fit_level") != "finished":
        raise ValueError("Profile a finished fit only")
    data = Path(spec["data"])
    if sha256(data / "dataset.json") != spec["dataset_sha256"]:
        raise ValueError("Dataset manifest changed")
    info = read_json(data / "dataset.json")
    if sha256(data / "X_test.parquet") != info["files_sha256"]["X_test.parquet"]:
        raise ValueError("Test features changed")
    source = Path(supervised.__file__).resolve().parent
    if args.source_root and source != args.source_root.resolve() / "supervised":
        raise ValueError(f"Wrong inference source: {source}")
    if any(args.output.resolve().is_relative_to(p) for p in (attempt, source)):
        raise ValueError("Output must be separate from model artifacts and source")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if args.device == "cpu" and torch.cuda.is_available():
        raise ValueError("Set CUDA_VISIBLE_DEVICES='' for a CPU diagnostic")
    torch.set_num_threads(args.n_jobs)
    manifest = dict(
        source=str(source),
        source_sha256=tree_hashes(source),
        artifact_sha256=tree_hashes(attempt / "automl"),
        dataset_sha256=spec["dataset_sha256"],
        device=args.device,
        n_jobs=args.n_jobs,
        training_version=spec["version"],
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        rows=args.rows,
        repeats=args.repeats,
        packages={
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        timing="Synchronized inclusive phase timings; instrumentation adds overhead. Do not sum nested phases.",
        status="running",
    )
    if args.reference:
        validate_reference(manifest, read_json(args.reference / "manifest.json"))
    write_json(args.output / "manifest.json", manifest)
    X = pd.read_parquet(data / "X_test.parquet")
    automl = AutoML(results_path=str(attempt / "automl"))
    automl.load(str(attempt / "automl"))
    synchronize = torch.cuda.synchronize if args.device == "cuda" else lambda: None
    summaries = []
    print(f"Source: {source}\nSaved models: {attempt / 'automl'}", flush=True)

    def run(frame, name, reference=None):
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        synchronize()
        started = time.perf_counter()
        def memory_snapshot():
            if args.device != "cuda":
                return None
            free, total = torch.cuda.mem_get_info()
            return dict(
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                free_bytes=free, total_bytes=total,
            )

        profiler = PredictionProfiler(synchronize, memory_snapshot)
        try:
            with profiler:
                prediction = automl.predict_all(frame)
        finally:
            # A failed admission is precisely when the memory trace is needed.
            write_json(args.output / f"{name}.trace.json", profiler.events)
        synchronize()
        elapsed = time.perf_counter() - started
        assert_unloaded(automl)
        prediction.to_parquet(args.output / f"{name}.parquet", index=False)
        if reference is not None:
            pd.testing.assert_frame_equal(
                prediction.reset_index(drop=True),
                reference.reset_index(drop=True),
                check_exact=False,
                rtol=1e-5,
                atol=1e-7,
            )
        if args.reference:
            pd.testing.assert_frame_equal(
                prediction.reset_index(drop=True),
                pd.read_parquet(args.reference / f"{name}.parquet"),
                check_exact=False,
                rtol=1e-5,
                atol=1e-7,
            )
        summary = dict(
            name=name,
            rows=len(frame),
            seconds=elapsed,
            rss_gib=psutil.Process().memory_info().rss / 1024**3,
            process_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024**2,
            gpu=gpu_stats(torch, args.device),
        )
        summaries.append(summary)
        write_json(args.output / "timings.json", summaries)
        print(f"{name}: {elapsed:.2f}s; GPU {summary['gpu']}", flush=True)
        return prediction

    try:
        for rows in dict.fromkeys(min(n, len(X)) for n in args.rows):
            reference = None
            for repeat in range(args.repeats):
                reference = run(
                    X.iloc[:rows].copy(), f"rows-{rows}-call-{repeat + 1}", reference
                )
        check_X = X.iloc[: min(20, len(X))].copy()
        reference = run(check_X, "before-reload")
        del automl
        automl = AutoML(results_path=str(attempt / "automl"))
        automl.load(str(attempt / "automl"))
        run(check_X, "after-reload", reference)
        verify_hashes(attempt / "automl", manifest["artifact_sha256"])
        verify_hashes(source, manifest["source_sha256"])
        manifest.update(
            status="success", reload_verified=True, artifacts_unchanged=True
        )
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.output / "manifest.json", manifest)


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Source checkout/snapshot; omit for the installed release",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Earlier profile output for numerical parity checks",
    )
    parser.add_argument("--rows", nargs="+", type=positive, default=[1, 20, 128, 334])
    parser.add_argument("--repeats", type=positive, default=2)
    parser.add_argument("--n-jobs", type=positive, default=15)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--gpu-lock", type=Path, default=BASE / "comparison-environments/gpu.lock"
    )
    args = parser.parse_args()
    with ExitStack() as stack:
        if args.device == "cuda":
            print(f"Waiting for GPU lock: {args.gpu_lock}", flush=True)
            stack.enter_context(lock(args.gpu_lock))
        main(args)
