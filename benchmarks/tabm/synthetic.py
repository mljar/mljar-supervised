"""Reproducible, opt-in TabM sizing benchmark; each case runs in isolation.

Example: python benchmarks/tabm/synthetic.py --profile smoke --device cpu
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def worker(args):
    import psutil
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.datasets import make_classification, make_regression, make_friedman1
    from sklearn.metrics import log_loss, mean_squared_error
    from sklearn.model_selection import train_test_split
    from supervised.algorithms.tabm import (
        TabMClassificationAlgorithm,
        TabMRegressionAlgorithm,
    )

    case = json.loads(args.case)
    result = dict(case, requested_device=args.device)
    done = threading.Event()
    process = psutil.Process()
    peak = [process.memory_info().rss]

    def monitor():
        while not done.wait(0.05):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    start = time.monotonic()
    try:
        n, d = case["rows"], case["features"]
        if case["task"] == "regression":
            X, y = make_regression(
                n_samples=n,
                n_features=d,
                n_informative=min(10, d),
                noise=2,
                random_state=42,
            )
        elif case["task"] == "friedman":
            X, y = make_friedman1(n_samples=n, n_features=d, noise=0.1, random_state=42)
        else:
            X, y = make_classification(
                n_samples=n,
                n_features=d,
                n_informative=min(10, d - 2),
                n_classes=3 if case["task"] == "multiclass" else 2,
                weights=[0.95, 0.05] if case["task"] == "imbalanced" else None,
                random_state=42,
            )
        X = pd.DataFrame(X.astype("float32"))
        if case["task"] == "categorical":
            rng = np.random.default_rng(42)
            for cardinality in [10, 100, 1000]:
                X[f"cat_{cardinality}"] = pd.Categorical(
                    rng.integers(0, cardinality, n)
                )
            X.iloc[::10, 0] = np.nan
        regression = case["task"] in ["regression", "friedman"]
        Xtr, Xtest, ytr, ytest = train_test_split(
            X, y, test_size=0.2, random_state=43, stratify=None if regression else y
        )
        Xtr, Xval, ytr, yval = train_test_split(
            Xtr,
            ytr,
            test_size=0.25,
            random_state=44,
            stratify=None if regression else ytr,
        )
        cls = (
            TabMRegressionAlgorithm if regression else TabMClassificationAlgorithm
        )
        task = (
            "regression"
            if regression
            else (
                "multiclass_classification"
                if case["task"] == "multiclass"
                else "binary_classification"
            )
        )
        model = cls(
            dict(
                ml_task=task,
                device=args.device,
                n_jobs=args.n_jobs,
                n_epochs=args.epochs,
                d_block=case["width"],
                n_blocks=case["depth"],
                seed=42,
            )
        )
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        train_start = time.monotonic()
        model.fit(Xtr, ytr, X_validation=Xval, y_validation=yval, max_time=args.seconds)
        result["fit_seconds"] = time.monotonic() - train_start
        predict_start = time.monotonic()
        p = model.predict(Xtest)
        result["predict_seconds"] = time.monotonic() - predict_start
        result["prediction_rows_per_second"] = len(Xtest) / result["predict_seconds"]
        result["test_metric"] = (
            mean_squared_error(ytest, p) if regression else log_loss(ytest, p)
        )
        result["metric_name"] = "mse" if regression else "logloss"
        result["training"] = model.training_info
        result["peak_cuda_allocated_bytes"] = (
            torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
        )
        result["peak_cuda_reserved_bytes"] = (
            torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else 0
        )
        result["status"] = "ok"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        done.set()
        thread.join()
        result.update(peak_rss_bytes=peak[0], elapsed_seconds=time.monotonic() - start)
        Path(args.output).write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=["smoke", "scaling", "large"], default="smoke"
    )
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--output", default="performance-results/tabm")
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        worker(args)
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.profile == "smoke":
        cases = [
            dict(task=task, rows=1000, features=20, width=w, depth=d)
            for task in ["binary", "multiclass", "regression"]
            for w in [64, 128]
            for d in [2, 3]
        ]
        cases += [
            dict(task=t, rows=3000, features=20, width=64, depth=2)
            for t in ["friedman", "imbalanced", "categorical"]
        ]
    else:
        sizes = [(10000, 50), (100000, 100), (10000, 1000)]
        if args.profile == "large":
            sizes = [(1000000, 50)]
        cases = [
            dict(task="binary", rows=n, features=f, width=w, depth=d)
            for n, f in sizes
            for w, d in [(64, 2), (128, 2), (128, 3)]
        ]
        if args.profile == "scaling":
            cases.append(
                dict(task="categorical", rows=50000, features=20, width=64, depth=2)
            )
    results = []
    for i, case in enumerate(cases):
        path = out / f"{i:03d}.json"
        command = [
            sys.executable,
            __file__,
            "--case",
            json.dumps(case),
            "--output",
            str(path),
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--seconds",
            str(args.seconds),
            "--n-jobs",
            str(args.n_jobs),
        ]
        with (out / f"{i:03d}.log").open("w") as log:
            try:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=args.seconds + 120,
                )
                result = json.loads(path.read_text())
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                result = dict(case, status="failed", error=str(exc))
                path.write_text(json.dumps(result, indent=2))
        results.append(result)
        (out / "summary.json").write_text(json.dumps(results, indent=2))
        print(f"{i + 1}/{len(cases)} {case}: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
