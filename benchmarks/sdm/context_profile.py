"""Profile large sklearn input with bounded context (not full-data fitting).

python benchmarks/sdm/context_profile.py --rows 1000000 --context 4096 --device cuda
"""

import argparse
import gc
import json
from pathlib import Path
import resource
import time

import torch
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from supervised.algorithms.sdm import (
    KumoClassificationAlgorithm,
    TabICLv2ClassificationAlgorithm,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="performance-results/sdm/profile.json")
    args = parser.parse_args()
    X, y = make_classification(
        n_samples=args.rows + 1024,
        n_features=args.features,
        n_informative=min(12, args.features - 2),
        n_redundant=2,
        random_state=42,
    )
    results = []
    for cls in [KumoClassificationAlgorithm, TabICLv2ClassificationAlgorithm]:
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        model = cls(
            {
                "ml_task": "binary_classification",
                "max_context_size": args.context,
                "num_estimators": 4,
                "predict_batch_size": 128,
                "device": args.device,
                "n_jobs": 4,
                "local_files_only": True,
            }
        )
        started = time.monotonic()
        model.fit(
            X[: args.rows],
            y[: args.rows],
            X_validation=X[args.rows :],
            y_validation=y[args.rows :],
            max_time=300,
        )
        fit_seconds = time.monotonic() - started
        started = time.monotonic()
        prediction = model.predict(X[args.rows :])
        row = dict(
            algorithm=model.algorithm_short_name,
            source_rows=args.rows,
            context_rows=len(model.context),
            features=args.features,
            query_rows=1024,
            fit_and_validation_seconds=fit_seconds,
            predict_seconds=time.monotonic() - started,
            auc=float(roc_auc_score(y[args.rows :], prediction)),
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            peak_cuda_reserved_bytes=(
                torch.cuda.max_memory_reserved()
                if args.device.startswith("cuda")
                else 0
            ),
            resources=model.training_info["resources"],
        )
        results.append(row)
        print(json.dumps(row), flush=True)
        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
