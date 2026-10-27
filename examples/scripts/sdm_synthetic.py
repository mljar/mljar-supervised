"""Synthetic Kumo/TabICLv2 comparison; downloads pinned HF weights on first use.

python examples/scripts/sdm_synthetic.py --device cuda --rows 1000
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_friedman1
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from supervised import AutoML


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--output", default="AutoML_SDM_synthetic")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    results = []
    for task, classes in [("binary", 2), ("multiclass", 3), ("regression", 0)]:
        X, y = (
            make_classification(
                n_samples=args.rows,
                n_features=20,
                n_informative=12,
                n_redundant=4,
                n_classes=classes,
                random_state=42,
            )
            if classes
            else make_friedman1(
                n_samples=args.rows, n_features=20, noise=1, random_state=42
            )
        )
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(20)])
        X["category"] = np.where(X.x0 > X.x0.median(), "high", "low")
        X.loc[::17, "x1"] = np.nan
        train, test, yt, ytest = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if classes else None
        )
        for algorithm in ["Kumo", "TabICLv2"]:
            started = time.monotonic()
            path = str(Path(args.output) / f"{task}_{algorithm}")
            model = AutoML(
                results_path=path,
                mode="Compete",
                algorithms=[algorithm],
                total_time_limit=300,
                train_ensemble=False,
                stack_models=False,
                golden_features=False,
                features_selection=False,
                boost_on_errors=False,
                kmeans_features=False,
                mix_encoding=False,
                explain_level=0,
                start_random_models=1,
                hill_climbing_steps=0,
                n_jobs=4,
                validation_strategy={
                    "validation_type": "split",
                    "train_ratio": 0.8,
                    "shuffle": True,
                    "stratify": bool(classes),
                },
                custom_params={
                    algorithm.lower(): {
                        "device": args.device,
                        "num_estimators": 4,
                        "local_files_only": args.local_files_only,
                    }
                },
            )
            model.fit(train, yt)
            pred = model.predict_proba(test) if classes else model.predict(test)
            score = (
                roc_auc_score(ytest, pred[:, 1])
                if classes == 2
                else (
                    accuracy_score(ytest, pred.argmax(1))
                    if classes
                    else r2_score(ytest, pred)
                )
            )
            restored = AutoML(results_path=path)
            again = restored.predict_proba(test) if classes else restored.predict(test)
            np.testing.assert_allclose(pred, again, atol=1e-6)
            results.append(
                dict(
                    algorithm=algorithm,
                    task=task,
                    rows=args.rows,
                    metric="AUC" if classes == 2 else "accuracy" if classes else "R2",
                    score=float(score),
                    seconds=time.monotonic() - started,
                    offline_reload=True,
                )
            )
            print(results[-1], flush=True)
    Path(args.output, "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
