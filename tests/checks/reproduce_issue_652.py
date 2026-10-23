"""Reproduce https://github.com/mljar/mljar-supervised/issues/652.

Run from the repository root:
    python -m tests.checks.reproduce_issue_652

Training reports and the original LightGBM traceback are retained in a temporary
directory. This script does not change library code or training defaults.
"""

import json
from pathlib import Path
import tempfile

import lightgbm
import numpy as np
import pandas as pd

from supervised import AutoML
from supervised.exceptions import AutoMLException


def main():
    root = Path(tempfile.mkdtemp(prefix="mljar-issue652-reproduction-"))
    print(f"pandas={pd.__version__}, LightGBM={lightgbm.__version__}")
    print(f"Training reports: {root}")

    rng = np.random.default_rng(652)
    X = pd.DataFrame(
        {"enum": np.tile([10, 20, 30], 40), "number": rng.normal(size=120)}
    ).astype({"enum": "category"})
    y = pd.Series(np.tile([0, 1], 60), name="target")
    cases = {
        "integer_target_01": y,
        "categorical_target_01": y.astype("category"),
        "categorical_target_12": (y + 1).astype("category"),
    }
    summary = []
    for name, target in cases.items():
        results_path = root / name
        automl = AutoML(
            results_path=str(results_path),
            mode="Explain",
            algorithms=["LightGBM"],
            total_time_limit=30,
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
            hill_climbing_steps=0,
            top_models_to_improve=0,
            n_jobs=1,
            verbose=0,
            validation_strategy={
                "validation_type": "split",
                "train_ratio": 0.8,
                "shuffle": True,
                "stratify": True,
            },
            random_state=652,
        )
        entry = {"case": name, "target_dtype": str(target.dtype)}
        try:
            automl.fit(X.copy(), target.copy())
            entry.update(status="PASS", predictions=len(automl.predict(X)))
        except AutoMLException as exc:
            error_report = results_path / "errors.md"
            entry.update(status="FAIL", error=str(exc))
            if error_report.exists():
                entry["error_report"] = str(error_report)
                entry["issue_652_reproduced"] = (
                    "Fields with bad pandas dtypes: target: category"
                    in error_report.read_text()
                )
        summary.append(entry)
        print(json.dumps(entry))

    (root / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
