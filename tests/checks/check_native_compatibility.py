"""Fresh-interpreter smoke checks; run against the installed wheel outside the repo.

Each invocation owns one case. No model downloads or optional-model training.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["base", "auto", "explicit", "torch-first"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.case == "torch-first":
        import torch

    from supervised import AutoML
    from supervised.exceptions import NativeCompatibilityError
    from supervised.utils import native_compatibility as native
    from sklearn.datasets import make_classification

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    if args.case in ("explicit", "torch-first"):
        if sys.platform != "darwin":
            assert native.check_native_compatibility(True)["reason"] == "platform"
            print(json.dumps(dict(case=args.case, status="not_needed")))
            return
        algorithm = "RealMLP" if args.case == "explicit" else "LightGBM"
        automl = AutoML(mode="Compete", algorithms=[algorithm], results_path=str(root / "run"))
        started = time.monotonic()
        try:
            # Decide without training an optional model on a compatible Mac.
            evidence = native.require_compatible(True, algorithm)
        except NativeCompatibilityError as error:
            evidence = error.result
            # Native crashes are cached; a subsequent API entry must also refuse.
            if evidence["status"] == "blocked":
                try:
                    automl.fit(X, y)
                except NativeCompatibilityError:
                    pass
                else:
                    raise AssertionError("fit bypassed the blocked decision")
        assert time.monotonic() - started < 3.5
        if args.case == "explicit":
            assert "torch" not in sys.modules
        print(json.dumps(dict(case=args.case, evidence=evidence)))
        return

    if args.case == "base":
        assert importlib.util.find_spec("torch") is None
        # Catch an unintended probe while allowing ordinary backend subprocesses.
        native._run_child = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("base install probed"))
        algorithms = ["LightGBM", "Decision Tree"]
    else:
        assert importlib.util.find_spec("pytabkit") is not None
        algorithms = "auto"

    automl = AutoML(
        results_path=str(root / "run"), mode="Compete", algorithms=algorithms,
        total_time_limit=20, model_time_limit=2, train_ensemble=True,
        stack_models=False, explain_level=0, golden_features=False,
        feature_discovery=False, features_selection=False, kmeans_features=False,
        boost_on_errors=False, start_random_models=1, hill_climbing_steps=0,
        validation_strategy={"validation_type": "split", "train_ratio": 0.8},
        n_jobs=2, verbose=0,
    )
    # On platforms without this gate, use the base case for actual training.
    # On compatible Macs selection is tested without training optional networks.
    if args.case == "auto":
        automl._ml_task = "binary_classification"
        effective = automl._get_algorithms()
        evidence = automl._native_compatibility
        if not evidence or not evidence.get("skipped_algorithms"):
            print(json.dumps(dict(case=args.case, effective=effective, evidence=evidence)))
            return
    automl.fit(X, y)
    predicted = automl.predict(X[:5])
    reloaded = AutoML(results_path=str(root / "run"), verbose=0)
    assert (predicted == reloaded.predict(X[:5])).all()
    assert "torch" not in sys.modules
    print(json.dumps(dict(case=args.case, effective=automl._algorithms,
                         evidence=automl._native_compatibility,
                         models=[model.get_type() for model in automl._models],
                         torch_in_parent="torch" in sys.modules)))


if __name__ == "__main__":
    main()
