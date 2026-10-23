"""Small RealMLP examples. Install with: pip install '.[realmlp]'."""

import argparse
from pathlib import Path

from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from supervised import AutoML


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="AutoML_realmlp_examples")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    for task in ["binary", "multiclass", "regression"]:
        if task == "regression":
            X, y = make_regression(
                n_samples=1000, n_features=20, noise=10, random_state=42
            )
        else:
            X, y = make_classification(
                n_samples=1000, n_features=20, n_informative=10,
                n_classes=2 if task == "binary" else 3, random_state=42,
            )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=None if task == "regression" else y,
        )
        automl = AutoML(
            results_path=str(Path(args.output) / task),
            mode="Compete",
            algorithms=["RealMLP"],
            total_time_limit=120,
            n_jobs=2,
            train_ensemble=False,
            stack_models=False,
            golden_features=False,
            features_selection=False,
            boost_on_errors=False,
            kmeans_features=False,
            mix_encoding=False,
            start_random_models=1,
            hill_climbing_steps=0,
            explain_level=1,
            validation_strategy={
                "validation_type": "split", "train_ratio": 0.8,
                "shuffle": True, "stratify": task != "regression",
            },
            custom_params={"realmlp": {"device": "cpu", "n_epochs": args.epochs}},
        )
        automl.fit(X_train, y_train)
        score = r2_score if task == "regression" else accuracy_score
        print(f"{task}: {score.__name__}={score(y_test, automl.predict(X_test)):.4f}")


if __name__ == "__main__":
    main()
