import os

import pandas as pd
from sklearn.datasets import make_classification

from supervised import AutoML


def main():
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_redundant=1,
        random_state=123,
    )

    # Construct two categorical sensitive features.
    sensitive_features = pd.DataFrame(
        {
            "gender": ["female" if i % 2 == 0 else "male" for i in range(len(y))],
            "group": ["A" if i % 3 == 0 else "B" for i in range(len(y))],
        }
    )

    results_path = "AutoML_report_structured_fairness"
    automl = AutoML(
        mode="Explain",
        total_time_limit=300,
        fairness_metric="demographic_parity_ratio",
        fairness_threshold=0.8,
        privileged_groups=[{"gender": "male"}],
        underprivileged_groups=[{"gender": "female"}],
        results_path=results_path,
        random_state=123,
        verbose=0,
    )
    automl.fit(X, y, sensitive_features=sensitive_features)

    print("\n=== report_structured(model_details=False) ===\n")
    print(automl.report_structured(model_details=False))

    print("\n=== report_structured(model_details=True) ===\n")
    print(automl.report_structured(model_details=True))

    payload = automl.report_structured(format="dict", model_details=False)
    print("\nTop-level keys:", sorted(payload.keys()))
    print("Number of models in report:", len(payload.get("models", [])))
    print("Fairness summary available:", payload.get("fairness_summary") is not None)

    report_path = os.path.join(results_path, "report_structured.json")
    print("Structured report JSON:", report_path)
    print("Exists:", os.path.exists(report_path))


if __name__ == "__main__":
    main()
