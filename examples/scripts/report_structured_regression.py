"""
Example: structured report for regression.

What this script does:
- Trains AutoML on synthetic regression data.
- Prints compact structured report output.
- Selects one model from the leaderboard and prints on-demand details for that model.

Why this helps:
- Shows the default short report flow.
- Demonstrates how to inspect one model deeply without printing all model details.
"""

import os

from sklearn.datasets import make_regression

from supervised import AutoML


def main():
    X, y = make_regression(
        n_samples=300,
        n_features=12,
        n_informative=8,
        noise=2.0,
        random_state=123,
    )

    results_path = "AutoML_report_structured_regression"
    automl = AutoML(
        mode="Explain",
        total_time_limit=300,
        results_path=results_path,
        random_state=123,
        verbose=0,
    )
    automl.fit(X, y)

    print("\n=== report_structured() ===\n")
    print(automl.report_structured())

    payload = automl.report_structured(format="dict")
    print("\nTop-level keys:", sorted(payload.keys()))
    selected_model_name = None
    leaderboard = payload.get("leaderboard", [])
    if len(leaderboard) > 1:
        selected_model_name = leaderboard[1].get("name")
    elif leaderboard:
        selected_model_name = leaderboard[0].get("name")
    print("Selected model:", selected_model_name)

    if selected_model_name:
        print(f"\n=== report_structured(model_name='{selected_model_name}') ===\n")
        print(automl.report_structured(model_name=selected_model_name))

    report_path = os.path.join(results_path, "report_structured.json")
    print("Structured report JSON:", report_path)
    print("Exists:", os.path.exists(report_path))


if __name__ == "__main__":
    main()
