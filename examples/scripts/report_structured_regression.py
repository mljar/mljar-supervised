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

    print("\n=== report_structured(model_details=False) ===\n")
    print(automl.report_structured(model_details=False))

    print("\n=== report_structured(model_details=True) ===\n")
    print(automl.report_structured(model_details=True))

    payload = automl.report_structured(format="dict", model_details=False)
    print("\nTop-level keys:", sorted(payload.keys()))
    print("Number of models in report:", len(payload.get("models", [])))

    report_path = os.path.join(results_path, "report_structured.json")
    print("Structured report JSON:", report_path)
    print("Exists:", os.path.exists(report_path))


if __name__ == "__main__":
    main()
