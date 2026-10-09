"""Train on raw IP addresses and reuse the saved pipeline on new addresses."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from supervised import AutoML


def main(results_path):
    # Synthetic labels depend on address components, purely for demonstration.
    rows = 128
    X = pd.DataFrame(
        {
            "client_ip": [
                f"192.0.2.{i}" if i % 2 == 0 else f"2001:db8::{i:x}"
                for i in range(1, rows + 1)
            ],
            "request_count": [(i * 7) % 50 + 1 for i in range(1, rows + 1)],
        }
    )
    y = pd.Series([int(i > rows // 2) for i in range(1, rows + 1)])
    X.loc[[9, 89], "client_ip"] = None

    print("Training input (first 10 rows, with target):")
    print(X.assign(target=y).head(10).fillna("<missing>").to_string(index=False))

    automl = AutoML(
        results_path=str(results_path),
        algorithms=["Decision Tree"],
        total_time_limit=30,
        train_ensemble=False,
        stack_models=False,
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        start_random_models=1,
        hill_climbing_steps=0,
        explain_level=0,
        n_jobs=1,
        verbose=0,
    )
    automl.fit(X, y)

    new_data = pd.DataFrame(
        {
            "client_ip": ["192.0.2.200", "2001:db8::c8", None],
            "request_count": [12, 7, 3],
        }
    )
    print("Prediction input:")
    print(new_data.fillna("<missing>").to_string(index=False))
    predictions = automl.predict(new_data.copy())
    restored = AutoML(results_path=str(results_path))
    np.testing.assert_array_equal(predictions, restored.predict(new_data.copy()))
    print(new_data.assign(prediction=predictions).to_string(index=False))
    print("Predictions match after reloading the saved model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, default=Path("AutoML_IP"))
    main(parser.parse_args().results_path)
