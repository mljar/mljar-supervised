"""Train, evaluate, and reload the README's Iris classifier.

Run in a new working directory with mljar-supervised==1.4.0b1 installed.
The Iris dataset ships with scikit-learn; no data download is needed.
"""

# README_QUICKSTART_START
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from supervised import AutoML

# Predict the flower species from four measurements.
X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

automl = AutoML(
    results_path="AutoML_Iris",
    mode="Explain",
    algorithms=["Baseline", "Decision Tree", "Linear", "LightGBM"],
    total_time_limit=120,
    n_jobs=2,
)
automl.fit(X_train, y_train)

predictions = automl.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, predictions):.1%}")
automl.report()  # Creates AutoML_Iris/README.html.
# Save example inputs so the reload example also works in a new Python session.
X_test.head(3).to_csv("iris_new_rows.csv", index=False)
# README_QUICKSTART_END

# README_RELOAD_START
import pandas as pd
from supervised import AutoML

saved_model = AutoML(results_path="AutoML_Iris")
new_rows = pd.read_csv("iris_new_rows.csv")
print(saved_model.predict(new_rows))
# README_RELOAD_END

import numpy as np

np.testing.assert_array_equal(predictions[:3], saved_model.predict(new_rows))
