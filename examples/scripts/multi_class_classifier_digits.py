import pandas as pd
import numpy as np
from supervised.automl import AutoML
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

digits = load_digits()
X = pd.DataFrame(digits.data)
y = digits.target



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)


automl = AutoML(
#        results_path="AutoML_1",
        total_time_limit=10,
        start_random_models=1,
        hill_climbing_steps=0,
        top_models_to_improve=0,
        train_ensemble=True)

automl.fit(X_train, y_train)
predictions = automl.predict(X_test)

print(predictions.head())
print("Test accuracy:", accuracy_score(y_test, predictions["label"].astype(int)))


