import pandas as pd
import numpy as np
from supervised.automl import AutoML
from sklearn.datasets import load_digits


digits = load_digits()
X = pd.DataFrame(digits.data)
y = digits.target

print(X)

automl = AutoML(
#        results_path="AutoML_1",
        total_time_limit=10,
        start_random_models=1,
        hill_climbing_steps=0,
        top_models_to_improve=0,
        train_ensemble=True)
automl.fit(X, y)

predictions = automl.predict(X)

print(predictions.head())
print(predictions.tail())
