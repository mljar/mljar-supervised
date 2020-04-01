import pandas as pd
import numpy as np
from supervised.automl import AutoML
from sklearn.datasets import load_digits


digits = load_digits()
X = pd.DataFrame(digits.data)
y = digits.target

automl = AutoML(
#        results_path="AutoML_1",
        total_time_limit=30,
        start_random_models=5,
        hill_climbing_steps=0,
        top_models_to_improve=0,
        train_ensemble=True)
automl.fit(X, y)

predictions = automl.predict(X)

print(predictions.head())
print(predictions.tail())
