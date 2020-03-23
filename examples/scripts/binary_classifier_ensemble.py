import pandas as pd
from supervised.automl import AutoML
from supervised.algorithms.ensemble import Ensemble
import os

df = pd.read_csv("https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv", skipinitialspace=True)

X = df[df.columns[:-1]]
y = df["income"]

results_path = "AutoML_1"
automl = AutoML(
        results_path=results_path,
        total_time_limit=400,
        start_random_models=10,
        hill_climbing_steps=0,
        top_models_to_improve=0,
        train_ensemble=False)


ensemble = Ensemble("logloss", "binary_classification")
ensemble.models = automl._models

oofs = {}
target = None
for i in range(1,10):
    oof = pd.read_csv(os.path.join(results_path, f"model_{i}", "predictions_out_of_folds.csv"))
    prediction_cols = [c for c in oof.columns if "prediction" in c]
    oofs[f"model_{i-1}"] = oof[prediction_cols]
    if target is None:
        target_columns = [c for c in oof.columns if "target" in c]
        target = oof[target_columns] 

ensemble.target = target
ensemble.target_columns = "target"
ensemble.fit(oofs, target)


ensemble.save(os.path.join(results_path, "ensemble"))
predictions = ensemble.predict(X)
print(predictions.head())
