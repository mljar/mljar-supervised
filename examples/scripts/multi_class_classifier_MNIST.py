import pandas as pd
import numpy as np
from supervised.automl import AutoML


from supervised.utils.config import mem


df = pd.read_csv("tests/data/MNIST/train.csv")

X = df[[f for f in df.columns if "pixel" in f]]
y = df["label"]

for _ in range(4):
    X = pd.concat([X, X], axis=0)
    y = pd.concat([y, y], axis=0)


mem()


automl = AutoML(
    # results_path="AutoML_12",
    total_time_limit=60 * 60,
    start_random_models=5,
    hill_climbing_steps=2,
    top_models_to_improve=3,
    train_ensemble=True,
)

mem()
print("Start fit")
automl.fit(X, y)

test = pd.read_csv("tests/data/MNIST/test.csv")
predictions = automl.predict(test)

print(predictions.head())
print(predictions.tail())

sub = pd.DataFrame({"ImageId": 0, "Label": predictions["label"]})
sub["ImageId"] = sub.index + 1
sub.to_csv("sub1.csv", index=False)
