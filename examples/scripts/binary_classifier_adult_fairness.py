
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from supervised.automl import AutoML

data = fetch_openml(data_id=1590, as_frame=True)
X = data.data
# data.target #
y = data.target # (data.target == ">50K") * 1
sensitive_features = X[["sex"]]

X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
    X, y, sensitive_features, stratify=y, test_size=0.75, random_state=42
)

automl = AutoML(
    algorithms=[
        "Xgboost"
    ],
    train_ensemble=False,
    fairness_metric="demographic_parity_ratio",  
    fairness_threshold=0.8,
    privileged_groups = [{"sex": "Male"}],
    underprivileged_groups = [{"sex": "Female"}],
)

automl.fit(X_train, y_train, sensitive_features=S_train)
