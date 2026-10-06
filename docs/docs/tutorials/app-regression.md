---
description: Beginner-friendly tutorial for training MLJAR AutoML on house prices regression data and sharing the model as a Mercury app.
social:
  cards_layout: default/variant
---

# Share a regression model as an app

You trained `AutoML`. What next?

You can share the trained regression model as an app so other people can enter values in a browser or upload a CSV file for scoring.

In this tutorial, we will:

- train a regression model on house prices data,
- keep the app input form compact by using only `10` features,
- generate an app from the trained model,
- run it locally,
- and optionally publish it online.

## Load data

We will use the house prices dataset from `datasets-for-start`:

```python
import pandas as pd

data = pd.read_csv(
    "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/house_prices/data.csv"
)

print(data.head())
```

The target column is `SalePrice`.

## Select a compact feature set

The full dataset has many columns. For an app tutorial, it is better to keep the single-prediction form short and readable.

We will use these `10` features:

- `OverallQual`
- `GrLivArea`
- `GarageCars`
- `TotalBsmtSF`
- `1stFlrSF`
- `FullBath`
- `YearBuilt`
- `YearRemodAdd`
- `Neighborhood`
- `LotArea`

```python
features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "YearBuilt",
    "YearRemodAdd",
    "Neighborhood",
    "LotArea",
]

X = data[features]
y = data["SalePrice"]
```

This keeps the generated Mercury app beginner-friendly and pleasant to use.

## Train AutoML

```python
from supervised import AutoML

automl = AutoML(
    results_path="AutoML_House_Prices_App",
    mode="Explain",
    random_state=1,
)
automl.fit(X, y)
```

Now the regression model is trained and ready to use.

## Generate the app

```python
automl.app()
```

This creates the generated app workspace in:

```text
AutoML_House_Prices_App/app
```

Because we selected only `10` features, the app will include the full single-prediction form instead of switching to batch-only mode.

## Run the app locally

The quickest way to preview it is:

```python
automl.local_app()
```

This starts Mercury, opens the browser, and keeps the app running until you stop it.

!!! note
    Press `Ctrl+C` in the terminal to stop the local app.

## Run the app manually

You can also run the generated app manually:

```bash
cd AutoML_House_Prices_App/app
pip install -r requirements.txt
mercury --working-dir=.
```

## What the app looks like

The generated regression app can include:

- a single prediction form for one property,
- a predicted value summary,
- batch CSV scoring,
- a downloadable scored file,
- feature context and feature importance plots when available.

This makes it easy to share a trained pricing model with non-technical users.

## Publish the app online

If you want the fastest hosted flow, use:

```python
automl.publish_app()
```

This helper will:

- sign you in through `platform.mljar.com`,
- create the app URL on the first publish,
- reuse the last successfully published app URL on later publishes,
- upload the generated files,
- and print the final app address.

Using `platform.mljar.com` is simply the fastest path.

## Host it on your own server

If you want full control, you can host the generated Mercury app yourself:

1. generate the app with `automl.app()`,
2. install dependencies on your server,
3. run Mercury,
4. deploy with your own infrastructure.

## Summary

After training a regression model with `AutoML`, you can immediately turn it into an interactive app.

That gives you a practical path from:

- trained model,
- to browser-based demo,
- to a shareable tool for other people.
