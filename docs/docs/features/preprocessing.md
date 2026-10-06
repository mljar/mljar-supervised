---
description: How preprocessing works in MLJAR AutoML, including missing values, categorical features, text, datetime columns, and prediction-time reuse of learned transforms.
social:
  cards_layout: default/variant
---

# Preprocessing

`mljar-supervised` applies preprocessing automatically as part of training the model pipeline.

This is the default user experience:

- pass raw tabular data to `AutoML.fit(X, y)`
- let `AutoML` detect what needs to be transformed
- use the trained model later on raw rows again

In most cases, you do **not** need to manually encode categoricals, scale numeric columns, or convert text before training.

## What is handled automatically

Depending on the data and the selected algorithms, `mljar-supervised` can automatically handle:

- missing values
- categorical features
- text features
- datetime features
- numeric scaling
- target preprocessing
- removal of empty or constant columns

The learned preprocessing is stored with the trained model and reused at prediction time.

## Missing values

Missing values are handled automatically.

For input features:

- numeric columns are filled with the median value
- categorical columns are filled with the most frequent value
- datetime columns are filled with the most frequent value
- text columns are filled with a placeholder value

For the target:

- rows with missing target values are removed during training

This means you usually do not need to fill `NaN` values yourself before calling `fit()`.

## Categorical features

Categorical columns are detected automatically.

`mljar-supervised` chooses an encoding strategy based on the data and the model requirements. Depending on the situation, it can use:

- integer encoding
- one-hot encoding
- mixed encoding for some categorical setups

You do not need to run `pd.get_dummies()` before training.

## Text features

Text columns are detected automatically and transformed with TF-IDF.

This is useful when your data contains free-text fields such as:

- names
- descriptions
- comments
- titles

You can see this behavior in the Titanic tutorial, where text from the `Name` column becomes model features automatically.

## Datetime features

Datetime columns are detected automatically and transformed into model-ready features.

You do not need to manually split them into year, month, day, and similar components before training.

## Numeric scaling

Some algorithms need scaled numeric inputs, and some do not.

`mljar-supervised` applies scaling when it is needed by the training pipeline. This is handled automatically, so you do not need to scale numeric columns yourself first.

## Target preprocessing

Target values can also be preprocessed automatically.

Examples:

- classification targets can be converted to numeric labels
- multiclass targets can be encoded automatically
- regression targets can be scaled when needed

This is especially useful when the target is not already in the exact numerical form expected by a specific learner.

## Empty and constant columns

Columns that contain only missing values or only one unique non-missing value are removed automatically.

This helps avoid training on features that carry no useful information.

## What happens at prediction time

The same preprocessing learned during training is applied automatically when you call:

- `predict()`
- `predict_proba()`
- `score()`
- app generation and app execution flows

You should pass data with the same feature columns, but you do not need to manually repeat the fitted preprocessing steps yourself.

This is an important part of the contract:

- train on raw tabular data
- predict on raw tabular data
- let the stored pipeline handle consistency

## What you should still do yourself

Automatic preprocessing does not replace data understanding.

You should still:

- choose the correct target column
- remove obvious leakage columns
- remove identifiers if they should not be used as predictors
- make sure training data reflects the real prediction scenario
- keep training and prediction columns aligned

For example, a customer ID or transaction ID might technically be usable by the model, but it is usually not a meaningful predictive feature.

## Common questions

### Do I need one-hot encoding before `fit()`?

No. Categorical encoding is handled automatically.

### Do I need to fill missing values myself?

Usually no. Missing values are handled automatically for both training and prediction inputs.

### Can I pass strings and categoricals directly?

Yes. Raw categorical and text columns are supported.

### Can I use the model later on raw rows?

Yes. The fitted preprocessing is stored with the model and reused automatically.

### What if new prediction data contains missing values?

`mljar-supervised` applies the learned preprocessing to new data as well. If new missing values appear, they are handled in the preprocessing pipeline before prediction.

## Example

```python
from supervised import AutoML

automl = AutoML(results_path="AutoML")
automl.fit(X, y)

predictions = automl.predict(new_data)
```

In this example:

- `X` can contain missing values
- `X` can contain categorical columns
- `X` can contain text columns
- `new_data` should have matching feature columns
- preprocessing is applied automatically in both training and prediction

## Related pages

- [Apps](apps.md)
- [AutoML modes](modes.md)
- [Explainability](explain.md)
