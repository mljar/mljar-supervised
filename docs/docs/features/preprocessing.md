---
title: "Automatic Data Preprocessing for Machine Learning"
description: "Learn how MLJAR AutoML handles missing values, categorical data, text, dates, and IP addresses, then reuses learned transformations for predictions."
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
- duration features (`timedelta` columns converted to total seconds)
- IPv4 and IPv6 address features
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
- IP columns use zero-filled address components and a missing-value indicator

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

One-hot columns are constructed together to avoid pandas frame fragmentation on
wide categorical datasets. Their values, names and ordering are preserved when
loading existing models.

## KMeans features with missing values

New KMeans feature transformers learn numeric medians from their training fold
before scaling and clustering. Missing values in an entirely empty input column
use zero. The medians, scaler and cluster centers are saved together and reused
for validation and prediction, including missing values first seen at prediction
time. KMeans features do not depend on the other rows in a prediction batch.
The transformer fills its own numeric input copy; it does not replace the
original feature values. Existing artifacts without saved imputation statistics
retain their previous behavior. New fits use a fixed clustering seed.

## Frozen categorical mappings

Integer feature encoders keep the mappings learned during fitting. Every unseen
category receives the same reserved code, equal to the number of fitted
categories. The code does not depend on other rows in the prediction batch or
on earlier prediction requests. Known-category codes remain unchanged after
save/reload. This policy also applies to loaded older models, so predictions
containing unseen categories can differ from earlier releases that assigned
new codes separately in each batch. Target-label encoding is unchanged.

With `mix_encoding=True`, the tuner can also try mixed categorical encoding for
`Neural Network`: low-cardinality columns use one-hot encoding, while columns
with many categories keep integer encoding. When switching strategies, AutoML
rebuilds the categorical preprocessing and its dependent scaling together.
Integer codes are scaled when the algorithm requires scaling; one-hot columns
use their binary values by default. Scaling refers to the fitted encoder's
actual output columns, so it never tries to scale a source column removed by
one-hot encoding. The fitted encoders and scalers are saved and reused during
prediction and model reload.

The categorical encoding selected from the input data is honored in every
training fold. A fold with a higher proportion of unique values does not switch
the column to text or skip its encoding. This keeps raw strings out of numeric
model inputs. One-hot encoding still falls back to integer encoding when the
fitted column has more than 200 categories.

Boolean columns can use NumPy `bool`, object-backed booleans, pandas nullable
`boolean`, or boolean categorical values. Integer encoding keeps their label
codes consistent during training, prediction, and reload, including when a
training fold contains only one boolean value. Encoded values replace the input
column with an integer column, so nullable boolean dtypes do not reject them.

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

Convert date strings to a pandas datetime dtype with `pd.to_datetime()` before training.

You do not need to manually split them into year, month, day, and similar components before training.

## Duration features

Pandas `timedelta` columns are automatically converted to numeric **total seconds**
before missing-value handling, scaling, and feature engineering. Each duration
column keeps its name and produces one numeric column. Negative and fractional
durations are supported; `NaT` becomes `NaN` and follows the algorithm's normal
missing-value handling. Empty and constant duration columns follow the usual
column-removal rules.

```python
X = pd.DataFrame({
    "elapsed": pd.to_timedelta(["1 day", "500ms", "-2s", None]),
    "temperature": [20.0, 21.0, 19.0, 22.0],
})
print(X)
```

Before any subsequent imputation or scaling, the features are:

| Input row | `elapsed` (seconds) | `temperature` |
| --- | ---: | ---: |
| `1 day` | 86400.0 | 20.0 |
| `500ms` | 0.5 | 21.0 |
| `-2s` | -2.0 | 19.0 |
| `NaT` | NaN | 22.0 |

The fitted preprocessing saves the names of duration columns and applies the same
conversion during validation and prediction, including after loading a saved
model. Conversion uses the duration's actual units, rather than assuming the
underlying values are nanoseconds. String columns are not automatically parsed
as durations; use `pd.to_timedelta()` before training when needed.

## IP address features

String columns are detected as IP addresses when every non-missing value is a valid
plain IPv4 or IPv6 address. Detection runs before categorical/text classification,
and surrounding whitespace is ignored. A column can contain both address versions.

Columns explicitly assigned pandas `category` dtype keep categorical handling.
Numeric columns are not interpreted as IP addresses. Empty and constant columns
are still removed as usual. If any non-missing training value is not a supported
address, the column follows the existing categorical/text rules.

Each detected IP column has up to 14 candidate numeric features:

- version (`4` or `6`, or `0` for missing)
- four IPv4 octets (`0`–`255`), zero-filled for IPv6
- eight IPv6 groups (`0`–`65535`), zero-filled for IPv4
- a missing-value indicator (`1` for missing, otherwise `0`)

For example, `client_ip` generates `client_ip_IP_Version`,
`client_ip_IP_IPv4_1` through `client_ip_IP_IPv4_4`,
`client_ip_IP_IPv6_1` through `client_ip_IP_IPv6_8`, and `client_ip_IP_Missing`.
**Generated IP columns that are constant after transformation are dropped.**
Only features with at least two distinct values in the learner's training data
are retained. This applies to every generated IP column, including version,
individual address components, and the missing-value indicator. Existing columns
with retained feature names cause an error rather than being overwritten.

Missing addresses have zero-filled version and components. Equivalent IPv6
spellings produce the same features. The fitted schema accepts unseen valid
addresses of either version, and scaling is applied when required by the algorithm.

For example, training on `192.0.2.1`, `192.0.2.2`, and `192.0.2.3` retains only
the varying fourth octet, `client_ip_IP_IPv4_4`. The version is always `4`, the
first three octets are always `192`, `0`, and `2`, all IPv6 groups are always `0`,
and the missing indicator is always `0`, so those columns are omitted.
Selection happens after IP expansion and before scaling, using each learner's
training fold, never its validation data.

The saved model stores the retained columns and reuses them at prediction time.
A dropped feature stays dropped during validation, prediction, and reload, even
if it varies in new data. There is no new constant-column check on prediction
rows. Valid IPv6 or
missing values are still accepted by an IPv4-only model, but their distinguishing
features may have been dropped because they did not vary during training. Malformed
prediction values raise an error identifying the column and row. CIDR networks,
addresses with ports, and IPv6 zone identifiers are outside the supported format.
This transformation does not look up geographic locations or hostnames.

See the [IP address tutorial](../tutorials/ip-addresses.md) for a runnable example.

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
