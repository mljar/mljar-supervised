---
description: How preprocessing works in MLJAR AutoML, including missing values, categorical features, text, datetime columns, IP addresses, and prediction-time reuse of learned transforms.
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
