# Train with IP addresses

Pass raw IPv4/IPv6 strings directly to `AutoML.fit()`. IP detection and numeric
transformation are automatic; no additional configuration is needed.

This example uses synthetic data with labels based on address components. It
demonstrates the preprocessing workflow rather than a real-world prediction task.

```python
import numpy as np
import pandas as pd
from supervised import AutoML

X = pd.DataFrame({
    "client_ip": [
        f"192.0.2.{i}" if i % 2 == 0 else f"2001:db8::{i:x}"
        for i in range(1, 129)
    ],
    "request_count": [(i * 7) % 50 + 1 for i in range(1, 129)],
})
y = pd.Series([int(i > 64) for i in range(1, 129)])
X.loc[[9, 89], "client_ip"] = None

print("Training input (first 10 rows, with target):")
print(X.assign(target=y).head(10).fillna("<missing>").to_string(index=False))

automl = AutoML(
    results_path="AutoML_IP",
    algorithms=["Decision Tree"],
    total_time_limit=30,
    train_ensemble=False,
    stack_models=False,
    golden_features=False,
    features_selection=False,
    kmeans_features=False,
    start_random_models=1,
    hill_climbing_steps=0,
    explain_level=0,
    n_jobs=1,
    verbose=0,
)
automl.fit(X, y)

new_data = pd.DataFrame({
    "client_ip": ["192.0.2.200", "2001:db8::c8", None],
    "request_count": [12, 7, 3],
})
print("Prediction input:")
print(new_data.fillna("<missing>").to_string(index=False))
predictions = automl.predict(new_data.copy())

restored = AutoML(results_path="AutoML_IP")
np.testing.assert_array_equal(
    predictions, restored.predict(new_data.copy())
)
print(new_data.assign(prediction=predictions))
```

Both training and prediction use raw strings. The model stores the generated
feature schema and reuses it after reload, including for addresses absent during
training and missing values that first appear during prediction.

## Inspect the input

The training input printout begins with:

```text
Training input (first 10 rows, with target):
  client_ip  request_count  target
2001:db8::1              8       0
  192.0.2.2             15       0
2001:db8::3             22       0
  192.0.2.4             29       0
2001:db8::5             36       0
  192.0.2.6             43       0
2001:db8::7             50       0
  192.0.2.8              7       0
2001:db8::9             14       0
  <missing>             21       0
```

`target` is shown alongside the input for inspection; it is passed separately as
`y` and is not an input feature. `<missing>` is a display label only: the data
passed to AutoML still contains a missing value.

The raw prediction input is:

```text
Prediction input:
   client_ip  request_count
 192.0.2.200             12
2001:db8::c8              7
   <missing>              3
```

## What features are constructed?

The input has two columns: `client_ip` and `request_count`. AutoML replaces
`client_ip` with numeric features, keeping only those that vary in training.
There are **14 candidate IP columns**. For this training input, six are always
zero and are omitted, leaving **8 IP columns**. The nonconstant numeric
`request_count` column stays unchanged with the Decision Tree configuration used
here, giving **9 model input columns**. Components are ordered from left to right
in the address:

| Candidate columns | Maximum count | Meaning |
| --- | --- | --- |
| `client_ip_IP_Version` | 1 | `4` for IPv4, `6` for IPv6, `0` for missing |
| `client_ip_IP_IPv4_1` through `client_ip_IP_IPv4_4` | 4 | Decimal IPv4 octets, each between `0` and `255` |
| `client_ip_IP_IPv6_1` through `client_ip_IP_IPv6_8` | 8 | Expanded IPv6 groups converted from hexadecimal to decimal, each between `0` and `65535` |
| `client_ip_IP_Missing` | 1 | `1` for missing, otherwise `0` |

The table below shows the exact numeric values constructed for the three
prediction rows after omitting constant training features. It is transposed for
readability: **each row names a separate
model input column**, and every value is a scalar number.

| Source input column | Model input column | IPv4 row: `192.0.2.200` | IPv6 row: `2001:db8::c8` | Missing IP row |
| --- | --- | ---: | ---: | ---: |
| `client_ip` | `client_ip_IP_Version` | 4 | 6 | 0 |
| `client_ip` | `client_ip_IP_IPv4_1` | 192 | 0 | 0 |
| `client_ip` | `client_ip_IP_IPv4_3` | 2 | 0 | 0 |
| `client_ip` | `client_ip_IP_IPv4_4` | 200 | 0 | 0 |
| `client_ip` | `client_ip_IP_IPv6_1` | 0 | 8193 | 0 |
| `client_ip` | `client_ip_IP_IPv6_2` | 0 | 3512 | 0 |
| `client_ip` | `client_ip_IP_IPv6_8` | 0 | 200 | 0 |
| `client_ip` | `client_ip_IP_Missing` | 0 | 0 | 1 |
| `request_count` | `request_count` (unchanged) | 12 | 7 | 3 |

`client_ip_IP_IPv4_2` and `client_ip_IP_IPv6_3` through
`client_ip_IP_IPv6_7` are omitted because they are zero for every training row.
The original `client_ip` string column is removed from model inputs. No feature
contains a list, and `request_count` remains a single numeric column. Its values
are independent of whether the IP address is IPv4, IPv6, or missing.

For IPv6, `::` expands to the omitted zero groups. The hexadecimal groups `2001`,
`db8`, and `c8` become decimal values `8193`, `3512`, and `200` respectively.
Unused components for the other address version are zero-filled. The missing
indicator, when retained, distinguishes missing addresses from valid all-zero addresses such as
`0.0.0.0` and `::`.

### Constant columns are dropped after transformation

After expanding an IP column, AutoML keeps only generated columns with at least
two distinct values in the learner's training data. Constant columns are dropped
before scaling. This includes version, individual components, and the missing
indicator; none of them is always kept.

For example, if training contains only `192.0.2.1`, `192.0.2.2`, and `192.0.2.3`:

| Candidate feature | Training values | Outcome |
| --- | --- | --- |
| `client_ip_IP_Version` | Always `4` | Dropped |
| `client_ip_IP_IPv4_1` | Always `192` | Dropped |
| `client_ip_IP_IPv4_2` | Always `0` | Dropped |
| `client_ip_IP_IPv4_3` | Always `2` | Dropped |
| `client_ip_IP_IPv4_4` | `1`, `2`, `3` | Kept |
| All eight IPv6 components | Always `0` | Dropped |
| `client_ip_IP_Missing` | Always `0` | Dropped |

Each learner saves its selection from its training fold. Validation, prediction,
and reload reuse exactly those columns. Dropped columns are never added back,
even if new data would make them vary. Retained columns are not dropped when a
prediction batch happens to contain constant values.

A new IPv6 or missing address is accepted, but only the retained features reach
the model. In this IPv4-only example, no version, IPv6, or missing-indicator
feature is available to distinguish those cases.

The Decision Tree in this example uses these values directly. Algorithms that
require scaling use scaling learned from the training data. You continue to pass
raw addresses to `predict()`; AutoML constructs the numeric columns internally.

## Run the script

From a repository checkout with `mljar-supervised` installed:

```bash
python examples/scripts/binary_classifier_ip_addresses.py --results-path AutoML_IP
```

The [complete script](https://github.com/mljar/mljar-supervised/blob/master/examples/scripts/binary_classifier_ip_addresses.py)
checks prediction equality after reload. Use a fresh results directory for a new
training run.

## Supported inputs

- Every non-missing training value must be a valid plain IPv4 or IPv6 string.
- Surrounding whitespace is ignored; equivalent IPv6 spellings are accepted.
- Explicit pandas `category` columns retain categorical handling. To use that
  behavior, assign `X["client_ip"] = X["client_ip"].astype("category")` before fit.
- Mixed IP/non-IP training columns follow existing categorical/text handling.
- Once detected as IP, malformed prediction values raise a column/row error.
- CIDR notation, ports, and IPv6 zone identifiers are excluded.

See [Preprocessing](../features/preprocessing.md#ip-address-features) for generated
features and missing-value behavior, and [Save and Load models](../features/save-and-load-models.md)
for model persistence.
