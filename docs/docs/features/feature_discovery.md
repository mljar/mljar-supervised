# Feature discovery

Feature discovery uses the included `features-goldmine` package to generate up to
three feature sets and evaluate them with the best current models. It runs after
golden features, before KMeans features and subsequent tuning.

Starting with beta 1, feature discovery is included in the base package
and requires Python 3.11 or newer:

```bash
python -m pip install "mljar-supervised==1.4.0b1"
```

No extra tag is needed. `[feature-discovery]` is retained as a compatibility alias.
See the [installation guide](../install.md) for supported environments.

```python
from supervised import AutoML

automl = AutoML(
    mode="Compete",
    total_time_limit=3600,
    golden_features=True,
    feature_discovery=True,
)
automl.fit(X, y)
```

`feature_discovery="auto"` enables the step in Compete when its dependency and
validation configuration are supported. Other modes require explicit `True`.
`False` disables it. Automatic selection prints a skip reason when unavailable;
explicit `True` raises an actionable error.

## Search and time allocation

The feature-engineering scheduling pool is shared in 30/50/20 proportions between
golden features, feature discovery and KMeans. With all three enabled, their
weights are 0.0375, 0.0625 and 0.025 respectively; mixed encoding retains 0.025.
Disabling a component removes its previous budget contribution, and the remaining
feature-engineering proportions are renormalized. See [Modes](modes.md) for the
complete stage allocation. Weights are normalized against other enabled steps,
not literal percentages of total runtime.

Discovery receives up to one quarter of its step allocation across all folds;
the rest is available for training. Unused time from earlier steps remains
available through the cumulative scheduler. Without a total budget, discovery
uses one quarter of `model_time_limit`, or 15 seconds if neither limit is set.
For tuning candidates with measured training costs, admission checks the time
remaining in the current step, including elapsed work outside model training.
This helps preserve later steps' allocations. Training costs are estimates;
slower fits and noninterruptible work can still overrun.

The step snapshots up to three best eligible individual models, respecting a
smaller positive `top_models_to_improve` setting. It retains each parent's chosen
hyperparameters and preprocessing, including existing golden features. Before
generating features, it reserves the discovery allowance and filters out parents
whose measured training time exceeds the remaining training allowance. Faster
models further down the leaderboard can then be selected. If no eligible parent
fits, generation is skipped with a console explanation. Each candidate is checked
again before training, using its parent's cost rather than the algorithm-wide
average.

Discovery uses the original numeric/categorical inputs; generated columns are
added before the learner's categorical encoding and scaling. Each profile is tried across
parents in rank order before trying the next profile. Nine models is a maximum,
not a guarantee: unavailable/empty/duplicate sets and insufficient time reduce it.
Original models remain available for validation comparison and ensembling.

Names identify the profile and inherited preprocessing:

```text
24_LightGBM_FeatureDiscovery_Default
25_CatBoost_GoldenFeatures_FeatureDiscovery_Default
26_LightGBM_FeatureDiscovery_Compact
```

Goldmine may return `default`, `compact`, `diverse` or `fast`; at most three are
requested. Only profiles available across the required folds are tried. Features
and fitted values may differ between folds. Successful models participate in
subsequent tuning, feature selection, ensembling and stacking.

## Training, prediction and resource use

A separate generator is fitted inside each outer training fold. Identical fold
data and discovery settings share the result across candidates. Training uses
Goldmine's out-of-fold target encodings; validation and prediction use frozen
mappings. The original target is passed before learner-specific target scaling
or encoding. Discovery is unweighted; sample weights still apply to downstream
model training and evaluation.

Initially, shuffled split and K-fold validation are supported. Custom/grouped or
time-ordered validation requires a compatible encoding policy and is disabled.
Ordinary shuffled folds assume independent rows.

Goldmine automatically limits discovery rows, columns, candidates and retained
state. Neighbor features and ICA are excluded from its defaults. Native threads
respect AutoML's `n_jobs`. These controls do not bound total process RSS or the
size of returned training matrices.

## Saved artifacts and reports

With `verbose=1` (the default), training prints fold progress, feature counts per
profile, newly listed feature names, and the path to each transformations report.
Shared features are listed once per discovery run, and cached folds are marked as
reused. Large feature lists show the first 20 new names per profile; the report
contains the complete definitions. `verbose=0` disables these console messages.

Artifacts live under `feature_discovery/<identity>/` in the AutoML results folder:

- `generator.joblib`: frozen transformations and inference state.
- `generator.md` and `generator.json`: formulas, dependencies, limits and decisions.
- `manifest.json`: input fingerprint, profiles and output-column mapping.
- `training.joblib`: shared training encodings, retained during interrupted search
  and removed after successful fitting. Inference does not need this file.

Model reports link to the corresponding fold reports and name the parent and
selected profile. Copy the **complete AutoML results directory** for deployment;
shared discovery artifacts use relative paths. Reload does not rerun discovery.
Use matching dependencies and only load trusted joblib artifacts.

New training-cache fingerprints compare column values, types, categorical
metadata, row indices and targets, independently of pandas' internal storage
layout. Selecting or copying equivalent columns does not invalidate the cache;
changed training values, targets or row order do. Manifests record the fingerprint
version. Inference from existing artifacts continues to use the frozen generator
and does not require a training-cache fingerprint.

Feature discovery is an additional search strategy, not a guarantee of better
accuracy. Keep original-feature candidates and select using validation results.
