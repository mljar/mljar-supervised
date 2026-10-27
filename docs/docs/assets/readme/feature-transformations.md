# Features Goldmine report

## Run

```json
{
  "task": "regression",
  "random_state": 1234,
  "n_jobs": 2,
  "time_limit": 2.6742437253706157,
  "max_feature_sets": 3,
  "max_selected_features": null,
  "selectivity": "balanced",
  "enabled_strategies": [
    "categorical_frequency",
    "categorical_group_deviation",
    "categorical_hash_cross",
    "categorical_oof_target",
    "categorical_prototypes",
    "grouped_row_stats",
    "path",
    "projection_pca",
    "residual_numeric"
  ]
}
```

Elapsed: 0.556 seconds. Stop reason: completed.
The time limit is cooperative; a noninterruptible operation or completion of training outputs can overrun it.

Target encoding uses out-of-fold training values; transform always uses frozen inference state. Discovery and feature selection must run inside each outer training fold.

## Feature sets

- **default** (8 features): `length_mul_width`, `material_te`, `material_freq`, `rule_length_gt_5p786_and_width_gt_3p215_009`, `grpstat_001_mean`, `grpstat_001_min`, `gde_length_z_material`, `cproto_material_zabs_mean`
  Measured transform: 0.010519 seconds for 256 rows on this machine.
- **compact** (4 features): `length_mul_width`, `material_te`, `material_freq`, `rule_length_gt_5p786_and_width_gt_3p215_009`
  Measured transform: 0.003149 seconds for 256 rows on this machine.

## Resource decisions

```json
{
  "training_rows": 300,
  "discovery_rows": 300,
  "discovery_columns": 3,
  "retained_reference_rows": 0,
  "fitted_state_bytes": 1690,
  "state_limit_bytes": 10485760,
  "decisions": [
    {
      "stage": "categorical_hash_cross",
      "reason": "No eligible candidates (data size, cardinality, task, or variance)"
    },
    {
      "stage": "projection_pca",
      "reason": "No eligible candidates (data size, cardinality, task, or variance)"
    }
  ]
}
```

The state limit covers uncompressed inference definitions and support state. It is not a process RAM or total artifact-size guarantee; metadata and reports add overhead.

## Transformations

Selection gain/rank statistics are proxies, not measured improvements in downstream accuracy.

### length_mul_width

'length' * 'width'

Inputs: `length`, `width`

Non-finite outputs use saved fill value: `13.30855210869069`.

Complete operation parameters and fitted metadata:

```json
{
  "name": "length_mul_width",
  "definition_version": 1,
  "source_columns": [
    "length",
    "width"
  ],
  "formula_name": "mul",
  "feature_type": "numeric_formula",
  "metadata": {
    "interaction_score": 2893633.817438013,
    "pair_count": 1584,
    "pair_total_gain": 4133003.8586506248
  },
  "output_fill": 13.30855210869069,
  "strategy": "lightgbm_paths",
  "mean_gain": 183595.94158211848,
  "median_rank": 1.0,
  "top_frequency": 1.0
}
```

### material_te

Smoothed target mean (smoothing=20). Training uses out-of-fold mappings and fold-local fallback means; inference uses the full-training mapping and saved fallback.

Inputs: `material`

Non-finite outputs use saved fill value: `16.314475800573998`.

Complete operation parameters and fitted metadata:

```json
{
  "name": "material_te",
  "definition_version": 1,
  "source_columns": [
    "material"
  ],
  "formula_name": "oof_target_mean",
  "feature_type": "categorical_oof_target",
  "metadata": {
    "column": "material",
    "mapping": {
      "A": 15.691620267763753,
      "B": 19.389088339350465,
      "C": 16.076882418535018
    },
    "fallback": 16.957953294811347,
    "task": "regression"
  },
  "output_fill": 16.314475800573998,
  "strategy": "categorical_oof_target",
  "mean_gain": 906.6158102750778,
  "median_rank": 6.0,
  "top_frequency": 1.0
}
```

### material_freq

Lookup of the full-training category frequency; unseen values use the saved fallback.

Inputs: `material`

Non-finite outputs use saved fill value: `0.32666666666666666`.

Complete operation parameters and fitted metadata:

```json
{
  "name": "material_freq",
  "definition_version": 1,
  "source_columns": [
    "material"
  ],
  "formula_name": "frequency",
  "feature_type": "categorical_frequency",
  "metadata": {
    "column": "material",
    "mapping": {
      "A": 0.37,
      "C": 0.32666666666666666,
      "B": 0.30333333333333334
    },
    "fallback": 0.0
  },
  "output_fill": 0.32666666666666666,
  "strategy": "categorical_frequency",
  "mean_gain": 234.88198437293372,
  "median_rank": 8.0,
  "top_frequency": 0.6666666666666666
}
```

### rule_length_gt_5p786_and_width_gt_3p215_009

1 when 'length' > 5.785713399991573 AND 'width' > 3.215040708902418; otherwise 0 (including missing comparisons)

Inputs: `length`, `width`

Non-finite outputs use saved fill value: `0.0`.

Complete operation parameters and fitted metadata:

```json
{
  "name": "rule_length_gt_5p786_and_width_gt_3p215_009",
  "definition_version": 1,
  "source_columns": [
    "length",
    "width"
  ],
  "formula_name": "path_rule",
  "feature_type": "binary_rule",
  "metadata": {
    "conditions": [
      {
        "feature": "length",
        "op": ">",
        "threshold": 5.785713399991573
      },
      {
        "feature": "width",
        "op": ">",
        "threshold": 3.215040708902418
      }
    ],
    "path_gain": 21773.879638671875,
    "path_depth": 3
  },
  "output_fill": 0.0,
  "strategy": "lightgbm_paths",
  "mean_gain": 7564.153317689896,
  "median_rank": 4.0,
  "top_frequency": 1.0
}
```

### grpstat_001_mean

Row-wise mean over the saved group columns; non-finite inputs are treated as missing.

Inputs: `length`, `width`

Non-finite outputs use saved fill value: `4.140760472481686`.

Complete operation parameters and fitted metadata:

```json
{
  "name": "grpstat_001_mean",
  "definition_version": 1,
  "source_columns": [
    "length",
    "width"
  ],
  "formula_name": "group_mean",
  "feature_type": "grouped_row_stats",
  "metadata": {
    "group_index": 1,
    "group_columns": [
      "length",
      "width"
    ],
    "stat": "mean"
  },
  "output_fill": 4.140760472481686,
  "strategy": "grouped_row_stats",
  "mean_gain": 8128.999034126599,
  "median_rank": 3.0,
  "top_frequency": 1.0
}
```

### grpstat_001_min

Row-wise min over the saved group columns; non-finite inputs are treated as missing.

Inputs: `length`, `width`

Non-finite outputs use saved fill value: `2.3563555626923316`.

Complete operation parameters and fitted metadata:

```json
{
  "name": "grpstat_001_min",
  "definition_version": 1,
  "source_columns": [
    "length",
    "width"
  ],
  "formula_name": "group_min",
  "feature_type": "grouped_row_stats",
  "metadata": {
    "group_index": 1,
    "group_columns": [
      "length",
      "width"
    ],
    "stat": "min"
  },
  "output_fill": 2.3563555626923316,
  "strategy": "grouped_row_stats",
  "mean_gain": 1512.8713727792103,
  "median_rank": 5.0,
  "top_frequency": 1.0
}
```

### gde_length_z_material

group_z: category lookup of saved group statistics, using saved global fallbacks for unseen categories.

Inputs: `length`, `material`

Non-finite outputs use saved fill value: `0.00016663915831996706`.

Complete operation parameters and fitted metadata:

```json
{
  "name": "gde_length_z_material",
  "definition_version": 1,
  "source_columns": [
    "length",
    "material"
  ],
  "formula_name": "group_z",
  "feature_type": "categorical_group_deviation",
  "metadata": {
    "column_cat": "material",
    "column_num": "length",
    "mapping_mean": {
      "A": 5.430053613263861,
      "B": 5.540494255163145,
      "C": 5.556943477999253
    },
    "mapping_std": {
      "A": 2.4957661861567653,
      "B": 2.4504602941808593,
      "C": 2.683810836627407
    },
    "fallback_mean": 5.50569782240163,
    "fallback_std": 2.539059591319262
  },
  "output_fill": 0.00016663915831996706,
  "strategy": "categorical_group_deviation",
  "mean_gain": 244.48280399044356,
  "median_rank": 9.0,
  "top_frequency": 1.0
}
```

### cproto_material_zabs_mean

prototype_zabs_mean: distance from the saved category centroid, with saved global centroid for unseen categories.

Inputs: `material`, `length`, `width`

Non-finite outputs use saved fill value: `22.06493331050745`.

Complete operation parameters and fitted metadata:

```json
{
  "name": "cproto_material_zabs_mean",
  "definition_version": 1,
  "source_columns": [
    "material",
    "length",
    "width"
  ],
  "formula_name": "prototype_zabs_mean",
  "feature_type": "categorical_prototype",
  "metadata": {
    "column_cat": "material",
    "anchor_columns": [
      "length",
      "width"
    ],
    "prototype_map": {
      "A": {
        "length": 5.416424026031831,
        "width": 2.8254847730191397
      },
      "B": {
        "length": 5.5481418228030375,
        "width": 2.9371488820323153
      },
      "C": {
        "length": 5.5674017750599925,
        "width": 2.919101646241434
      }
    },
    "global_prototype": {
      "length": 5.50569782240163,
      "width": 2.889937731339086
    },
    "anchor_std": {
      "length": 0.08217339842147976,
      "width": 0.059942604371900576
    }
  },
  "output_fill": 22.06493331050745,
  "strategy": "categorical_prototypes",
  "mean_gain": 745.2445792059103,
  "median_rank": 7.0,
  "top_frequency": 1.0
}
```

## Rejections

```json
{
  "count": 25,
  "reason_counts": {
    "duplicate_candidate": 22,
    "almost_identical_to_parent": 3
  },
  "by_feature": {
    "rule_length_gt_5p786_and_width_le_3p215_001": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_le_3p215_002": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_le_3p215_003": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_le_3p215_004": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_le_3p215_005": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_le_3p215_006": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_le_3p215_007": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_le_3p215_008": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_gt_3p215_010": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_gt_3p215_011": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_gt_3p215_012": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_gt_3p215_013": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_gt_3p215_014": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_le_3p215_015": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_le_3p215_016": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_le_3p215_017": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_le_3p215_018": "duplicate_candidate",
    "rule_length_gt_5p786_and_width_le_3p215_019": "duplicate_candidate",
    "gde_length_minus_material_mean": "almost_identical_to_parent",
    "gde_width_minus_material_mean": "almost_identical_to_parent",
    "gde_width_z_material": "almost_identical_to_parent",
    "res_width_mul_length": "duplicate_candidate",
    "res_width_div_length": "duplicate_candidate",
    "res_length_div_width": "duplicate_candidate",
    "res_width_absdiff_length": "duplicate_candidate"
  }
}
```

## Saved artifact

```json
{
  "file": "generator.joblib",
  "serialization_version": "2",
  "bytes": 6107,
  "sha256": "787a87ed92b1f29416fabd49091b5e967354e27b7692f6e7248976f44c615c71"
}
```

The joblib file restores inference. JSON and Markdown describe the work; they are not executable replacements for the artifact.
Load only trusted joblib files. The returned out-of-fold training matrix is not saved.

```python
from features_goldmine import GoldenFeatures
gf = GoldenFeatures.load('generator.joblib')
features = gf.transform(X_new, feature_set="all")
```
