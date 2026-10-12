# SDM synthetic validation

Install the optional backend using [the installation guide](../../docs/docs/features/sdm.md).
Both checkpoints and their notices must be cached before running offline tests.

```bash
MLJAR_TEST_SDM=1 MLJAR_TEST_SDM_DEVICE=cuda python -m pytest \
  tests/tests_algorithms/test_sdm.py tests/tests_automl/test_sdm.py \
  tests/tests_tuner/test_sdm.py -q
# Repeat with MLJAR_TEST_SDM_DEVICE=cpu for CPU validation.
python examples/scripts/sdm_synthetic.py --device cuda --rows 1000
python benchmarks/sdm/context_profile.py --rows 1000000 --context 4096 --device cuda
```

Integration tests exercise actual pretrained weights, three tasks, 20 classes,
missing/categorical data, rare-class context sampling, validation isolation,
query batching, portable offline reload, shared weights, resource/time rejection,
Compete scheduling, weighted-run exclusions, reports, and permutation importance.
Ordinary tests skip checkpoint inference unless `MLJAR_TEST_SDM=1` is set.

To exercise all three Kumo sizes (larger HF downloads), cache the classifier and
regressor checkpoints for each size, then run:

```bash
MLJAR_TEST_KUMO_SIZES=1 MLJAR_TEST_SDM_DEVICE=cuda python -m pytest \
  tests/tests_algorithms/test_sdm.py -k all_kumo_sizes -q
```

These tests cover binary, multiclass, regression, and save/reload for each size.
The SDM workflow also provides an `all_kumo_sizes` manual input.

## Observed synthetic scores

Local sklearn-generated run with 1,000 rows, 20 numeric features plus a categorical
feature and missing values, seed 42. An 80/20 outer split separates the test set;
AutoML uses a further 80/20 train/validation split. Four estimators, maximum context
1,024, one default model per algorithm, no feature engineering or ensemble.
Regression uses nonlinear `make_friedman1` data with noise 1.

| Task / held-out metric | Kumo small | TabICLv2 |
|---|---:|---:|
| Binary AUC | 0.9888 | 0.9883 |
| Three-class accuracy | 0.920 | 0.925 |
| Regression R² | 0.9405 | 0.9369 |

All six AutoML results reproduced their predictions after reload. This is an
integration check, not a TabArena comparison or a claim about general ranking.
Raw local results: `performance-results/sdm/synthetic-gpu/summary.json`.

## Large-input profile

NVIDIA RTX 3070, PyTorch 2.11.0+cu128, four CPU threads, pinned SDM revision from
the installation guide. One million generated training rows, 20 features,
**4,096 retained context rows**, four estimators, prediction batches of 128.
Validation and timed prediction both use the same 1,024 held-out rows.

| Measurement | Kumo small | TabICLv2 |
|---|---:|---:|
| Context preparation + validation | 6.58 s | 4.16 s |
| Predict 1,024 rows | 3.39 s | 3.74 s |
| Peak PyTorch reserved GPU memory | 564 MiB | 548 MiB |
| Process peak RSS | 2.19 GiB | 2.26 GiB |

The first model includes cold-start costs. RSS is a process-wide high-water mark
and includes generated input and imports; it is cumulative across the two models.
PyTorch reserved memory excludes CUDA driver/context allocations. Admission
estimates budget additional working memory, not total process RSS. These timings
are smoke profiles, not controlled performance comparisons.

This does **not** measure full-million-row context or prediction over one million
queries. The full AutoML flow also computes training-set predictions and can take
much longer; its learner deadline covers those predictions.
Raw local results: `performance-results/sdm/profile-final.json`.
