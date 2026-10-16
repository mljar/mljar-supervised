## mljar-supervised 1.4.0rc4

Fourth release candidate for 1.4.0, fixing prediction consistency and
feature-discovery scheduling and improving SDM inference.

```bash
python -m pip install "mljar-supervised[feature-discovery]==1.4.0rc4"
```

- Fix feature-discovery fingerprints so saved transformations can be replayed with equivalent data representations.
- Keep unseen-category encoding independent of prediction batch contents, including boolean categories, and preserve mappings after reload.
- Reuse Kumo/TabICLv2 training-context computation across prediction batches within a fold, subject to memory checks. Release temporary caches after each prediction session, including failures.
- Keep FP16 as the default SDM precision on CUDA and FP32 on CPU. Allow resource-checked FP32 candidates during hill climbing, persist the selected precision, and stabilize FP16 query batch shapes for consistent predictions.
- Check candidate training costs against the remaining stage wall-clock budget. Feature discovery selects affordable parents before generating features and reserves time for model training.
- When both mixed encoding and feature discovery run, transfer half of the mixed-encoding scheduling weight to discovery. Other stages retain their allocations; with all Compete stages enabled, discovery receives approximately 3.70% of the total budget.
- Correct the RealMLP/TabM smoke-test configuration so automatic feature discovery does not change the expected model count.
- Update the benchmark launcher for RC4, retaining earlier environments and artifacts and supporting comparisons against multiple saved releases. Selected-model evaluation is the default; all-model predictions remain optional.

Native categorical changes for LightGBM and XGBoost, and the related categorical
preprocessing experiment, are deferred until after the stable release.

## mljar-supervised 1.4.0rc3

Third release candidate for 1.4.0, adding feature discovery and improving saved
ensemble/stacked-model prediction memory use.

```bash
python -m pip install "mljar-supervised[feature-discovery]==1.4.0rc3"
```

- Add the separate `feature_discovery` step after golden features. Generate up to three feature sets with Goldmine and try them on up to three best current models, within the existing shared feature-engineering time allocation.
- Fit discovery inside each training fold, reuse training encodings across compatible candidates, and persist frozen transformations with portable Markdown/JSON reports. Models use the `_FeatureDiscovery_<SetName>` suffix and participate in subsequent tuning, ensembling and stacking.
- Enable discovery automatically in Compete when the optional `features-goldmine>=1.1,<1.2` dependency, Python 3.10+, and shuffled split/K-fold validation are available. Custom/grouped/time-ordered encodings are not supported. The base installation retains Python 3.9 support; `[all]` includes discovery on supported Python versions.
- Update the TabArena launcher for RC3, retaining older environments and results, and pin features-goldmine 1.1.0 in benchmark environments.

- Load saved AutoML learners lazily and release each saved fold's runtime model after prediction, including failures, to prevent fold networks accumulating in GPU memory.
- Reuse model predictions within each stacking/ensemble request while preserving preprocessing, fold averaging, ensemble weights, and output labels. Existing artifacts work without retraining; separate requests compute fresh predictions.
- Preserve Kumo's FP32 RMS normalization fallback for mixed-precision inputs without PyTorch's dtype-mismatch warning. Keep the original checkpoint weights, output dtype and eligible SDM optimized kernels.
- Add normalization regression checks for FP16/BF16/FP32, empty and strided inputs, all Kumo model sizes, checkpoint compatibility and opt-in CUDA execution.

## mljar-supervised 1.4.0rc2

Second release candidate for 1.4.0, fixing console noise and GPU memory retention
observed while benchmarking the optional Compete models.

```bash
python -m pip install "mljar-supervised==1.4.0rc2"
```

### Fixes and improvements

- Release completed early-stopping checkpoint copies, including shared Kumo/TabICLv2 GPU networks, so later folds and candidates can reuse their memory.
- Collect unreachable tensors and release unused CUDA cache before GPU admission checks. TabM, Kumo and TabICLv2 rejection messages now include the memory estimate, budget and available memory; genuine shortages still reject the candidate.
- Fix pandas slice-assignment warnings and missing-value imputation writing through shared data, while preserving fold isolation without copying the full training dataset.
- Use scoped compatibility bridges for pytabkit's deprecated scikit-learn finite-check keyword and Lightning's deprecated PyTorch leaf-spec construction. Restore upstream functions and settings after fitting or prediction, including failures.
- Disable routine Lightning INFO output during MLJAR calls. Keep other backend warnings/errors visible and avoid the GPU-available advisory when a portable RealMLP model intentionally predicts on CPU.
- Record TabM's valid single-bin numerical encoding as `single_bin_feature_count` in training diagnostics instead of repeating its advisory in the console. Encoding and predictions are unchanged.
- Keep RealMLP prediction logging disabled with pytabkit 1.7.2, matching its training configuration.

### Validation and compatibility

- Added regression tests for preprocessing ownership, warning/logging behavior, checkpoint memory release, and GPU admission cleanup.
- Validated Python 3.9 and pandas 3 preprocessing, both supported pytabkit releases, GPU training/reload, and a short 10-fold credit-g run with RealMLP, TabM, Kumo and TabICLv2.
- Optional-model installation requirements, default model configurations and batch sizes are unchanged from RC1. This validation is not a full performance benchmark.

## mljar-supervised 1.4.0rc1

First release candidate for 1.4.0. Install this version explicitly to test it:

```bash
python -m pip install "mljar-supervised==1.4.0rc1"
```

### New features

- Added `AutoML.get_feature_importance()` for the best model, a named model or all models. It reuses saved permutation importance and can compute missing values when evaluation data is supplied. Raw and normalized results are supported. [#809](https://github.com/mljar/mljar-supervised/issues/809)
- Added optional RealMLP and TabM mini models in Compete mode, with CPU/NVIDIA GPU support, resource controls and fixed `custom_params` overrides. [#841](https://github.com/mljar/mljar-supervised/issues/841), [#842](https://github.com/mljar/mljar-supervised/issues/842)
- Added optional Kumo and TabICLv2 pretrained models in Compete mode, with bounded training context, checkpoint persistence and offline reload. [#843](https://github.com/mljar/mljar-supervised/issues/843)
- Added model blueprints to reports and separated stacked models in leaderboard plots. [#123](https://github.com/mljar/mljar-supervised/issues/123), [#102](https://github.com/mljar/mljar-supervised/issues/102)
- Added automatic IP address preprocessing, duration-to-seconds conversion, and explicit F1 averaging options. [#23](https://github.com/mljar/mljar-supervised/issues/23), [#650](https://github.com/mljar/mljar-supervised/issues/650), [#646](https://github.com/mljar/mljar-supervised/issues/646)

### Fixes and improvements

- Fixed categorical binary-target encoding, boolean-label consistency after reload, and categorical preprocessing across validation folds.
- Fixed binary stacking column names used by fitted preprocessing, including KMeans features.
- Made saved artifact paths portable across platforms and fixed reloaded ensemble losses, Optuna resume and `need_retrain()` behavior.
- Fixed Nearest Neighbors subsampling for rare classes and skipped expensive interpretation on large datasets.
- Fixed deprecated Decision Tree regression criteria and KMeans improvement handling when no eligible models exist.
- Improved training-data memory lifecycle, metadata preparation, report link cursors and legacy report-cache refresh.
- Enforced the abstract learner contract and declared `psutil` as a test dependency for resource-limit checks.

### Installation and compatibility

- Standard MLJAR continues to support Python 3.9+ and does not require PyTorch.
- RealMLP and TabM mini are optional; use `[realmlp]`, `[tabm]` or `[all]`. Their CPU setup is tested on Python 3.11 and 3.12. Install the appropriate PyTorch build first.
- Kumo and TabICLv2 require Python 3.11+, `[tfm]` (or `[kumo]`, `[tabiclv2]`, `[all]`) and a separate pinned SDM backend installation. See the [installation guide](docs/docs/install.md).
- These four optional algorithms currently support Compete mode. RealMLP and TabM become automatic candidates when their dependencies are installed; Kumo and TabICLv2 also require CUDA for automatic selection. CPU selection for Kumo/TabICLv2 is explicit.
- `custom_params` overrides remain unsupported in Optuna mode. These optional models do not support sample weights.

Please report RC testing results with the package/Python versions, operating system, selected algorithms and a minimal reproduction when possible.

## mljar-supervised 1.3.1

- Fixed `publish_app()` to upload `mljar_app.json` and reuse the last successfully published app URL by default instead of creating a new URL on every publish.
- Improved generated app numeric widgets by inferring friendlier steps from training data, with integer-style inputs for count-like fields such as age or number of children.
- Added subtle AI disclosure text to generated apps, generated app `README.md`, generated notebooks, AutoML reports, structured markdown reports, and training console output.
- Updated app documentation to describe the new `publish_app()` URL reuse behavior.
- Relaxed brittle sample-weight score equality tests to check for close behavior instead of exact end-to-end AutoML score matches.

## mljar-supervised 1.3.0

- Added `AutoML.app()`, `local_app()`, and `publish_app()` workflows for generating and sharing Mercury-based prediction apps.
- Added fairness certificates to fairness-aware reports and structured report outputs.
- Improved generated app UX: clearer startup instructions, better Mercury availability messaging, batch-only app generation for wide datasets, and cleaner single-prediction dashboards.
- Added published-app state tracking and friendlier publish progress/error output.
- Moved docs into this repository, refreshed docs build tooling, and added a docs GitHub Actions workflow.

## mljar-supervised 1.2.0

- Added `AutoML.report_structured(format="markdown"|"dict"|"json", model_details=...)` for LLM-friendly reporting.
- New on-demand artifact: `<results_path>/report_structured.json` (full structured payload).
- Improved markdown report readability (clean run summary, clearer best/model sections, better feature-importance naming).
- Added examples for classification, regression, and fairness structured reports.
- Updated structured-report tests and fixed macOS smoke workflow by installing `libomp`.
