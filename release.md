## mljar-supervised 1.4.0rc6

One AutoML for classic machine learning, neural networks, and tabular foundation
models. This release candidate brings the changes tested in local beta builds
into the public 1.4 release series. Requires Python 3.11+.

```bash
python -m pip install "mljar-supervised==1.4.0rc6"
```

Choose your PyTorch build before adding `[realmlp]`, `[tabm]`, or `[tfm]` for
RealMLP, TabM mini, Kumo, and TabICLv2. `[all]` adds all optional model backends;
`[apps]` adds the runtime for exported prediction apps. See the
[installation guide](docs/docs/install.md).

### Validation and training

- Score cross-validation models and ensembles by the arithmetic mean of their
  fold scores across all repeats. Averaging repeated OOF predictions no longer
  substitutes for averaging the validation metrics used for model selection.
- Select a shared RealMLP epoch across CV folds and repeats by default. Evaluate
  checkpoints at epochs 10, 20, …, 250, 256 for the default training schedule,
  then restore each fold's own weights at the chosen epoch. Save the selection
  curve, verify restored predictions, and clean up temporary checkpoints.
- Use mean fold scores for shared XGBoost, LightGBM, and CatBoost round selection.
- For eligible small datasets, automatic validation uses five folds and three
  repeats, with stacking disabled. Dataset size, class counts, time budget, and
  explicit validation settings still determine eligibility.
- Reject incomplete candidates from final model selection and ensembling.
  Use completed-fold costs to decide whether to continue a candidate and reduce
  wasted training. Keep routine continuation messages out of the console.

### Compete defaults and saved models

- Keep golden features enabled. Disable KMeans features, feature discovery, and
  boost-on-errors by default. Each remains available through its explicit option.
- Give generated features separate semantic identities, storage names, and
  display labels; preserve their fitted meaning when replaying saved models.
- Store KMeans artifacts separately for each CV fold and repeat, fixing changed
  predictions after reloading repeated-CV models.
- Use fold models to evaluate Kumo and TabICLv2, and one separately prepared
  full-training-data context for inference. Retain their predictions as ensemble
  and stacking inputs without fitting new TFM models on stacked features.
- Improve resource admission, expected budget-skip reporting, and PyTorch input
  handling. Keep failed candidates out of the ensemble.
- Include Python 3.11+ installation extras, exported-app dependencies, native
  compatibility checks, and documentation updates developed during the betas.

### Known limitations

This is a release candidate. Validation remains part of adaptive model and
feature selection; it is not an unbiased estimate of final test performance.
The deferred validation-boundary work for legacy golden features, feature
selection, and optional boost-on-errors is not resolved by this release. Use a
separate held-out test set for final evaluation. No general accuracy improvement
is claimed from the partial local benchmark comparison.

Historical entries below describe the earlier local candidates and published
betas. The current behavior is summarized above.

## mljar-supervised 1.4.0b4 (unreleased)

- Give discovered features separate semantic identities, storage names and display
  labels. Scope learned features to their fitted generators, align importance by
  identity, and block shared generated-column selection in new training and TFM
  full-data preparation. Preserve legacy model replay and training encodings.
- Keep Kumo and TabICLv2 predictions as inputs to stacking and ordinary
  ensembles, but do not train new TFM models on stacked features.
- For TFM cross-validation, preserve fold-based OOF scores and artifacts, then
  use one full-training-data context for inference. Prepare full-context
  preprocessing separately, enforce full-data resource limits, and include
  preparation time in the training budget. Holdout and older saved runs retain
  their original prediction behavior.
- Cap automatic cross-validation at five folds below 5,000 training rows or,
  for classification, below 250 samples in the smallest observed class.
  Use cleaned, unweighted counts before oversampling. Preserve explicit
  validation strategies, existing mode defaults and time-budget constraints.
  Save the automatic CV policy and its reason with the run.

## mljar-supervised 1.4.0b3 (unreleased)

- Select one shared boosting-round count across CV folds for XGBoost, LightGBM
  and CatBoost. Use 10-round checkpoints and 300-round patience, retain complete
  fold boosters, and select only prefixes available in every fold by pooled OOF
  performance. Update saved models and OOF predictions before ensembling.
- Report mean fold scores, shared round counts and selection curves. Keep
  checkpoint predictions on disk and clean them up on success or failure.
- Use fixed 100/200/300-tree Random Forest and Extra Trees models based on the
  complete training/validation row count instead of selecting per-fold sizes.

## mljar-supervised 1.4.0b2 (unreleased)

Local beta 2 builds require Python 3.11+ and retain beta 1's base dependencies
and optional model/app extras. The README's measured tutorial results remain
pinned to the published beta 1 used to produce them.

- Disable stacking for classification when the smallest observed class has less
  than 15% of training rows, including explicit stacking requests. Use unweighted
  counts after dropping missing targets. Keep regression and ordinary ensembles
  unchanged; exactly 15% remains eligible subject to other constraints.
- Keep TFM training contexts complete and skip oversized fits instead of silently
  reducing the training context.
- Print the MLJAR jar header and installed version at the start of training;
  `verbose=0` remains quiet.
- Refresh the README, beginner tutorials, example scripts, report media, and
  citation. Improve docs navigation and keep generated churn dataset metadata
  synchronized with its source.
- Align the SDM oversized-feature test with the structured resource-skip error,
  including its diagnostic metadata.
- Make the KMeans/stacking prediction regression test independent of optional
  feature discovery, model-ranking competition, and wall-clock search budgets.

This entry describes the candidate artifact. Publication and final platform
validation are separate steps; see the retained release-validation report.

## mljar-supervised 1.4.0b1

First beta release for testing the 1.4.0 installation and prediction workflows.
Requires Python **3.11+**; Python 3.11 and 3.12 are tested. Install the exact
version explicitly: beta versions sort before the previously published RCs.

```bash
python -m pip install "mljar-supervised==1.4.0b1"
```

Feature discovery is included in this base installation. Choose a PyTorch build
before adding optional neural/TFM extras; use `[apps]` to run exported apps.
See the [installation guide](docs/docs/install.md) for CPU/CUDA commands.

- Add `[apps]` with `mercury>=3.2.12,<4`, keeping `[all]` limited to optional model backends. Export requirements for the selected prediction graph, pin MLJAR/joblib/scikit-learn versions, and bundle shared discovery transformations plus TFM weights/licenses for portable predictions.

- Require Python 3.11+ for the base package, all extras, and newly generated prediction apps. Test Python 3.11 and 3.12; older releases remain available for older interpreters.
- Include `features-goldmine>=1.1,<1.2` in the base installation. Keep `[feature-discovery]` as an empty compatibility alias and preserve existing mode/validation rules and `feature_discovery=False`. Neural/TFM models and the app runtime remain optional.

- Install `structured-data-models==0.2.1` from PyPI through `[tfm]`, `[kumo]`, `[tabiclv2]`, or `[all]`. No separate Git backend installation is needed for beta 1.
- Keep the exact backend-version check and existing pinned model checkpoints. Weights still download on first use.
- The published `1.4.0a1` retains its original Git backend requirement; use a fresh environment when upgrading to beta 1.
- Add a macOS compatibility probe before mixing PyTorch with the standard native ML libraries. All probe cases share a three-second deadline. Automatic selection can skip optional Torch models; explicit requests and incompatible saved models raise `NativeCompatibilityError` before execution. Decisions appear in run reports. Training stays in the main process; Linux and Windows retain their existing paths.

- Correct saved-model test metadata, quoted app-path expectations on Windows, and simulated GPU allocation diagnostics. Preserve test reports and separate app CI stages for easier diagnosis.

Known limits: optional Torch models can be refused on incompatible macOS native
runtimes; the guard detects unsafe execution rather than repairing the installed
libraries. Apple MPS is not supported by these integrations. Local CUDA checks
cover Linux, an RTX 3070, PyTorch 2.11.0+cu128, and SDM 0.2.1, including all three
Kumo sizes. Python versions newer than 3.12 are not yet validated.

## mljar-supervised 1.4.0a1

Alpha release for testing the accumulated RC6/RC7 changes. Install this exact
version explicitly: alpha versions sort before release candidates.

```bash
python -m pip install "mljar-supervised[all]==1.4.0a1"
```

- Install CPU-only XGBoost on Linux x86-64/ARM64 and Windows x86-64, avoiding the CUDA-driver initialization failures observed during CPU multiclass evaluation. Other platforms retain their standard XGBoost package.
- Document migration from overlapping XGBoost distributions, and check CPU multiclass training, evaluation and prediction before the alpha benchmark starts. GPU support for the optional neural and pretrained algorithms is unchanged.
- Include algorithm early stopping, whole-trial budget accounting, measured stacking-cost admission, fixed ensemble/stacking limits, inference fixes, 30/50/20 feature-engineering allocation and readable feature-discovery training output from RC6/RC7.
- Add a full 12-dataset alpha benchmark using retained models, logs and results, with earlier version outputs preserved for comparison.

Use a fresh environment or follow the [XGBoost migration instructions](docs/docs/install.md#xgboost-cpu-installation). Pip does not remove an existing `xgboost` distribution when installing `xgboost-cpu`.

## mljar-supervised 1.4.0rc7 (local candidate)

Local candidate for validation and benchmarking; not published to PyPI.

- Add automatic algorithm early stopping to ordinary Compete runs. After three distinct successful configurations, pause original-feature random search and hill climbing when an algorithm retains less than 80% of the leader's improvement over a fixed initial reference.
- Apply one rule to all built-in and custom evaluation metrics using their optimization direction. Retain trained models for ensembles and stacking, and keep changed-feature searches eligible.
- Persist policy state and skip evidence, print one message per paused algorithm, and summarize decisions in the Markdown report. Older interrupted searches retain their original behavior.
- Allocate tuning allowances among eligible pending algorithms within existing stage and total budgets.
- Share the existing feature-engineering budget in 30/50/20 proportions between golden features, feature discovery, and KMeans, retaining separate stages and existing API arguments.
- Retain RC6 fixes and the fixed five-per-algorithm stacking input limit and 30 ensemble selection rounds. Add isolated local RC7 benchmarks with earlier results preserved.
- Bound initial and subsequent trials by the remaining algorithm/stage allowance, include failed-attempt wall time, and reserve time for validation. Preserve explicit per-model time limits.
- Treat expected training and inference deadline expirations as budget skips with saved diagnostics instead of candidate errors. Retain genuine failures as errors and release partial learners.
- Reuse measured stacking costs to reject unaffordable compatible trials before training. Allow cheaper RealMLP architectures to probe their cost; persist observations for resumed runs.
- Print feature-discovery fold progress, profile sizes, newly listed feature names, cache reuse, and transformation report paths. Respect AutoML verbosity and avoid repeating shared features across folds.

Local validation found a dependency/environment limitation: GPU-enabled XGBoost
3.2.0 and 3.4.1 raised CUDA-driver errors during CPU multiclass evaluation on the
validation machine. All 14 affected cases in each environment passed with the
matching CPU-only XGBoost wheel and unchanged MLJAR code. The original failures
are retained in the validation logs; this candidate does not include a driver
workaround or an XGBoost dependency pin.

## Earlier local RC6 prototype (not published)

Local candidate for validation and benchmarking; not yet published to PyPI.

- Build categorical encodings in batches to avoid fragmented DataFrames and repeated performance warnings.
- Fit and save KMeans missing-value imputation on training folds, retaining compatibility with existing artifacts.
- Record resource admission rejections as skipped candidates with diagnostics; preserve genuine failures and release partially trained learners.
- Use at most five stacking inputs per algorithm and 30 greedy ensemble selection rounds. These are fixed internal limits; repeated ensemble selections adjust weights.
- Count shared prediction dependencies once when enforcing prediction-time limits.
- Support isolated local-wheel benchmarks with artifact hashes, prediction dependency counts, resource skips, memory diagnostics, and retained training logs.

## mljar-supervised 1.4.0rc5

Fifth release candidate for 1.4.0, improving saved-model inference and avoiding
redundant SDM tuning candidates.

```bash
python -m pip install "mljar-supervised[feature-discovery]==1.4.0rc5"
```

- Reuse one frozen Kumo/TabICLv2 checkpoint across compatible saved folds within a prediction request. Keep fold-specific contexts isolated, check each fold's resource requirements, and evict retained weights when memory is needed.
- Reuse CPU predictions for equivalent saved SDM folds using their fitted preprocessing, ordered context and targets, classes, checkpoint, execution settings, and transformed inputs. Bound this cache to 32 MiB and 1% of available RAM; preserve all stacking columns and ensemble weights.
- Check free GPU memory before invoking garbage collection and CUDA cache cleanup for SDM. Clean up and remeasure only when the initial check fails; keep existing memory limits. RealMLP and TabM retain their cleanup behavior.
- Skip hill-climbing changes to the SDM context-size cap when both caps already cover every training fold. Continue exploring changes to precision, seeds, estimator counts and model size.
- Record expected sample-weight incompatibility skips in reports without repeating warnings in the console.
- Add saved-fold reuse, memory-pressure, reload-parity and real-checkpoint tests, plus a profiler for existing saved models.
- Update benchmark setup and preflight for RC5, retaining RC1–RC4 environments and results and comparing against the saved 1.3.2 baseline.

Inference improvements apply to existing saved models without retraining or new
arguments. The tuning change affects new training runs. Native categorical
changes for LightGBM and XGBoost remain deferred.

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
