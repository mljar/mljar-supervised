# Compare published MLJAR releases on TabArena

From the repository root, run one command in each terminal:

```bash
./benchmarks/run.sh 1.3.2
./benchmarks/run.sh 1.4.0rc2
```

No environment activation is necessary. The launcher uses Python 3 and `uv`
to install separate Python 3.11 environments under
`performance-results/tabarena/comparison-environments/`. It verifies both the
package version and its import location, so the working tree cannot shadow the
published packages. The common numerical/model dependencies are pinned in
`constraints.txt`; complete resolved environments are recorded with each run.
Environment directory names include the dependency-profile hash, so changing
pins preserves older environments. XGBoost 2.1.4 is shared by both versions:
the multiclass CPU smoke exposed unwanted CUDA initialization in 3.2.0.
Linux, `taskset`, and at least two physical cores are required.

Defaults: 12 datasets, official repeat 0 / fold 0, Compete, 3600 seconds per
dataset per version, 15 CPU threads, seed 1234, explanations disabled. Both
versions train on the exact same exported data and split indices; their hashes
are checked before fitting. MLJAR controls internal validation and ensembling.
The dataset header describes sizes, feature types, missing values and the
training target. MLJAR stdout/stderr stream live and are also saved to
`console.log` in every attempt directory.

The CPU sets use separate physical cores, keeping SMT siblings together and
reserving two physical cores on larger machines. 15 threads per worker fits the
current 18-core / 36-thread workstation. Lower `--n-jobs` on smaller machines.
Version locks prevent duplicate runs; release candidates also share a CPU-slot lock.
RC1 and RC2 use the same candidate CPU partition and must run sequentially.

RC2 installs CUDA 12.8 PyTorch, pytabkit and the pinned NVIDIA SDM revision.
The GPU is checked and all Kumo sizes / TabICLv2 classification and regression
weights are cached before training. Only RC2 uses the GPU. A GPU lock prevents
another instance of this harness using it concurrently; it cannot reserve the
GPU against unrelated applications. To explicitly use CPU for all new models,
pass `--device cpu` with a separate `--run` directory. Missing CUDA fails early
instead of silently dropping optional algorithms. Some candidates may exceed
available VRAM; their original errors and partial artifacts remain saved.

The first setup needs internet access and several GB of disk space. Prepare
without starting training:

```bash
./benchmarks/run.sh 1.3.2 --prepare-only
./benchmarks/run.sh 1.4.0rc2 --prepare-only
```

The TabArena data environment/checkout at `performance-results/tabarena/venv`
and `source` is reused (or bootstrapped), pinned to commit
`10b1ecb9fa628b8f69db6763f8aeaafdd5d5ef77`. The training environments do not
import TabArena or the development MLJAR source. The data exporter uses
TabArena's `OpenMLTaskWrapper.get_train_test_split(fold=0, repeat=0, sample=0)`.

## Short checks and individual datasets

```bash
./benchmarks/run.sh 1.3.2 --suite smoke --seconds 60
./benchmarks/run.sh 1.4.0rc2 --suite smoke --seconds 60

./benchmarks/run.sh 1.4.0rc2 --datasets credit-g --seconds 120 \
    --run performance-results/tabarena/rc2-credit-check
```

The smoke suite covers binary, multiclass and regression. A short budget may
not reach every optional algorithm; `--algorithms RealMLP TabM Kumo TabICLv2`
can be used for a separate diagnostic run. Such overrides are recorded and
must not be mixed with automatic-selection benchmark results.

## Outputs, timing and resume

The default shared result root is
`performance-results/tabarena/comparison-1.3.2-vs-1.4.0rc2-small-3600s-15threads/`.
RC1 environments and outputs remain in their original locations. The default
for a new 1.3.2 run now pairs it with RC2. An already-running 1.3.2 job continues
in its original output directory.

It contains frozen harness code, a manifest, and separate version/dataset/split
directories. Each attempt retains the complete `automl/` directory, its reports,
all trained models and checkpoints, console output, sampled process-tree RSS,
dataset description, requested/resolved parameters, model inventory, algorithm
coverage, and held-out predictions for each model that can be evaluated.
`classes.json` maps encoded class indices to original labels; prediction rows
follow the original test split order saved in `test_row_indices.json`.

The final model is selected using internal validation only. Outer test data is
first loaded after fitting. Test scores are ROC AUC, multiclass log loss, and
RMSE; `metric_error` uses 1 - AUC for binary tasks so lower always means better.
Internal validation values are printed directly from MLJAR's leaderboard CSV,
including random-feature candidates. Every model's parameters, stacking flag,
ensemble membership and available training/device metadata remain inspectable.
Prediction failures are recorded separately. The selected model is reloaded
and its predictions checked before a case is marked successful.

The supervisor allows 900 seconds of training grace and a separate 1800-second
evaluation budget by default. Actual fitting, inference and wall time are
recorded separately. A sampled process-tree RSS ceiling defaults to 96 GiB;
shared pages can be counted more than once, so this is not a cgroup allocation.
With two concurrent versions, the 24 one-hour fits take roughly 12 hours plus
evaluation and overruns if the full budgets are consumed.

Ctrl+C terminates the active worker process group. Rerunning the same command
skips completed cases, including completed failures; interrupted cases start a
new attempt without overwriting earlier artifacts. Use a fresh `--run` directory
to retry completed failures or change settings. Settings, environment, or harness
changes cannot silently reuse old results. Keep the complete result directory,
shared data directory, and environment freezes for future inspection/reloading.

`comparison/report.md`, `datasets.csv`, `models.csv`, and `paired.csv` update
after each dataset. Regenerate without training:

```bash
python3 benchmarks/tabarena/comparison/report.py --run PATH_TO_RUN
```

### Reuse the existing 1.3.2 baseline

Run RC2 normally while the existing 1.3.2 process continues. Afterward, compare
their results without copying models or repeating baseline training:

```bash
python3 benchmarks/tabarena/comparison/report.py \
    --run performance-results/tabarena/comparison-1.3.2-vs-1.4.0rc2-small-3600s-15threads \
    --baseline-run performance-results/tabarena/comparison-1.3.2-vs-1.4.0rc1-small-3600s-15threads
```

This reads the old baseline without changing its files and writes the comparison
under the RC2 run. Only successful cases with matching data hashes, metrics,
splits, seeds and budgets are paired. The manifests must also agree on the
TabArena revision, protocol, algorithm selection and resource limits. Source
paths are listed in the report. If either job is still running, repeat the report
command after it finishes; the automatic per-dataset report uses its own run's
results.

### Older runs

The launcher still accepts `1.4.0rc1`. For a **new** RC1 comparison, also pass
`--candidate-version 1.4.0rc1` to the 1.3.2 command and use a fresh `--run` path
for both versions. Existing runs retain their frozen code and configuration;
the updated harness deliberately rejects resuming them with changed code.
To resume the original interrupted run with its original runner:

```bash
benchmark_run="$PWD/performance-results/tabarena/comparison-1.3.2-vs-1.4.0rc1-small-3600s-15threads"
MLJAR_BENCHMARK_ROOT="$PWD" python3 -u "$benchmark_run/code/run.py" \
    --version 1.3.2 --suite small --seconds 3600 --n-jobs 15 --run "$benchmark_run"
```

Use the original settings and requested version; do not start a second process
while that version is still running.

This is a 12-dataset subset of TabArena-Lite. Compare external methods only on
the same dataset/task, r0f0 split and metric. Disclose hardware, GPU and time
budget differences; these runs are not the full public leaderboard protocol.
MLJAR is evaluated as an AutoML system, which owns its internal validation,
as described in [TabArena's protocol](https://github.com/autogluon/tabarena#contributing-a-model-or-system).

Tests:

```bash
performance-results/tabarena/comparison-environments/1.3.2-*/bin/python -m pytest \
    benchmarks/tabarena/comparison/test_comparison.py -q
```
