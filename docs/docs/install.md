# Installation

The **current source checkout requires Python 3.11+** for every installation
option. It includes feature discovery (`features-goldmine>=1.1,<1.2`) in the base
package. No extra tag is needed. These changes are not yet published in `1.4.0a1`.
From the repository root, install them with:

```bash
python -m pip install .
```

The `[feature-discovery]` extra remains a compatibility alias and adds nothing.
`[realmlp]`, `[tabm]`, `[tfm]`, and `[all]` still add optional model backends.
A base install does not require PyTorch, SDM, or the Mercury app runtime.
Feature discovery runs automatically in eligible Compete workflows; disable it
with `feature_discovery=False`. Installation does not enable it in every mode.

For **Kumo and TabICLv2**, use Python 3.11+, `[tfm]` (or `[kumo]`, `[tabiclv2]`,
`[all]`). The current source checkout installs `structured-data-models==0.2.1`
from PyPI through these extras. The published alpha still needs its older Git backend. See the
[Kumo/TabICLv2 instructions](features/sdm.md#installation). Weights download from
Hugging Face on first use, not during package installation.

Choose the models and hardware you want to use:

| Setup | What to install |
|---|---|
| Standard MLJAR | `mljar-supervised`; PyTorch is not required |
| RealMLP on CPU | CPU PyTorch, then MLJAR with `[realmlp]` or `[all]` |
| RealMLP on NVIDIA GPU | CUDA PyTorch, then the same MLJAR extra |
| Kumo / TabICLv2 on CPU or NVIDIA GPU | Python 3.11+, chosen PyTorch build, then `[tfm]`; see version-specific commands below |

**TabM mini uses the same CPU/GPU setup.** Substitute `[tabm]` for `[realmlp]`
in the commands below, or use `[all]` for both. These extras share
pytabkit/PyTorch dependencies, so installing any one makes both models available
in Compete mode. See [TabM mini](features/tabm.md) for usage and tuning.

There are no separate CPU/GPU editions of MLJAR. The PyTorch build controls
the available devices for RealMLP, TabM, Kumo, and TabICLv2. Other algorithms
use their own device settings.

!!! note "Alpha release"
    Version `1.4.0a1` includes RealMLP, TabM mini, Kumo and TabICLv2 support,
    plus optional [feature discovery](features/feature_discovery.md) powered by
    features-goldmine. Install `[feature-discovery]` or `[all]` on Python 3.10+
    to enable it automatically in Compete.
    Select the exact alpha version in the commands below to test
    these models. The stable `1.3.2` package does not include these integrations.

## Create an environment

Use Python 3.11 or 3.12 for the RealMLP setup covered by our CI. Create a fresh
virtual environment for your chosen CPU or GPU build.

=== "Linux / macOS"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    ```

=== "Windows PowerShell"

    ```powershell
    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    ```

    If activation is unavailable, use `.\.venv\Scripts\python.exe` in place of
    `python` in the commands below.

## Standard MLJAR

```bash
python -m pip install mljar-supervised
```

To test the alpha release:

```bash
python -m pip install "mljar-supervised==1.4.0a1"
```

The existing sklearn `Neural Network` does not require PyTorch or pytabkit.
RealMLP is optional.

## XGBoost CPU installation

Starting with `1.4.0a1`, MLJAR installs `xgboost-cpu` on Linux x86-64/ARM64 and
Windows x86-64. Other platforms retain `xgboost`. MLJAR trains XGBoost on CPU;
this choice does not disable CUDA for RealMLP, TabM, Kumo or TabICLv2.

Use a fresh environment when upgrading if possible. The `xgboost`,
`xgboost-cpu` and `xgboost-cu12` distributions share the `xgboost` import and
must not be installed together. Pip does not automatically remove the old
provider when a dependency changes its name.

On the platforms using `xgboost-cpu`, migrate an existing MLJAR environment with:

```bash
python -m pip uninstall -y xgboost xgboost-cpu xgboost-cu12
python -m pip install --upgrade "mljar-supervised==1.4.0a1"
python -m pip check
```

If another application requires the regular `xgboost` distribution, use a separate
MLJAR environment. Keep only one provider, rather than installing overlapping
packages or ignoring dependency-check failures. The benchmark launcher creates
an isolated environment automatically and checks CPU multiclass evaluation.

## Kumo and TabICLv2 on CPU or NVIDIA GPU

Use Python **3.11+**. In the current source checkout, `[tfm]`, `[kumo]`,
`[tabiclv2]`, and `[all]` install the tested PyPI backend
`structured-data-models==0.2.1` automatically. This change is not yet published;
`1.4.0a1` uses the older Git backend.

Install your PyTorch build first:

=== "CPU (Linux / Windows)"

    ```bash
    python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu
    ```

=== "NVIDIA GPU (Linux / Windows)"

    For a CUDA 12.8-compatible driver:

    ```bash
    python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
    ```

    For other drivers, choose a compatible build from the
    [PyTorch installation selector](https://pytorch.org/get-started/locally/).
    Keep PyTorch within `>=2.7,<3`.

From the root of the current MLJAR source checkout, run:

```bash
python -m pip install ".[tfm]"
```

No separate SDM install or Git dependency is needed for this checkout.

!!! note "Published alpha 1.4.0a1"
    To reproduce the published alpha in a separate environment, use its original
    backend. Installing SDM 0.2.1 alongside the published alpha is not supported.

    ```bash
    python -m pip install "mljar-supervised[tfm]==1.4.0a1"
    python -m pip install "git+https://github.com/NVIDIA/structured-data-models.git@b97cb914e2bdcf0d165fc2c83b5fdf795aad6a69"
    ```

| Extra (current source checkout) | Dependencies it adds |
|---|---|
| `[kumo]`, `[tabiclv2]`, `[tfm]` | SDM 0.2.1, PyTorch, Hugging Face Hub, psutil |
| `[all]` | Those dependencies plus pytabkit for RealMLP/TabM; feature discovery is already in base |

The three TFM extras share dependencies; installing one does not restrict which
of the two algorithms can be selected. SDM installs its own dependencies,
including PyArrow and safetensors. A standard MLJAR installation does not load
SDM or download model weights.

Verify the backend and device:

```bash
python -m pip check
python -c "from supervised.algorithms.sdm import check_dependencies; check_dependencies(); import torch; print('CUDA available:', torch.cuda.is_available())"
```

A CPU installation should print `False`. Select `algorithms=["Kumo", "TabICLv2"]`
explicitly for CPU runs. Automatic Compete selection includes both when the tested
backend and CUDA are available. Weights download from Hugging Face on first fit.
See [Kumo and TabICLv2](features/sdm.md) for model sizes, tuning, memory limits,
and offline use.

## RealMLP on CPU

Install PyTorch first. The examples pin version 2.11.0 for reproducibility;
MLJAR's supported dependency range is `torch>=2.7,<3`.

=== "Linux / Windows"

    ```bash
    python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu
    ```

=== "macOS"

    ```bash
    python -m pip install "torch==2.11.0"
    ```

    Use a Mac/Python combination with a compatible PyTorch wheel. This RealMLP
    integration uses CPU on macOS; Apple MPS acceleration is not enabled.

This selects CPU-only PyTorch. Other MLJAR dependencies, such as XGBoost on
Linux, may still install NVIDIA libraries.

Then follow [Install the RealMLP extra](#install-the-realmlp-extra).

## RealMLP on NVIDIA GPU

For Linux or Windows with an NVIDIA GPU and a CUDA 12.8-compatible driver:

```bash
nvidia-smi
python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
```

If this build does not match your hardware or driver, choose a compatible build
from the [PyTorch installation selector](https://pytorch.org/get-started/locally/)
or [version-specific commands](https://pytorch.org/get-started/previous-versions/).
Keep the installed torch version within `>=2.7,<3`. RealMLP needs `torch`; it does
not require `torchvision` or `torchaudio`.

Then install the same RealMLP extra below. RealMLP currently supports a single
CUDA device, not multi-GPU training.

## Install the RealMLP extra

After installing your chosen PyTorch build, install the release candidate:

```bash
python -m pip install "mljar-supervised[realmlp]==1.4.0a1"
# Or all optional model dependencies:
python -m pip install "mljar-supervised[all]==1.4.0a1"
```

`[realmlp]` and `[tabm]` share pytabkit/PyTorch dependencies. `[all]` also adds
Hugging Face support for Kumo and TabICLv2. For the published alpha shown above,
install its Git SDM backend separately as described above. The current source
checkout includes SDM 0.2.1 from PyPI in `[all]`.

To install from source instead:

```bash
git clone https://github.com/mljar/mljar-supervised.git
cd mljar-supervised
python -m pip install ".[realmlp]"
```

From an existing checkout, run the final command from its root directory.
Use `python -m pip install ".[all]"` for all optional model dependencies.

Installing a compatible PyTorch build first lets pip reuse that build. The extra
adds pytabkit and the other RealMLP dependencies.

## Verify the installation

See [native runtime compatibility](#native-runtime-compatibility) for the macOS
check in the current source checkout. Installing the optional dependencies does
not by itself establish that the native libraries can run together.

```bash
python -m pip check
python -c "from importlib.metadata import version; import torch; print('MLJAR:', version('mljar-supervised')); print('pytabkit:', version('pytabkit')); print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

For CPU-only PyTorch, expect `CUDA build: None` and `CUDA available: False`.
For the NVIDIA setup, expect `CUDA available: True`. If it is false, check that
this environment has the CUDA build installed and that `nvidia-smi` sees the GPU.

## Try RealMLP

RealMLP is available only in Compete mode. `AutoML(mode="Compete")` includes it
automatically when its optional dependencies are installed. Select it explicitly
to train only RealMLP:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from supervised import AutoML

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
automl = AutoML(
    mode="Compete",
    algorithms=["RealMLP"],
    total_time_limit=120,
    custom_params={"realmlp": {"device": "auto"}},
)
automl.fit(X_train, y_train)
print(automl.predict(X_test[:5]))
```

The default `device="auto"` selects CUDA when available, otherwise CPU. To force
CPU training, use `custom_params={"realmlp": {"device": "cpu"}}`. Setting the
training device does not change installed packages or remove CUDA downloads.
GPU memory failures do not trigger an automatic retry on CPU.

For a short check of binary classification, multiclass classification and
regression, run this script from the source checkout. Use a fresh output directory:

```bash
python examples/scripts/realmlp_synthetic.py --output AutoML_realmlp_check
```

The script uses CPU and eight epochs per task. See the [RealMLP guide](features/realmlp.md)
for supported data, tuning parameters, memory limits and explanations.

## Native runtime compatibility

The current source checkout checks for the reproduced macOS conflict between
PyTorch and the native libraries used by standard ML models. This protection is
not included in the published `1.4.0a1` wheel.

When Torch models are needed, MLJAR runs tiny synthetic workloads in a temporary
Python process before allowing them in the main process. All cases share one
**three-second deadline**, including interpreter startup. A hung child is killed
and reaped; operating-system scheduling and termination can add a small delay.
A slow or cold start can time out even without a native conflict.

- Standard models with Torch unused and unloaded run without a probe. A base
  installation still requires no Torch.
- Automatic selection skips optional Torch models if the check fails or cannot
  finish, provided Torch has not been loaded in the main process. Training
  continues with the remaining models.
- Explicit requests raise `NativeCompatibilityError` instead of silently changing
  the requested algorithms. Saved models, ensembles, and resumed runs are checked
  too; their members are never replaced to bypass a failure.
- If Torch is already loaded, MLJAR also checks standard-model operations. An
  unverified mixed environment stops with the same error.
- Linux and Windows keep their existing execution paths; this macOS check does
  not launch children on those platforms.

The child receives no training data and downloads no weights. Only one probe
child runs at a time. It needs at least 1 GiB of available host RAM to start and
is stopped if its sampled RSS exceeds 768 MiB. Sampling is not a hard OS memory
limit. Training itself stays in the main process; no training worker is retained.

Results are cached in memory for the inspected environment and loaded backend
state. Low-memory and other inconclusive results can be checked again on a later
request. Reports record skipped algorithms and reasons, including in quiet mode.
The run's `params.json` preserves requested and effective algorithms, while
`native_compatibility.log` holds bounded child diagnostics when available.

This detects the tested conflict; it does not repair incompatible binaries or
guarantee safety for arbitrary third-party imports and workloads. A timeout means
“unverified,” not “known broken.” There is no import-order requirement for users
and no unverified version pin to apply. To keep working after a refusal, use the
standard models in a fresh process where Torch has not been imported. Running
the blocked Torch models requires a compatible environment.
