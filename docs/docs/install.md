# Installation

Choose the models and hardware you want to use:

| Setup | What to install |
|---|---|
| Standard MLJAR | `mljar-supervised`; PyTorch is not required |
| RealMLP on CPU | CPU PyTorch, then MLJAR with `[realmlp]` or `[all]` |
| RealMLP on NVIDIA GPU | CUDA PyTorch, then the same MLJAR extra |

**TabM mini uses the same CPU/GPU setup.** Substitute `[tabm]` for `[realmlp]`
in the commands below, or use `[all]` for both. The three extras currently share
pytabkit/PyTorch dependencies, so installing any one makes both models available
in Compete mode. See [TabM mini](features/tabm.md) for usage and tuning.

There are no separate CPU/GPU editions of MLJAR. The PyTorch build controls
RealMLP's available devices. These instructions do not change GPU support for
other algorithms.

!!! note "RealMLP release status"
    RealMLP and TabM mini are currently available from source. The published PyPI 1.3.2 package
    does not include the extra. Use the source commands below until a release
    with RealMLP is published. Installing `mljar-supervised[realmlp]` from an older
    release will not add RealMLP.

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

The existing sklearn `Neural Network` does not require PyTorch or pytabkit.
RealMLP is optional.

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

After installing your chosen PyTorch build, install from source:

```bash
git clone https://github.com/mljar/mljar-supervised.git
cd mljar-supervised
python -m pip install ".[realmlp]"
```

If you already have this checkout, run the final command from its root directory.
For all optional model dependencies, use `python -m pip install ".[all]"` instead.
Currently `[all]`, `[realmlp]` and `[tabm]` install the same optional dependencies.

After a PyPI release includes RealMLP, the equivalent commands will be:

```bash
python -m pip install "mljar-supervised[realmlp]"
# Or all optional model dependencies:
python -m pip install "mljar-supervised[all]"
```

Installing a compatible PyTorch build first lets pip reuse that build. The extra
adds pytabkit and the other RealMLP dependencies.

## Verify the installation

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
