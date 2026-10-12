# Installation

```bash
python -m pip install mljar-supervised
```

The default installation includes the existing models. RealMLP and PyTorch are
optional; the existing sklearn `Neural Network` does not need them.

```bash
python -m pip install "mljar-supervised[realmlp]"
# Or all supported optional model dependencies (currently the same):
python -m pip install "mljar-supervised[all]"
```

These extras are available in source checkouts containing RealMLP; until a release
with this feature is published, install from that checkout:

```bash
python -m pip install ".[realmlp]"
```

RealMLP is available only in Compete mode. After installing the extra,
`AutoML(mode="Compete")` includes it automatically. To train only RealMLP, use
`AutoML(mode="Compete", algorithms=["RealMLP"])`. Explain and Perform stay unchanged.

## CPU and NVIDIA GPU

Use a fresh virtual environment. Choose the PyTorch build before installing the
extra. The installed version must satisfy MLJAR's and pytabkit's requirements.

For CPU-only PyTorch on supported platforms:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "mljar-supervised[realmlp]"
```

For NVIDIA GPUs, use the command supplied by the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/) for
your operating system and driver-compatible CUDA build, then install the same
MLJAR extra. Do not install torchvision or torchaudio solely for RealMLP.

Verify the result:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m pip check
```

RealMLP supports CPU and a single CUDA device. Explicit `device="cpu"` uses the
CPU even when a GPU is available. `device="auto"` selects CUDA when available,
otherwise CPU; it does not retry GPU memory failures on CPU. Apple MPS is not
enabled in this integration. Optional backend support depends on available
PyTorch wheels; the initial CI target is Python 3.11 and 3.12.

See [RealMLP](features/realmlp.md) for memory budgets and examples.
