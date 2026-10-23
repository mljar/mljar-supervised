"""Lazy SDM/PyTorch adapter. Frozen weights are loaded from pinned local files."""

from contextlib import contextmanager
import threading
import weakref

import numpy as np
import pandas as pd
import torch
import sdm
from sdm._kernels.rmsnorm_cast import _triton_rmsnorm_cast
from sdm.models.callback import Callback


_network_locks = weakref.WeakKeyDictionary()
_network_locks_guard = threading.Lock()


def _network_lock(model):
    # Early-stopping copies share frozen weights, but a fitted SDM context is
    # mutable. Serialize requests sharing that network, without retaining it.
    with _network_locks_guard:
        if model not in _network_locks:
            _network_locks[model] = threading.RLock()
        return _network_locks[model]


def _cache_fits_memory(learner):
    from supervised.algorithms.realmlp_resources import available_memory
    from supervised.algorithms.sdm import working_memory_bytes

    rows = len(learner.context)
    cols = learner.training_info.get("effective_columns", learner.context.shape[1])
    # SDM offloads ensemble caches to host memory. Allow a conservative working
    # allocation per member, in addition to model and inference workspaces.
    working = working_memory_bytes(
        learner.params, learner.algorithm_short_name, rows, cols, learner.device
    )
    weights = sum(p.numel() * p.element_size() for p in learner.model.parameters())
    estimate = (
        512 * 1024**2
        + 3 * weights
        + working * (1 + learner.params.get("num_estimators", 8))
    )
    budget = available_memory() * learner.params.get("memory_fraction", 0.6)
    if learner.params.get("memory_limit_gb") is not None:
        budget = min(budget, learner.params["memory_limit_gb"] * 1024**3)
    return estimate <= budget


def _pad_query(frame, batch_size):
    # In FP16, changing GEMM shapes changes rounding even for identical rows.
    # Queries attend only to context, so duplicate padding rows do not affect
    # real queries. Keep the configured shape even for a one-row request.
    if 0 < len(frame) < batch_size:
        return pd.concat(
            [frame, frame.iloc[[-1] * (batch_size - len(frame))]], ignore_index=True
        )
    return frame


def _prediction_values(learner, out):
    if learner.ml_task == "regression":
        return out.numerical.float().mean(dim=-1).cpu().numpy()
    columns = out.columns[sdm.Stype.numerical]
    indices = [columns.index(str(i)) for i in range(len(learner.classes_))]
    probabilities = out.numerical[..., indices].float().cpu().numpy()
    probabilities = np.clip(probabilities, 0, 1)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


@contextmanager
def _autocast(learner):
    # Precision is part of each candidate's saved configuration. CPU stays in
    # FP32; explicit FP32 CUDA trials also disable TF32 for this call.
    full_precision = learner.effective_precision == "fp32"
    disable_tf32 = full_precision and learner.device.startswith("cuda")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        if disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
        with torch.amp.autocast(
            torch.device(learner.device).type,
            dtype=torch.float16,
            enabled=learner.device.startswith("cuda") and not full_precision,
        ):
            yield
    finally:
        if disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32


@contextmanager
def prediction_session(learner):
    """Reuse a fold's context across query batches, clearing it on every exit."""
    model = learner.model
    with _network_lock(model), torch.inference_mode(), _autocast(learner):
        try:
            learner._check_time()
            use_cache = _cache_fits_memory(learner)
            types = {
                c: "categorical" if isinstance(t, pd.CategoricalDtype) else "numerical"
                for c, t in learner.context.dtypes.items()
            }
            if use_cache:
                context = sdm.TableTensor.from_pandas(
                    learner.context, stypes=types, device=learner.device
                )
                regression = learner.ml_task == "regression"
                target = sdm.TableTensor.from_pandas(
                    pd.DataFrame({"target": learner.context_y}),
                    stypes={"target": "numerical" if regression else "categorical"},
                    device=learner.device,
                )
                # SDM.fit does not call on_model_forward_end. A temporary hook
                # keeps deadline checks between context-encoding estimators.
                network = model.models["regression" if regression else "classification"]
                with network.register_forward_hook(lambda *_: learner._check_time()):
                    model.fit(
                        context,
                        target,
                        num_estimators=learner.params.get("num_estimators", 8),
                        estimator_batch_size=1,
                        generator=torch.Generator(learner.device).manual_seed(
                            learner.params.get("seed", 1234)
                        ),
                        callbacks=[DeadlineCallback(learner._check_time)],
                    )
                learner._check_time()
                del context, target

            def predict(frame):
                count = len(frame)
                frame = _pad_query(frame, learner.params.get("predict_batch_size", 128))
                if not use_cache:
                    return predict_batch(learner, frame)[:count]
                query = sdm.TableTensor.from_pandas(
                    frame, stypes=types, device=learner.device
                )
                out = model.predict(
                    query, callbacks=[DeadlineCallback(learner._check_time)]
                )
                return _prediction_values(learner, out)[:count]

            yield predict
        finally:
            model.clear()


class _KumoRMSNorm(sdm.nn.RMSNorm):
    """Preserve PyTorch's FP32 fallback for mixed-precision SDM norms."""

    def forward(self, x, rope=None):
        if (
            rope is None
            and x.dtype in (torch.float16, torch.bfloat16)
            and self.weight is not None
            and self.weight.dtype == torch.float32
            and x.shape[-1:] == self.normalized_shape
        ):
            # Preserve the pinned SDM fast path whenever it is eligible.
            # Other shapes, empty tensors, disabled autocast/grad mode or a
            # missing Triton kernel reach PyTorch's mixed-dtype fallback.
            sdm_autocast = (
                x.is_cuda
                and torch.is_autocast_enabled("cuda")
                and not torch.is_grad_enabled()
            )
            if (
                sdm_autocast
                and _triton_rmsnorm_cast is not None
                and x.size(-1) in (32, 64, 128, 256, 512)
                and x.numel() > 0
                and self.weight.device == x.device
            ):
                return super().forward(x)
            # PyTorch warns and then performs exactly these FP32 operations.
            # Keep the weights, epsilon and final cast unchanged. Casting the
            # weights down would discard precision from the checkpoint.
            value = x.float()
            eps = self.eps if self.eps is not None else torch.finfo(value.dtype).eps
            scale = torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + eps)
            dtype = x.dtype
            if sdm_autocast:
                dtype = torch.get_autocast_dtype("cuda")
            return ((value * scale) * self.weight).to(dtype)
        return super().forward(x, rope=rope)


def _prepare_kumo_norms(model):
    # Change only the affected instances, leaving SDM's fast kernels and other
    # users of torch/SDM untouched. The checkpoint/state_dict keys stay stable.
    for name, module in list(model.named_modules()):
        if type(module) is sdm.nn.RMSNorm and module.weight is not None:
            replacement = _KumoRMSNorm(
                module.normalized_shape,
                eps=module.eps,
                elementwise_affine=module.elementwise_affine,
                device="meta",
            )
            replacement.weight = module.weight
            replacement.train(module.training)
            model.set_submodule(name, replacement)


class DeadlineCallback(Callback):
    def __init__(self, check_time):
        self.check_time = check_time

    def on_model_forward_end(self, model, out):
        self.check_time()
        return out


def load_network(algorithm, ml_task, checkpoint_path, device, size="small"):
    task = "regression" if ml_task == "regression" else "classification"
    cls = sdm.models.KumoTabular if algorithm == "Kumo" else sdm.models.TabICLv2
    kwargs = {"size": size} if algorithm == "Kumo" else {}
    model = cls(task=task, pretrained=False, device="meta", **kwargs)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if algorithm == "TabICLv2":
        from sdm.models.tabiclv2.ckpt import remap_ckpt

        state = remap_ckpt(state["state_dict"], is_classifier=task == "classification")
    model.models[task].load_state_dict(state, assign=True)
    if algorithm == "Kumo":
        _prepare_kumo_norms(model)
    model.eval()
    model.requires_grad_(False)
    return model


def predict_batch(learner, frame):
    # Public forward API: bounded uncached inference, sequential ensemble members.
    # Resetting the generator reproduces the same recipe for each query batch.
    types = {
        c: "categorical" if isinstance(t, pd.CategoricalDtype) else "numerical"
        for c, t in learner.context.dtypes.items()
    }
    context = sdm.TableTensor.from_pandas(
        learner.context, stypes=types, device=learner.device
    )
    query = sdm.TableTensor.from_pandas(frame, stypes=types, device=learner.device)
    regression = learner.ml_task == "regression"
    target = pd.DataFrame({"target": learner.context_y})
    y = sdm.TableTensor.from_pandas(
        target,
        stypes={"target": "numerical" if regression else "categorical"},
        device=learner.device,
    )
    with torch.inference_mode(), _autocast(learner):
        out = learner.model(
            x_context=context,
            y_context=y,
            x_query=query,
            num_estimators=learner.params.get("num_estimators", 8),
            estimator_batch_size=1,
            generator=torch.Generator(learner.device).manual_seed(
                learner.params.get("seed", 1234)
            ),
            callbacks=[DeadlineCallback(learner._check_time)],
        )
    return _prediction_values(learner, out)
