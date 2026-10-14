"""Lazy SDM/PyTorch adapter. Frozen weights are loaded from pinned local files."""

import numpy as np
import pandas as pd
import torch
import sdm
from sdm._kernels.rmsnorm_cast import _triton_rmsnorm_cast
from sdm.models.callback import Callback


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
    with torch.inference_mode(), torch.amp.autocast(
        torch.device(learner.device).type,
        dtype=torch.float16,
        enabled=learner.device.startswith("cuda"),
    ):
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
    if regression:
        return out.numerical.float().mean(dim=-1).cpu().numpy()
    columns = out.columns[sdm.Stype.numerical]
    indices = [columns.index(str(i)) for i in range(len(learner.classes_))]
    probabilities = out.numerical[..., indices].float().cpu().numpy()
    probabilities = np.clip(probabilities, 0, 1)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)
