"""Lazy SDM/PyTorch adapter. Frozen weights are loaded from pinned local files."""

import numpy as np
import pandas as pd
import torch
import sdm
from sdm.models.callback import Callback


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
