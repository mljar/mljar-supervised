"""Lazy pytabkit TabM-mini adapter with bounded training and checkpoint history.

Adapted from pytabkit 1.7's TabM implementation (Apache-2.0), by David
Holzmüller and contributors: https://github.com/dholzmueller/pytabkit.
See LICENSE-pytabkit.txt for the upstream license.
The architecture/embeddings remain provided by pytabkit; MLJAR owns the loop.
"""

import copy
import time

import numpy as np
import torch
from pytabkit import TabM_D_Classifier, TabM_D_Regressor
from pytabkit.models.alg_interfaces.sub_split_interfaces import (
    SingleSplitWrapperAlgInterface,
)
from pytabkit.models.alg_interfaces.tabm_interface import (
    TabMSubSplitInterface,
    get_tabm_auto_batch_size,
)
from pytabkit.models.nn_models import rtdl_num_embeddings
from pytabkit.models.nn_models.models import PreprocessingFactory
from pytabkit.models.nn_models.tabm import Model, make_parameter_groups
from pytabkit.models.training.metrics import Metrics

from supervised.algorithms._pytabkit_compat import (
    PytabkitCompatibilityMixin,
    single_bin_diagnostics,
)


class TimedTabMInterface(TabMSubSplitInterface):
    @single_bin_diagnostics()
    def fit(self, ds, idxs_list, interface_resources, logger, tmp_folders, name):
        split = idxs_list[0]
        if split.val_idxs is None:
            raise ValueError("TabM requires an explicit validation fold")
        cfg = self.config
        deadline = cfg.get("deadline")

        def expired():
            return deadline is not None and time.monotonic() >= deadline

        if expired():
            raise ValueError("TabM time budget exhausted before training")
        torch.manual_seed(split.sub_split_seeds[0])
        self.n_classes_ = ds.get_n_classes()
        self.task_type_ = "regression" if self.n_classes_ == 0 else "classification"
        self.device_ = torch.device(
            interface_resources.gpu_devices[0]
            if interface_resources.gpu_devices
            else "cpu"
        )
        train = ds.get_sub_dataset(split.train_idxs[0])
        valid = ds.get_sub_dataset(split.val_idxs[0])
        factory = PreprocessingFactory(tfms=["quantile_tabr"])
        self.tfm_, train = factory.create(ds.tensor_infos).fit_transform(train)
        valid = self.tfm_(valid)
        x = train.tensors["x_cont"]
        self.num_col_mask_ = ~torch.all(x == x[0:1], dim=0)
        data = {}
        for key, part in [("train", train), ("val", valid)]:
            part.tensors["x_cont"] = part.tensors["x_cont"][:, self.num_col_mask_]
            data[key] = part.to(self.device_).tensors
        n_train = train.n_samples
        y_train = data["train"]["y"].clone()
        if self.n_classes_ == 0:
            self.y_mean_ = y_train.mean().item()
            self.y_std_ = y_train.std(correction=0).item()
            y_train = (y_train - self.y_mean_) / (self.y_std_ + 1e-30)
        n_numeric = data["train"]["x_cont"].shape[1]
        bins = None
        if n_numeric and n_train > 2:
            bins = rtdl_num_embeddings.compute_bins(
                data["train"]["x_cont"], n_bins=min(cfg["num_emb_n_bins"], n_train - 1)
            )
        self.single_bin_feature_count = sum(len(edges) == 2 for edges in (bins or []))
        self.model_ = Model(
            n_num_features=n_numeric,
            cat_cardinalities=ds.tensor_infos["x_cat"].get_cat_sizes().numpy().tolist(),
            n_classes=max(1, self.n_classes_),
            backbone={
                "type": "MLP",
                "n_blocks": cfg["n_blocks"],
                "d_block": cfg["d_block"],
                "dropout": cfg["dropout"],
            },
            bins=bins,
            num_embeddings=(
                None
                if bins is None
                else {
                    "type": "PiecewiseLinearEmbeddings",
                    "d_embedding": cfg["d_embedding"],
                    "activation": False,
                    "version": "B",
                }
            ),
            arch_type="tabm-mini",
            k=cfg["tabm_k"],
            share_training_batches=False,
        ).to(self.device_)
        optimizer = torch.optim.AdamW(
            make_parameter_groups(self.model_),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        batch_size = cfg["batch_size"]
        if batch_size == "auto":
            batch_size = get_tabm_auto_batch_size(n_train)
        metric = cfg["val_metric_name"]
        self.history = []
        self.best_epoch = None
        self.stop_reason = "max_epochs"
        best_score, best_state, bad_epochs = float("inf"), None, 0

        def forward(part, index):
            values = data[part]
            return self.model_(
                values["x_cont"][index],
                values["x_cat"][index] if values["x_cat"].shape[1] else None,
            ).float()

        for epoch in range(1, cfg["n_epochs"] + 1):
            if expired():
                self.stop_reason = "time_limit"
                break
            # Independent permutations per ensemble member, matching upstream.
            order = torch.rand((cfg["tabm_k"], n_train), device=self.device_).argsort(
                dim=1
            )
            self.model_.train()
            for indices in order.split(batch_size, dim=1):
                if expired():
                    self.stop_reason = "time_limit"
                    break
                indices = indices.T.flatten()
                optimizer.zero_grad(set_to_none=True)
                output = forward("train", indices).flatten(0, 1)
                target = y_train[indices]
                loss = (
                    torch.nn.functional.mse_loss(output, target)
                    if self.n_classes_ == 0
                    else torch.nn.functional.cross_entropy(output, target.squeeze(-1))
                )
                if not torch.isfinite(loss):
                    raise ValueError("TabM training loss is not finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
            if self.stop_reason == "time_limit":
                break
            self.model_.eval()
            predictions = []
            with torch.no_grad():
                for indices in torch.arange(valid.n_samples, device=self.device_).split(
                    1024
                ):
                    if expired():
                        self.stop_reason = "time_limit"
                        break
                    output = forward("val", indices)
                    if self.n_classes_:
                        output = output.softmax(dim=-1)
                    else:
                        output = output * self.y_std_ + self.y_mean_
                    predictions.append(output.mean(dim=1).cpu())
            if self.stop_reason == "time_limit":
                break
            prediction = torch.cat(predictions)
            # pytabkit metrics expect logits; log(mean probabilities) preserves
            # the ensemble distribution when the metric applies softmax.
            if self.n_classes_:
                prediction = torch.log(prediction + 1e-30)
            score = float(Metrics.apply(prediction, data["val"]["y"].cpu(), metric))
            if not np.isfinite(score):
                raise ValueError("TabM validation score is not finite")
            self.history.append({"epoch": epoch, "metric": metric, "validation": score})
            if score < best_score:
                best_score, self.best_epoch, bad_epochs = score, epoch, 0
                best_state = copy.deepcopy(self.model_.state_dict())
            else:
                bad_epochs += 1
            if bad_epochs > cfg["patience"]:
                self.stop_reason = "early_stopping"
                break
        if best_state is None:
            raise ValueError(
                "TabM finished without a validation checkpoint; increase the time budget"
            )
        self.model_.load_state_dict(best_state)
        self.fit_params = [{"best_epoch": self.best_epoch}]
        return None

    def to(self, device):
        self.device_ = torch.device(device)
        self.model_.to(device)
        self.num_col_mask_ = self.num_col_mask_.to(device)
        # Preprocessing is performed on CPU before transfer in upstream predict.
        self.tfm_.to("cpu")


class TabMMixin(PytabkitCompatibilityMixin):
    def _create_alg_interface(self, n_cv):
        if n_cv != 1:
            raise ValueError("MLJAR owns TabM cross-validation")
        config = dict(self.get_config(), deadline=getattr(self, "deadline", None))
        return SingleSplitWrapperAlgInterface([TimedTabMInterface(**config)], **config)

    def to(self, device):
        for interface in self.cv_alg_interface_.sub_split_interfaces:
            interface.to(device)


class TabMClassifier(TabMMixin, TabM_D_Classifier):
    pass


class TabMRegressor(TabMMixin, TabM_D_Regressor):
    pass
