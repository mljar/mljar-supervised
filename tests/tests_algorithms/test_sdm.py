"""Unit checks run without SDM; real checkpoint tests require MLJAR_TEST_SDM=1."""

import importlib.util
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split

from supervised.algorithms.sdm import (
    KumoClassificationAlgorithm,
    KumoRegressionAlgorithm,
    TabICLv2ClassificationAlgorithm,
    TabICLv2RegressionAlgorithm,
    validate_options,
)

CLASSES = [KumoClassificationAlgorithm, TabICLv2ClassificationAlgorithm]
real_models = pytest.mark.skipif(
    os.environ.get("MLJAR_TEST_SDM") != "1",
    reason="opt-in pretrained checkpoint integration",
)

sdm_installed = pytest.mark.skipif(
    importlib.util.find_spec("sdm") is None, reason="optional SDM backend"
)


@pytest.fixture
def torch_warn_always():
    import torch

    # The reference call must not consume TORCH_WARN_ONCE and accidentally
    # prevent the adapted call from failing when warnings are errors.
    previous = torch.is_warn_always_enabled()
    torch.set_warn_always(True)
    try:
        yield
    finally:
        torch.set_warn_always(previous)


@sdm_installed
@pytest.mark.parametrize("width", [128, 512, 1024])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("eps", [None, 1e-6])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "zeros", "empty"])
def test_kumo_norm_matches_torch_without_dtype_warning(
    width, dtype, eps, layout, torch_warn_always
):
    import torch
    from sdm.nn import RMSNorm

    from supervised.algorithms._sdm_backend import _prepare_kumo_norms

    generator = torch.Generator().manual_seed(42)
    norm = RMSNorm(width, eps=eps).eval().requires_grad_(False)
    norm.weight.copy_(torch.randn(width, generator=generator))
    x = torch.randn(4, 2 * width, generator=generator).to(getattr(torch, dtype))
    x = x[:, ::2] if layout == "strided" else x[:, :width].contiguous()
    if layout == "zeros":
        x.zero_()
    elif layout == "empty":
        x = x[:0]
    with torch.inference_mode(), warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        expected = norm(x)
    model = torch.nn.Sequential(norm)
    _prepare_kumo_norms(model)
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = model(x)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model[0].weight is norm.weight


@sdm_installed
@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_kumo_norm_adaptation_preserves_checkpoint_and_fast_modules(size):
    import sdm

    from supervised.algorithms._sdm_backend import _KumoRMSNorm, _prepare_kumo_norms

    model = (
        sdm.models.KumoTabular(
            task="classification", size=size, pretrained=False, device="meta"
        )
        .eval()
        .requires_grad_(False)
    )
    weights = dict(model.named_parameters())
    keys = set(model.state_dict())
    norms = {
        name: module
        for name, module in model.named_modules()
        if type(module) is sdm.nn.RMSNorm
    }
    _prepare_kumo_norms(model)
    changed = 0
    for name, original in norms.items():
        current = model.get_submodule(name)
        if original.weight is not None:
            assert isinstance(current, _KumoRMSNorm)
            assert current.weight is original.weight
            assert current.eps == original.eps
            assert not current.training
            changed += 1
        else:
            assert current is original
    assert changed > 0
    assert set(model.state_dict()) == keys
    assert all(param is weights[name] for name, param in model.named_parameters())
    model.load_state_dict(model.state_dict(), strict=True)
    _prepare_kumo_norms(model)  # Loading/preparing twice must be harmless.


@sdm_installed
@pytest.mark.skipif(
    os.environ.get("MLJAR_TEST_SDM_DEVICE") != "cuda",
    reason="opt-in CUDA normalization check",
)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("width", [128, 512, 1024])
@pytest.mark.parametrize("empty", [False, True])
def test_kumo_norm_cuda_matches_fallback(dtype, width, empty, torch_warn_always):
    import torch
    from sdm.nn import RMSNorm

    from supervised.algorithms._sdm_backend import _prepare_kumo_norms

    norm = RMSNorm(width).cuda().eval().requires_grad_(False)
    x = torch.randn(3, 5, width, device="cuda").to(getattr(torch, dtype))
    if empty:
        x = x[:0]
    model = torch.nn.Sequential(norm)
    with torch.inference_mode(), torch.autocast("cuda", dtype=getattr(torch, dtype)):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            expected = norm(x)
        _prepare_kumo_norms(model)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            actual = model(x)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cls", CLASSES)
def test_weights_rejected(cls):
    m = cls({"ml_task": "binary_classification"})
    with pytest.raises(ValueError, match="sample weights"):
        m.fit(np.zeros((3, 2)), [0, 1, 0], sample_weight=np.ones(3))
    with pytest.raises(ValueError, match="sample weights"):
        m.fit(np.zeros((3, 2)), [0, 1, 0], sample_weight_validation=np.ones(3))


@pytest.mark.parametrize("cls", CLASSES)
def test_context_sampling_preserves_rare_classes(cls):
    m = cls(
        {"ml_task": "multiclass_classification", "max_context_size": 10, "seed": 42}
    )
    y = np.array([0] * 100 + [1] * 3 + [2])
    a = m._context_indices(y)
    assert len(a) == 10 and len(set(a)) == 10
    assert set(y[a]) == {0, 1, 2}
    np.testing.assert_array_equal(a, m._context_indices(y))
    m.params["max_context_size"] = 2
    with pytest.raises(ValueError, match="number of classes"):
        m._context_indices(y)


@pytest.mark.parametrize(
    "options",
    [
        {"num_estimators": 0},
        {"max_context_size": True},
        {"device": "mps"},
        {"size": "extra-large"},
        {"lr": 0.1},
        {"memory_fraction": 2},
        {"precision": "bf16"},
        {"precision": 16},
        {"precision": None},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        validate_options(options, "Kumo")


@pytest.mark.parametrize("cls", CLASSES)
def test_fold_local_preprocessing(cls):
    m = cls({"ml_task": "binary_classification"})
    train = pd.DataFrame({"x": [1, np.nan, 3], "cat": ["a", "b", None]})
    a = m._preprocess(train, fit=True)
    b = m._preprocess(pd.DataFrame({"x": [np.nan], "cat": ["unseen"]}))
    assert b.f0.iloc[0] == 2
    assert len(m.category_maps["cat"]) == 2
    assert b.f1.iloc[0] == 1
    with pytest.raises(ValueError, match="sparse"):
        m._preprocess(sparse.csr_matrix(np.ones((3, 2))))


@pytest.mark.parametrize("cls", CLASSES)
def test_resource_and_deadline_rejection(cls):
    m = cls({"ml_task": "binary_classification", "memory_limit_gb": 0.001})
    with pytest.raises(Exception, match="RAM exceeds"):
        m._check_resources(100, 10)
    m._deadline = 0
    with pytest.raises(Exception, match="time budget"):
        m._check_time()


@pytest.mark.parametrize("cls", CLASSES)
def test_precision_resources_and_cpu_behavior(cls, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "supervised.algorithms.sdm.available_memory", lambda: 100 * 1024**3
    )
    monkeypatch.setattr(
        "supervised.algorithms.realmlp_resources.available_gpu_memory",
        lambda _: (100 * 1024**3, 100 * 1024**3),
    )
    m = cls({"ml_task": "binary_classification", "precision": "fp16"})
    assert m.effective_precision == "fp32"
    cpu = m._check_resources(1024, 10, 100_000_000)
    m.params["precision"] = "fp32"
    assert m._check_resources(1024, 10, 100_000_000) == cpu
    m.device = "cuda:0"
    fp32 = m._check_resources(1024, 10, 100_000_000)
    m.params["precision"] = "fp16"
    fp16 = m._check_resources(1024, 10, 100_000_000)
    assert m.effective_precision == "fp16"
    assert fp32["estimated_vram_bytes"] > fp16["estimated_vram_bytes"]
    assert fp32["estimated_ram_bytes"] > fp16["estimated_ram_bytes"]
    m.training_info = dict(
        device=m.device,
        context_rows=1024,
        effective_columns=10,
        checkpoint_bytes=100_000_000,
    )
    assert m.can_try_fp32(m.params)
    budget = (fp16["estimated_vram_bytes"] + fp32["estimated_vram_bytes"]) / 2 / 1024**3
    m.params["gpu_memory_limit_gb"] = budget
    m._check_resources(1024, 10, 100_000_000)  # FP16 still fits.
    assert not m.can_try_fp32(m.params)
    assert m.params["precision"] == "fp16"  # No silent candidate mutation.
    m.params.pop("gpu_memory_limit_gb")
    assert not m.can_try_fp32(dict(m.params, memory_limit_gb=0.001))
    # A resumed lazy fold reads lightweight metadata without loading weights.
    import json

    m.model_file_path = str(tmp_path / "learner_fold_0.kumo")
    (tmp_path / "learner_fold_0_training.log.json").write_text(
        json.dumps(m.training_info)
    )
    m.training_info = {}
    monkeypatch.setattr(
        m, "_load_network", lambda: pytest.fail("Preflight must not load a network")
    )
    assert m.can_try_fp32(m.params)
    (tmp_path / "learner_fold_0_training.log.json").unlink()
    assert not m.can_try_fp32(m.params)
    m.training_info = dict(device="cpu")
    assert not m.can_try_fp32(m.params)


@real_models
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize(
    "classification,regression",
    [
        (KumoClassificationAlgorithm, KumoRegressionAlgorithm),
        (TabICLv2ClassificationAlgorithm, TabICLv2RegressionAlgorithm),
    ],
)
@pytest.mark.parametrize(
    "task,n_classes",
    [("binary_classification", 2), ("multiclass_classification", 3), ("regression", 0)],
)
def test_checkpoint_predictions_and_portable_offline_reload(
    classification, regression, task, n_classes, precision, tmp_path, monkeypatch
):
    X, y = (
        make_classification(
            n_samples=500,
            n_features=10,
            n_informative=8,
            n_redundant=0,
            n_classes=n_classes,
            random_state=42,
        )
        if n_classes
        else make_regression(n_samples=500, n_features=10, noise=1, random_state=42)
    )
    X = pd.DataFrame(X)
    X["category"] = np.where(X[0] > 0, "high", "low")
    X.loc[::13, 1] = np.nan
    train, valid, yt, yv = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if n_classes else None
    )
    cls = classification if n_classes else regression
    m = cls(
        {
            "ml_task": task,
            "device": os.environ.get("MLJAR_TEST_SDM_DEVICE", "cpu"),
            "max_context_size": 256,
            "num_estimators": 2,
            "predict_batch_size": 128,
            "local_files_only": True,
            "n_jobs": 4,
            "precision": precision,
        }
    )
    m.fit(train, yt, X_validation=valid, y_validation=yv, max_time=120)
    assert m.training_info["requested_precision"] == precision
    assert m.training_info["precision"] == (
        precision if m.device.startswith("cuda") else "fp32"
    )
    p = m.predict(valid)
    assert np.isfinite(p).all()
    assert m.model._cache is None
    np.testing.assert_allclose(p[:13], m.predict(valid.iloc[:13]), atol=1e-6, rtol=1e-5)
    # Check mathematical equivalence in FP32. Splitting TabICLv2's context and
    # query GEMMs changes FP16 rounding (also present in uncached inference).
    # Request/reload consistency and query-shape tolerances are checked in the
    # actual production precision separately below.
    from unittest.mock import patch
    import torch

    scale = float(np.std(yt)) if not n_classes else 1.0
    with patch(
        "supervised.algorithms._sdm_backend._autocast",
        lambda _: torch.autocast(m.device.split(":")[0], enabled=False),
    ):
        cached_fp32 = m.predict(valid)
        with patch(
            "supervised.algorithms._sdm_backend._cache_fits_memory", return_value=False
        ):
            np.testing.assert_allclose(
                cached_fp32, m.predict(valid), atol=1e-5 * scale, rtol=1e-5
            )
    assert m.model._cache is None
    assert m.training_info["context_rows"] == 256
    if n_classes:
        np.testing.assert_allclose(m.predict_proba(valid).sum(1), 1, atol=1e-6)
        if n_classes == 2:
            assert roc_auc_score(yv, p) > 0.7
    else:
        assert r2_score(yv, p) > 0.8
    # A different configured batch shape changes FP16 GEMM rounding. Keep strict
    # same-configuration subset/reload checks above and below, while allowing a
    # bounded numerical difference between these different kernel shapes.
    m.params["predict_batch_size"] = 37
    scale = float(np.std(yt)) if not n_classes else 1.0
    tolerance = 5e-3 if m.effective_precision == "fp16" else 1e-5
    np.testing.assert_allclose(
        p, m.predict(valid), atol=tolerance * scale, rtol=tolerance
    )
    artifact = tmp_path / "original"
    artifact.mkdir()
    m.save(str(artifact / "fold0.model"))
    m.copy().save(str(artifact / "fold1.model"))
    assert (
        len(list((artifact / "sdm_weights").rglob("*.pt")))
        + len(list((artifact / "sdm_weights").rglob("*.ckpt")))
        == 1
    )
    assert list((artifact / "sdm_weights").rglob("*LICENSE"))
    moved = tmp_path / "moved"
    shutil.move(artifact, moved)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *a, **k: pytest.fail("Reload must not use HF"),
    )
    reloaded = cls({"ml_task": task})
    reloaded.load(str(moved / "fold0.model"))
    assert reloaded.params["precision"] == precision
    np.testing.assert_allclose(m.predict(valid), reloaded.predict(valid), atol=1e-6)


@real_models
@pytest.mark.parametrize("cls", CLASSES)
def test_twenty_classes(cls):
    X, y = make_classification(
        n_samples=400,
        n_features=12,
        n_informative=10,
        n_redundant=0,
        n_classes=20,
        n_clusters_per_class=1,
        random_state=12,
    )
    m = cls(
        {
            "ml_task": "multiclass_classification",
            "device": os.environ.get("MLJAR_TEST_SDM_DEVICE", "cpu"),
            "num_estimators": 2,
            "max_context_size": 256,
            "n_jobs": 4,
            "local_files_only": True,
        }
    )
    m.fit(X[:320], y[:320], X_validation=X[320:], y_validation=y[320:], max_time=120)
    p = m.predict(X[320:])
    assert p.shape == (80, 20)
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)


@real_models
def test_reject_truncation_and_constant_context_before_download(monkeypatch):
    m = KumoClassificationAlgorithm(
        {"ml_task": "binary_classification", "device": "cpu"}
    )
    monkeypatch.setattr(
        m, "_download", lambda: pytest.fail("Must reject before download")
    )
    X = np.random.default_rng(42).normal(size=(10, 501))
    with pytest.raises(ValueError, match="500 effective columns"):
        m.fit(X, np.arange(10) % 2, X_validation=X, y_validation=np.arange(10) % 2)
    with pytest.raises(ValueError, match="nonconstant"):
        m.fit(
            np.ones((10, 3)),
            np.arange(10) % 2,
            X_validation=np.ones((2, 3)),
            y_validation=[0, 1],
        )


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_kumo_size_validation_and_resource_scaling(size, monkeypatch):
    validate_options({"size": size}, "Kumo")
    with pytest.raises(ValueError, match="Unknown"):
        validate_options({"size": size}, "TabICLv2")
    monkeypatch.setattr(
        "supervised.algorithms.sdm.available_memory", lambda: 100 * 1024**3
    )
    models = [
        KumoClassificationAlgorithm({"ml_task": "binary_classification", "size": s})
        for s in ["small", "medium", "large"]
    ]
    estimates = [
        m._check_resources(4096, 20, 100_000_000)["estimated_ram_bytes"] for m in models
    ]
    assert estimates[0] < estimates[1] < estimates[2]


def test_sizes_do_not_overwrite_shared_weights(tmp_path):
    import joblib

    paths = []
    for size in ["small", "medium", "large"]:
        source = tmp_path / "source" / size
        source.mkdir(parents=True)
        (source / "classifier.pt").write_bytes(size.encode())
        (source / "LICENSE").write_text("test license")
        m = KumoClassificationAlgorithm(
            {"ml_task": "binary_classification", "size": size}
        )
        m.checkpoint_path = str(source / "classifier.pt")
        m.notice_paths = [str(source / "LICENSE")]
        m.checkpoint_info = {"revision": "test", "filename": size + "/classifier.pt"}
        artifact = tmp_path / (size + ".model")
        m.save(str(artifact), weights_root=tmp_path)
        path = tmp_path / joblib.load(artifact)["checkpoint_relative"]
        assert path.read_bytes() == size.encode()
        paths.append(path)
    assert len(set(paths)) == 3
    for path, size in zip(paths, ["small", "medium", "large"]):
        assert path.read_bytes() == size.encode()


@pytest.mark.skipif(
    os.environ.get("MLJAR_TEST_KUMO_SIZES") != "1",
    reason="opt-in all-size checkpoint integration",
)
@pytest.mark.parametrize("size", ["small", "medium", "large"])
@pytest.mark.parametrize(
    "task,classes",
    [("binary_classification", 2), ("multiclass_classification", 3), ("regression", 0)],
)
def test_all_kumo_sizes_synthetic_and_reload(size, task, classes, tmp_path):
    X, y = (
        make_classification(
            n_samples=300,
            n_features=8,
            n_informative=6,
            n_redundant=0,
            n_classes=classes,
            random_state=42,
        )
        if classes
        else make_regression(n_samples=300, n_features=8, noise=1, random_state=42)
    )
    cls = KumoClassificationAlgorithm if classes else KumoRegressionAlgorithm
    m = cls(
        {
            "ml_task": task,
            "size": size,
            "device": os.environ.get("MLJAR_TEST_SDM_DEVICE", "cpu"),
            "num_estimators": 2,
            "max_context_size": 128,
            "predict_batch_size": 64,
            "n_jobs": 4,
            "local_files_only": True,
        }
    )
    m.fit(X[:240], y[:240], X_validation=X[240:], y_validation=y[240:], max_time=180)
    p = m.predict(X[240:])
    assert np.isfinite(p).all()
    if classes:
        np.testing.assert_allclose(m.predict_proba(X[240:]).sum(1), 1, atol=1e-6)
    else:
        assert r2_score(y[240:], p) > 0.8
    assert m.checkpoint_info["filename"].startswith(size + "/")
    artifact = str(tmp_path / "fold.model")
    m.save(artifact)
    del m
    loaded = cls({"ml_task": task})
    loaded.load(artifact)
    assert loaded.params["size"] == size
    np.testing.assert_allclose(p, loaded.predict(X[240:]), atol=1e-6)
