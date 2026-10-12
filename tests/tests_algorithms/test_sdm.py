"""Unit checks run without SDM; real checkpoint tests require MLJAR_TEST_SDM=1."""

import os
import shutil

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


@real_models
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
    classification, regression, task, n_classes, tmp_path, monkeypatch
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
        }
    )
    m.fit(train, yt, X_validation=valid, y_validation=yv, max_time=120)
    p = m.predict(valid)
    assert np.isfinite(p).all()
    assert m.training_info["context_rows"] == 256
    if n_classes:
        np.testing.assert_allclose(m.predict_proba(valid).sum(1), 1, atol=1e-6)
        if n_classes == 2:
            assert roc_auc_score(yv, p) > 0.7
    else:
        assert r2_score(yv, p) > 0.8
    # Query batch boundaries should not change predictions.
    m.params["predict_batch_size"] = 37
    # FP16 attention kernels may vary slightly with the query shape.
    scale = float(np.std(yt)) if not n_classes else 1.0
    np.testing.assert_allclose(p, m.predict(valid), atol=1e-3 * scale, rtol=1e-3)
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
