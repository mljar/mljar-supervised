"""Optional backend calls must be quiet without hiding validation failures."""

import importlib.util
import logging
import threading
import warnings
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_classification, make_regression


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pytabkit") is None, reason="optional pytabkit backend"
)


def test_compatibility_restores_globals_after_nested_failure(caplog):
    from pytabkit.models.sklearn import sklearn_base
    from pytorch_lightning.utilities import _pytree
    from torch.utils._pytree import tree_unflatten

    from supervised.algorithms._pytabkit_compat import backend_context

    originals = (
        sklearn_base.check_X_y_wrapper,
        sklearn_base.check_array_wrapper,
        _pytree.LeafSpec,
    )
    logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
    upstream_logger = logging.getLogger("lightning")
    original_filters = list(logger.filters)
    old_level, old_propagate = upstream_logger.level, upstream_logger.propagate
    old_warning_filters = list(warnings.filters)
    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError, match="failure"):
            with backend_context(), backend_context():
                logger.info("hidden backend info")
                logger.warning("visible backend warning")
                logging.getLogger("supervised.test").info("visible MLJAR info")
                other = threading.Thread(
                    target=logger.info, args=("other thread info",)
                )
                other.start()
                other.join()
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    flat, spec = _pytree._tree_flatten({"a": [1, 2], "b": None})
                    assert tree_unflatten(flat, spec) == {"a": [1, 2], "b": None}
                # Simulate the upstream logger/filter side effects before failure.
                upstream_logger.setLevel(logging.ERROR)
                upstream_logger.propagate = False
                warnings.filterwarnings("ignore", message="upstream change")
                raise RuntimeError("failure")
        logger.info("restored backend info")
    assert "hidden backend info" not in caplog.text
    assert "visible backend warning" in caplog.text
    assert "visible MLJAR info" in caplog.text
    assert "other thread info" in caplog.text
    assert "restored backend info" in caplog.text
    assert originals == (
        sklearn_base.check_X_y_wrapper,
        sklearn_base.check_array_wrapper,
        _pytree.LeafSpec,
    )
    assert logger.filters == original_filters
    assert (upstream_logger.level, upstream_logger.propagate) == (
        old_level,
        old_propagate,
    )
    assert warnings.filters == old_warning_filters


@pytest.mark.parametrize("keyword", ["force_all_finite", "ensure_all_finite"])
def test_finite_validation_still_rejects_infinity(keyword):
    from pytabkit.models.sklearn import sklearn_base
    from supervised.algorithms._pytabkit_compat import backend_context

    with backend_context(), warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        for check, args in [
            (sklearn_base.check_array_wrapper, ()),
            (sklearn_base.check_X_y_wrapper, ([0, 1],)),
        ]:
            check([[1.0], [np.nan]], *args, **{keyword: "allow-nan"})
            with pytest.raises(ValueError, match="infinity"):
                check([[1.0], [np.inf]], *args, **{keyword: "allow-nan"})


def test_single_bin_diagnostic_preserves_encoding_and_other_warnings():
    import torch
    from pytabkit.models.nn_models import rtdl_num_embeddings as embeddings
    from supervised.algorithms._pytabkit_compat import single_bin_diagnostics

    bins = [torch.tensor([0.0, 1.0])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        reference = embeddings.PiecewiseLinearEncoding(bins)
    with single_bin_diagnostics():
        with pytest.warns(UserWarning, match="unrelated"):
            warnings.warn("unrelated warning", UserWarning)
        actual = embeddings.PiecewiseLinearEncoding(bins)
        embeddings._check_bins(bins)
        with pytest.raises(ValueError, match="at least two"):
            embeddings._check_bins([torch.tensor([0.0])])
    x = torch.tensor([[-1.0], [0.0], [0.5], [1.0], [2.0]])
    assert_allclose(actual(x), reference(x), rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", ["RealMLP", "TabM"])
@pytest.mark.parametrize(
    "task", ["binary_classification", "multiclass_classification", "regression"]
)
def test_fit_predict_reload_without_reported_warnings(
    algorithm, task, tmp_path, caplog
):
    from supervised.algorithms.realmlp import (
        RealMLPClassificationAlgorithm,
        RealMLPRegressionAlgorithm,
    )
    from supervised.algorithms.tabm import (
        TabMClassificationAlgorithm,
        TabMRegressionAlgorithm,
    )

    if task == "regression":
        X, y = make_regression(n_samples=120, n_features=6, random_state=43)
    else:
        X, y = make_classification(
            n_samples=120,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            n_classes=3 if task.startswith("multi") else 2,
            random_state=43,
        )
    X = pd.DataFrame(X)
    X["binary"] = (np.arange(len(X)) % 5 == 0).astype(float)
    X.loc[::7, 0] = np.nan
    cls = {
        ("RealMLP", False): RealMLPClassificationAlgorithm,
        ("RealMLP", True): RealMLPRegressionAlgorithm,
        ("TabM", False): TabMClassificationAlgorithm,
        ("TabM", True): TabMRegressionAlgorithm,
    }[algorithm, task == "regression"]
    params = dict(ml_task=task, device="cpu", n_jobs=2, n_epochs=2, seed=42)
    model = cls(params)
    with caplog.at_level(logging.INFO), warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.filterwarnings(
            "error", message=r"The \d+-th feature has just two bin edges"
        )
        model.fit(X[:90], y[:90], X_validation=X[90:], y_validation=y[90:])
        expected = model.predict(X[90:])
        path = tmp_path / "model"
        model.save(path)
        loaded = cls(params)
        loaded.load(path)
        assert_allclose(loaded.predict(X[90:]), expected, rtol=1e-6, atol=1e-6)
    assert np.isfinite(expected).all()
    assert not any(
        r.name.startswith("pytorch_lightning") and r.levelno < logging.WARNING
        for r in caplog.records
    )
    if algorithm == "TabM":
        assert model.training_info["single_bin_feature_count"] > 0
        assert (
            loaded.training_info["single_bin_feature_count"]
            == model.training_info["single_bin_feature_count"]
        )

    # Compatibility changes API spelling/logging, not prediction semantics.
    from supervised.algorithms import _pytabkit_compat

    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        patch.setattr(_pytabkit_compat, "backend_context", nullcontext)
        assert_allclose(model.predict(X[90:]), expected, rtol=0, atol=0)
