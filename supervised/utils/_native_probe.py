"""Standalone diagnostic: execute by file path, never import AutoML to launch it."""

import json
import sys


def main(order):
    # No core dumps or model/data downloads for this disposable diagnostic.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass
    if order == "torch-first":
        import torch
        import lightgbm
        import xgboost
    elif order == "boosting-first":
        import lightgbm
        import xgboost
        import torch
    else:
        raise ValueError("Unknown probe order")
    import numpy as np
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(2)
    x = torch.randn(400, 10, requires_grad=True)
    assert x.min(dim=0).values.shape == (10,)
    x.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    X = np.random.default_rng(42).normal(size=(80, 4))
    y = (X[:, 0] > 0).astype(int)
    for model in (
        lightgbm.LGBMClassifier(n_estimators=3, n_jobs=2, verbosity=-1),
        xgboost.XGBClassifier(n_estimators=3, n_jobs=2, tree_method="hist"),
    ):
        model.fit(X, y)
        assert np.isfinite(model.predict_proba(X)).all()
    with threadpool_limits(limits=2):
        KMeans(n_clusters=2, n_init=1, random_state=42).fit(X)
    assert torch.randn(400, 10).min(dim=0).values.shape == (10,)
    print(json.dumps(dict(status="passed", order=order)), flush=True)


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except ImportError:
        print(json.dumps(dict(reason="import_error")), flush=True)
        raise
