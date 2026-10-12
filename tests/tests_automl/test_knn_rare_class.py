import numpy as np
import pandas as pd
import pytest

from supervised import AutoML


def test_custom_fold_singleton_class_fits_and_reloads(tmp_path):
    rng = np.random.default_rng(1234)
    X = pd.DataFrame(rng.normal(size=(2000, 3)), columns=["a", "b", "c"])
    y = np.r_[np.zeros(1980, dtype=int), np.ones(20, dtype=int)]
    train = np.r_[np.arange(1960), 1980]
    validation = np.r_[np.arange(1960, 1980), np.arange(1981, 2000)]
    assert np.bincount(y[train]).tolist() == [1960, 1]
    original = X.copy()
    directory = tmp_path / "rare_class"
    model = AutoML(
        results_path=str(directory),
        algorithms=["Nearest Neighbors"],
        ml_task="binary_classification",
        explain_level=0,
        validation_strategy={"validation_type": "custom"},
        train_ensemble=False,
        stack_models=False,
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        boost_on_errors=False,
        mix_encoding=False,
        start_random_models=1,
        hill_climbing_steps=0,
        total_time_limit=120,
        n_jobs=1,
        verbose=0,
    )
    with pytest.warns(UserWarning, match="class-preserving subsampling"):
        model.fit(X, y, cv=[(train, validation)])
    assert len(model._models) == 1
    member = model._models[0]
    assert np.isfinite(member.get_final_loss())
    expected = model.predict_proba(X.iloc[validation])
    assert expected.shape == (len(validation), 2)
    np.testing.assert_allclose(expected.sum(axis=1), 1)
    assert member.learners[0].model.n_samples_fit_ == 1000
    np.testing.assert_array_equal(member.learners[0].model.classes_, [0, 1])
    loaded = AutoML(results_path=str(directory))
    np.testing.assert_array_equal(loaded.predict_proba(X.iloc[validation]), expected)
    pd.testing.assert_frame_equal(X, original)
    assert getattr(model, "_training_store", None) is None
