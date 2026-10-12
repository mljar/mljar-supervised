import numpy as np
import pandas as pd
import pytest

from supervised import AutoML


@pytest.mark.parametrize("labels", [[0, 1], [1, 2], [0, 1, 2]])
def test_lightgbm_categorical_target_fits_and_reloads(tmp_path, labels):
    values = np.tile(labels, 60)
    rng = np.random.default_rng(652)
    X = pd.DataFrame({"enum": values, "number": rng.normal(size=len(values))})
    X["enum"] = X["enum"].astype("category")
    y = pd.Series(
        pd.Categorical(values, categories=[99] + labels[::-1], ordered=True),
        name="target",
    )
    original_X, original_y = X.copy(), y.copy()
    directory = tmp_path / "categorical_target"
    model = AutoML(
        results_path=str(directory),
        mode="Explain",
        algorithms=["LightGBM"],
        total_time_limit=30,
        train_ensemble=False,
        explain_level=0,
        start_random_models=1,
        hill_climbing_steps=0,
        top_models_to_improve=0,
        n_jobs=1,
        verbose=0,
        validation_strategy={
            "validation_type": "split",
            "train_ratio": 0.8,
            "shuffle": True,
            "stratify": True,
        },
        random_state=652,
    )
    model.fit(X, y)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    assert set(predictions.astype(str)) <= {str(label) for label in labels}
    if labels == [0, 1]:
        assert pd.api.types.is_integer_dtype(predictions.dtype)
    assert probabilities.shape == (len(X), len(labels))
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert model._models[0].preprocessings[0].get_target_class_names() == [
        str(label) for label in labels
    ]

    restored = AutoML(results_path=str(directory))
    np.testing.assert_array_equal(restored.predict(X), predictions)
    np.testing.assert_allclose(restored.predict_proba(X), probabilities)
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)
