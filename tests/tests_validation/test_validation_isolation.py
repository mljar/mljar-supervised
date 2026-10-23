"""Row selection must preserve sklearn splits and isolate every training consumer."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from supervised.preprocessing.preprocessing import Preprocessing
from supervised.utils.utils import Store, dump_data, load_data
from supervised.validation.validator_custom import CustomValidator
from supervised.validation.validator_kfold import KFoldValidator
from supervised.validation.validator_split import SplitValidator


def inputs(tmp_path, index=None):
    n = 24
    frames = {
        "X": pd.DataFrame(
            {
                "number": np.arange(n, dtype=float),
                "missing": [np.nan, 1.0, 2.0] * 8,
                "category": pd.Categorical(["a", "b", "c"] * 8),
                "nullable": pd.Series([1, None, 3] * 8, dtype="Int64"),
                "date": pd.date_range("2020-01-01", periods=n),
                "constant": np.ones(n),
            }
        ),
        "y": pd.DataFrame({"target": ["a", "b", "c"] * 8}),
        "sample_weight": pd.DataFrame({"sample_weight": np.linspace(0.5, 2, n)}),
        "sensitive_features": pd.DataFrame({"group": ["u", "v"] * 12}),
    }
    store = Store()
    params = {"results_path": str(tmp_path)}
    for name, frame in frames.items():
        if index is not None:
            frame.index = index
        path = str(tmp_path / (name + ".data"))
        dump_data(path, frame, store=store)
        params[name + "_path"] = path
    return frames, store, params


def expected(frames, rows, positional=True):
    result = {}
    for key, frame in frames.items():
        subset = frame.iloc[rows] if positional else frame.loc[rows]
        result[key] = subset.iloc[:, 0] if key in ("y", "sample_weight") else subset
    return result


def assert_data(actual, reference):
    assert actual.keys() == reference.keys()
    for key in reference:
        if isinstance(reference[key], pd.Series):
            pd.testing.assert_series_equal(actual[key], reference[key])
        else:
            pd.testing.assert_frame_equal(actual[key], reference[key])


def assert_isolated(
    validator,
    frames,
    store,
    params,
    train_rows,
    valid_rows,
    k=0,
    repeat=0,
    positional=True,
):
    train, valid = validator.get_split(k, repeat)
    assert_data(train, expected(frames, train_rows, positional))
    assert_data(valid, expected(frames, valid_rows, positional))
    saved = {name: frame.copy(deep=True) for name, frame in frames.items()}
    # Overlapping custom folds and subsequent models must survive mutations.
    train["X"].iloc[0, 0] = -999
    train["X"].drop(columns=["constant"], inplace=True)
    train["y"].iloc[0] = "changed"
    train["sample_weight"].iloc[0] = -1
    train["sensitive_features"].iloc[0, 0] = "changed"
    assert_data(valid, expected(saved, valid_rows, positional))
    valid["X"].iloc[0, 1] = -999
    for key, frame in frames.items():
        pd.testing.assert_frame_equal(frame, saved[key])
        pd.testing.assert_frame_equal(
            load_data(params[key + "_path"], store=store), saved[key]
        )
    fresh_train, fresh_valid = validator.get_split(k, repeat)
    assert_data(fresh_train, expected(saved, train_rows, positional))
    assert_data(fresh_valid, expected(saved, valid_rows, positional))


@pytest.mark.parametrize(
    "shuffle,stratify", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("seed", [17, 1234])
@pytest.mark.parametrize(
    "optional",
    [
        (),
        ("sample_weight",),
        ("sensitive_features",),
        ("sample_weight", "sensitive_features"),
    ],
)
def test_split_matches_sklearn_and_isolates(
    tmp_path, shuffle, stratify, seed, optional
):
    frames, store, params = inputs(tmp_path, index=np.arange(24) * 3 + 100)
    for name in ("sample_weight", "sensitive_features"):
        if name not in optional:
            del frames[name]
            del params[name + "_path"]
    params.update(
        shuffle=shuffle,
        stratify=stratify,
        random_seed=seed,
        train_ratio=0.625,
        repeats=2 if shuffle else 1,
    )
    validator = SplitValidator(params, data_store=store)
    for repeat in range(validator.get_repeats()):
        ti, vi = train_test_split(
            np.arange(24),
            train_size=0.625,
            test_size=0.375,
            shuffle=shuffle,
            stratify=frames["y"].target if stratify and shuffle else None,
            random_state=seed + repeat,
        )
        if optional == ("sample_weight", "sensitive_features"):
            assert_isolated(validator, frames, store, params, ti, vi, repeat=repeat)
        else:
            train, valid = validator.get_split(repeat=repeat)
            assert_data(train, expected(frames, ti))
            assert_data(valid, expected(frames, vi))
        prefix = f"repeat_{repeat}_" if shuffle else ""
        np.testing.assert_array_equal(
            np.load(tmp_path / f"split_{prefix}train_indices.npy"),
            frames["X"].index[ti],
        )
        np.testing.assert_array_equal(
            np.load(tmp_path / f"split_{prefix}validation_indices.npy"),
            frames["X"].index[vi],
        )


@pytest.mark.parametrize("stratify", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
def test_kfold_repeats_and_saved_indices(tmp_path, stratify, shuffle):
    frames, store, params = inputs(tmp_path)
    params.update(
        k_folds=3,
        repeats=2 if shuffle else 1,
        shuffle=shuffle,
        stratify=stratify,
        random_seed=91,
    )
    validator = KFoldValidator(params, data_store=store)
    cls = StratifiedKFold if stratify else KFold
    for repeat in range(validator.get_repeats()):
        splitter = cls(
            3, shuffle=shuffle, random_state=91 + repeat if shuffle else None
        )
        for k, (ti, vi) in enumerate(splitter.split(frames["X"], frames["y"].target)):
            assert_isolated(
                validator, frames, store, params, ti, vi, k, repeat, positional=False
            )
    # A second model reuses saved folds and still sees original values.
    second = KFoldValidator(params, data_store=store)
    for k in range(3):
        first_pair, second_pair = validator.get_split(k), second.get_split(k)
        for a, b in zip(first_pair, second_pair):
            assert_data(a, b)


@pytest.mark.parametrize("boolean", [False, True])
def test_custom_overlap_and_nonstandard_index(tmp_path, boolean):
    frames, store, params = inputs(tmp_path, index=np.arange(24) * 3 + 100)
    ti, vi = np.array([4, 1, 4, 2, 3]), np.array([1, 2, 5])
    if boolean:
        ti, vi = np.arange(24) < 16, np.arange(24) >= 8
    params["cv_path"] = str(tmp_path / "cv.joblib")
    joblib.dump([(ti, vi)], params["cv_path"])
    validator = CustomValidator(params, data_store=store)
    assert_isolated(validator, frames, store, params, ti, vi)


@pytest.mark.parametrize("kind", ["split", "kfold", "custom"])
def test_real_preprocessing_does_not_mutate_next_model(tmp_path, monkeypatch, kind):
    frames, store, params = inputs(tmp_path)
    # Fail if validation tries to copy the entire feature matrix.
    original_get = store.get

    def get(key):
        assert key != params["X_path"]
        return original_get(key)

    monkeypatch.setattr(store, "get", get)
    params.update(k_folds=3, stratify=True, shuffle=True)
    params["cv_path"] = str(tmp_path / "cv.joblib")
    joblib.dump([(np.arange(16), np.arange(8, 24))], params["cv_path"])
    cls = {"split": SplitValidator, "kfold": KFoldValidator, "custom": CustomValidator}[
        kind
    ]
    validator = cls(params, data_store=store)
    train, valid = validator.get_split(0)
    originals = tuple(
        {k: v.copy(deep=True) for k, v in data.items()} for data in (train, valid)
    )
    config = {
        "ml_task": "multiclass_classification",
        "target_preprocessing": ["categorical_to_int"],
        "columns_preprocessing": {
            "constant": ["remove_column"],
            "number": ["scale_normal"],
            "missing": ["na_fill_median"],
            "category": ["categorical_to_int"],
        },
    }
    prep = Preprocessing(config)
    prep.fit_and_transform(train["X"], train["y"], train["sample_weight"])
    prep.transform(valid["X"], valid["y"], valid["sample_weight"])
    for data, original in zip(cls(params, data_store=store).get_split(0), originals):
        assert_data(data, original)


def test_missing_targets_and_inconsistent_lengths(tmp_path):
    frames, store, params = inputs(tmp_path)
    frames["y"].iloc[0, 0] = None
    validator = SplitValidator(dict(params, shuffle=False), data_store=store)
    train, valid = validator.get_split()
    assert train["y"].isna().sum() == 1
    store.set(params["sample_weight_path"], frames["sample_weight"].iloc[:-1])
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        validator.get_split()


@pytest.mark.parametrize("kind", ["split", "kfold", "custom"])
def test_multiple_models_match_full_copy_reference(tmp_path, monkeypatch, kind):
    from supervised import AutoML
    from tests.tests_automl.test_training_store_lifecycle import automl

    rng = np.random.default_rng(91)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    y = X.a.to_numpy() * 2 + rng.normal(size=120) * 0.1
    X.loc[::7, "b"] = np.nan
    X["category"] = ["a", "b", "c"] * 40
    original = X.copy(deep=True)
    strategy = {"validation_type": kind, "shuffle": True, "stratify": False}
    fit_args = {}
    if kind == "kfold":
        strategy.update(k_folds=3, repeats=2)
    elif kind == "custom":
        fit_args["cv"] = list(KFold(3, shuffle=True, random_state=91).split(X))
    else:
        strategy.update(train_ratio=0.8, repeats=2)
    settings = dict(
        algorithms=["Linear", "Decision Tree"], validation_strategy=strategy
    )
    optimized = automl(tmp_path / "selected", **settings)
    optimized.fit(X, y, **fit_args)

    # Reproduce the previous full-copy-before-selection behavior.
    def full_copy(self, key, rows, *, positional=True):
        frame = self.get(key)
        return frame.iloc[rows] if positional else frame.loc[rows]

    monkeypatch.setattr(Store, "get_rows", full_copy)
    reference = automl(tmp_path / "reference", **settings)
    reference.fit(X, y, **fit_args)
    assert len(optimized._models) == len(reference._models) == 2
    for actual, expected_model in zip(optimized._models, reference._models):
        assert actual.get_name() == expected_model.get_name()
        np.testing.assert_allclose(
            actual.predict(X), expected_model.predict(X), rtol=0, atol=0
        )
    np.testing.assert_allclose(
        optimized.predict(X), reference.predict(X), rtol=0, atol=0
    )
    reloaded = AutoML(results_path=str(tmp_path / "selected"))
    np.testing.assert_allclose(
        optimized.predict(X), reloaded.predict(X), rtol=0, atol=0
    )
    pd.testing.assert_frame_equal(X, original)
