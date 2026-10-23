import json

import numpy as np
import pandas as pd
import pytest

from supervised.apps.metadata import _build_feature
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.preprocessing.ip_transformer import IPTransformer
from supervised.preprocessing.preprocessing import Preprocessing
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.tuner.data_info import DataInfo
from supervised.tuner.preprocessing_tuner import PreprocessingTuner


@pytest.mark.parametrize("dtype", [object, "str", "string"])
def test_detect_mixed_addresses(dtype):
    values = pd.Series([" 192.0.2.1 ", "2001:db8::1", None], dtype=dtype)
    assert PreprocessingUtils.get_type(values) == PreprocessingUtils.IP
    assert PreprocessingUtils.is_ip(values)
    assert not PreprocessingUtils.is_text(values)
    assert not PreprocessingUtils.is_categorical(values)


@pytest.mark.parametrize(
    "values",
    [
        ["192.0.2.1", "a category"],
        ["192.0.2.1", "999.0.0.1"],
        ["192.0.2.0/24"],
        ["192.0.2.1:80"],
        ["fe80::1%eth0"],
        [None, pd.NA],
        [1, 2],
        ["1", "2"],
    ],
)
def test_non_ip_columns(values):
    assert not PreprocessingUtils.is_ip(pd.Series(values))


def test_categorical_and_high_cardinality_detection():
    values = pd.Series([f"192.0.2.{i}" for i in range(256)])
    assert PreprocessingUtils.get_type(values) == PreprocessingUtils.IP
    assert (
        PreprocessingUtils.get_type(values.astype("category"))
        == PreprocessingUtils.CATEGORICAL
    )


def test_components_and_equivalent_ipv6():
    X = pd.DataFrame(
        {
            "ip": [
                "255.255.255.255",
                "2001:db8::1",
                "2001:0db8:0000:0000:0000:0000:0000:0001",
                None,
                "0.0.0.0",
                "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
            ]
        },
        index=[10, 20, 30, 40, 50, 60],
    )
    transformer = IPTransformer()
    transformer.fit(X, "ip")
    result = transformer.transform(X.copy())
    assert "ip" not in result
    assert result.shape == (6, 14)
    np.testing.assert_array_equal(result.loc[10].values, [4] + [255] * 4 + [0] * 9)
    np.testing.assert_array_equal(
        result.loc[20].values, [6] + [0] * 4 + [8193, 3512, 0, 0, 0, 0, 0, 1, 0]
    )
    np.testing.assert_array_equal(result.loc[20], result.loc[30])
    np.testing.assert_array_equal(result.loc[40].values, [0] * 13 + [1])
    np.testing.assert_array_equal(result.loc[50].values, [4] + [0] * 13)
    np.testing.assert_array_equal(
        result.loc[60].values, [6] + [0] * 4 + [65535] * 8 + [0]
    )
    restored = IPTransformer()
    restored.from_json(json.loads(json.dumps(transformer.to_json())))
    pd.testing.assert_frame_equal(result, restored.transform(X.copy()))
    assert restored.transform(X.iloc[:0].copy()).shape == (0, 14)


def test_invalid_input_and_name_collision():
    transformer = IPTransformer()
    transformer.fit(pd.DataFrame({"ip": ["192.0.2.1", "2001:db8::1"]}), "ip")
    for value in ("bad", "192.0.2.0/24", "fe80::1%eth0", 123):
        with pytest.raises(ValueError, match="column 'ip', row 'row-id'"):
            transformer.transform(pd.DataFrame({"ip": [value]}, index=["row-id"]))
    collision = pd.DataFrame(
        {"ip": ["192.0.2.1", "2001:db8::1"], "ip_IP_Version": [99, 98]}
    )
    with pytest.raises(ValueError, match="already exist"):
        transformer.fit(collision, "ip")
    with pytest.raises(ValueError, match="already exist"):
        transformer.transform(collision)


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("training_missing", [False, True])
def test_pipeline_selection_scaling_and_reload(scaled, training_missing):
    X = pd.DataFrame({"ip": ["192.0.2.1", "192.0.2.2", None, "192.0.2.4"]})
    if not training_missing:
        X.loc[2, "ip"] = "192.0.2.3"
    y = pd.Series([0, 1, 0, 1])
    info = DataInfo.compute(X, y, "binary_classification")
    assert info["columns_info"]["ip"] == (
        ["missing_values"] if training_missing else []
    ) + ["ip_transform"]
    required = ["ip_transform"] + (["scale"] if scaled else [])
    params = PreprocessingTuner.get(required, info, "binary_classification")
    assert params["columns_preprocessing"]["ip"] == ["ip_transform"] + (
        ["scale_normal"] if scaled else []
    )
    pipeline = Preprocessing(params)
    result, _, _ = pipeline.fit_and_transform(X.copy(), y.copy())
    expected_columns = ["ip_IP_IPv4_4"]
    if training_missing:
        expected_columns = [
            "ip_IP_Version",
            "ip_IP_IPv4_1",
            "ip_IP_IPv4_3",
            "ip_IP_IPv4_4",
            "ip_IP_Missing",
        ]
    assert list(result.columns) == expected_columns
    assert result.nunique().gt(1).all()
    assert not result.isna().any().any()
    if scaled:
        np.testing.assert_allclose(result.mean(), 0, atol=1e-12)
    new = pd.DataFrame({"ip": ["2001:db8::9", None]})
    expected, _, _ = pipeline.transform(new.copy(), None)
    assert list(expected.columns) == expected_columns
    restored = Preprocessing()
    restored.from_json(json.loads(json.dumps(pipeline.to_json())), ".")
    actual, _, _ = restored.transform(new.copy(), None)
    pd.testing.assert_frame_equal(expected, actual)


def test_all_feature_consuming_algorithms_select_ip_transform():
    info = {
        "columns_info": {"ip": ["missing_values", "ip_transform"]},
        "target_info": [],
    }
    for task, algorithms in AlgorithmsRegistry.registry.items():
        for name, algorithm in algorithms.items():
            if name == "Baseline":
                continue
            required = algorithm["required_preprocessing"]
            params = PreprocessingTuner.get(required, info, task)
            steps = params["columns_preprocessing"]["ip"]
            assert steps == ["ip_transform"] + (
                ["scale_normal"] if "scale" in required else []
            ), (task, name)


def test_ipv6_constants_are_pruned_and_validation_does_not_add_columns():
    training = pd.DataFrame({"ip": ["2001:db8::1", "2001:db8::2"]})
    transformer = IPTransformer()
    transformer.fit(training, "ip")
    assert transformer._new_columns == ["ip_IP_IPv6_8"]
    validation = pd.DataFrame(
        {"ip": ["ffff:ffff:ffff:ffff:ffff:ffff:ffff:3", "192.0.2.4", None]}
    )
    actual = transformer.transform(validation.copy())
    pd.testing.assert_frame_equal(
        actual, pd.DataFrame({"ip_IP_IPv6_8": [3.0, 0.0, 0.0]})
    )
    restored = IPTransformer()
    restored.from_json(json.loads(json.dumps(transformer.to_json())))
    pd.testing.assert_frame_equal(actual, restored.transform(validation.copy()))


def test_constant_ip_transform_outputs_no_columns():
    training = pd.DataFrame({"ip": ["192.0.2.1", "192.0.2.1"], "count": [1, 2]})
    transformer = IPTransformer()
    transformer.fit(training, "ip")
    assert transformer._new_columns == []
    pd.testing.assert_frame_equal(
        transformer.transform(training.copy()), training[["count"]]
    )


def test_version_one_saved_transform_keeps_original_schema():
    source = pd.DataFrame(
        {"ip": ["255.255.255.255", "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", None]}
    )
    transformer = IPTransformer()
    transformer.fit(source, "ip")
    legacy = transformer.to_json()
    legacy["version"] = 1
    legacy.pop("active_indices")
    restored = IPTransformer()
    restored.from_json(legacy)
    actual = restored.transform(pd.DataFrame({"ip": ["192.0.2.1"]}))
    assert actual.shape == (1, 14)
    assert actual["ip_IP_Version"].iloc[0] == 4
    assert actual["ip_IP_Missing"].iloc[0] == 0


def test_empty_constant_and_app_schema():
    X = pd.DataFrame({"empty": [None, None], "constant": ["192.0.2.1"] * 2})
    info = DataInfo.compute(X, pd.Series([0, 1]), "binary_classification")
    assert info["columns_info"] == {
        "empty": ["empty_column"],
        "constant": ["constant_column"],
    }
    feature = _build_feature(
        "ip", ["ip_transform"], pd.Series(["192.0.2.1", "2001:db8::1"])
    )
    assert (feature["widget"], feature["dtype"]) == ("text", "str")
    fallback = _build_feature("ip", ["ip_transform"], None, ["192.0.2.1"])
    assert (fallback["widget"], fallback["dtype"]) == ("text", "str")
    restored = Preprocessing()
    restored.from_json(
        {"params": {"columns_preprocessing": {}, "target_preprocessing": []}}, "."
    )
    assert restored._ip_transforms == []
