import copy
import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from supervised.exceptions import AutoMLException
from supervised.preprocessing.feature_identity import (
    build_identities, digest, file_digest, input_identity, reject_generated_drops,
)
from supervised.preprocessing.feature_discovery_transformer import FeatureDiscoveryTransformer, fingerprint
from supervised.preprocessing.preprocessing import Preprocessing
from supervised.utils.additional_metrics import AdditionalMetrics
from supervised.utils.feature_importance_identity import importance_matrix


def formula(op, columns=("a", "b"), fill=None):
    return dict(feature_type="numeric_formula", formula_name=op,
                source_columns=list(columns), output_fill=fill)


def test_stateless_identity_is_order_independent_but_operations_are_not():
    defs = {"multiply": formula("mul"), "divide": formula("div")}
    first = build_identities(list(defs), defs, "fold0", ["a", "b"], "1.1")
    second = build_identities(list(reversed(defs)), defs, "fold1", ["a", "b"], "1.1")
    assert first == second
    reversed_mul = build_identities(["different label"], {"different label": formula("mul", ("b", "a"))}, "fold2", ["a", "b"], "1.1")
    assert first["multiply"]["identity"] == reversed_mul["different label"]["identity"]
    for op in ["sub", "div"]:
        left = build_identities(["slot0"], {"slot0": formula(op)}, "fold0", ["a", "b"], "1.1")
        right = build_identities(["slot0"], {"slot0": formula(op, ("b", "a"))}, "fold0", ["a", "b"], "1.1")
        assert left["slot0"]["identity"] != right["slot0"]["identity"]
    assert first["multiply"]["identity"] != first["divide"]["identity"]


@pytest.mark.parametrize("definition", [{}, {"feature_type": "projection_pca"}, formula("mul", fill=1.5), {"feature_type": "binary_rule", "metadata": {"conditions": [{"feature": "a", "op": "<=", "threshold": 3}]}}])
def test_learned_and_unknown_outputs_stay_fitted_scope(definition):
    a = build_identities(["component2"], {"component2": definition}, "fit-a", ["a", "b"], "1.1")
    b = build_identities(["component2"], {"component2": definition}, "fit-b", ["a", "b"], "1.1")
    assert a["component2"]["identity"] != b["component2"]["identity"]


def test_hash_collisions_fail_closed(monkeypatch):
    monkeypatch.setattr("supervised.preprocessing.feature_identity.digest", lambda x: "collision")
    with pytest.raises(AutoMLException, match="collision"):
        build_identities(["a", "b"], {"a": formula("mul"), "b": formula("div")}, "fit", ["a", "b"], "1.1")


def manifest(root, scope="fold0", definitions=None):
    root.mkdir(exist_ok=True)
    (root / "generator.joblib").write_bytes(b"fitted-generator")
    sha = file_digest(root / "generator.joblib")
    generator = digest(dict(fit_scope=scope, state=sha))
    definitions = definitions or {"multiply": formula("mul"), "pca": {"feature_type": "projection_pca"}}
    records = build_identities(list(definitions), definitions, generator, ["a", "b"], "1.1")
    X = pd.DataFrame({"a": [1., 2., 3.], "b": [3., 2., 1.]})
    y = pd.Series([0, 1, 0], name="target")
    info = dict(identity_schema_version=1, generator_id=generator, generator_sha256=sha,
                fit_scope=scope, feature_identities=records, fingerprint=fingerprint(X, y), fingerprint_version=2,
                input_columns=list(X), directory=".", sets={"default": list(definitions)},
                rename={n: r["storage_column"] for n, r in records.items()})
    joblib.dump(pd.DataFrame({n: [-123.] * 3 for n in definitions}), root / "training.joblib")
    return info, X, y


def test_replay_order_state_binding_and_training_encodings(tmp_path):
    info, X, y = manifest(tmp_path)
    adapter = FeatureDiscoveryTransformer(info, "default", tmp_path)
    training = adapter.fit_transform(X, y)
    assert (training == -123).all().all()
    assert list(training) == [info["rename"][n] for n in info["sets"]["default"]]
    class Generator:
        def transform(self, X, **kwargs):
            return pd.DataFrame({"pca": X.a + 1, "multiply": X.a * X.b})
    adapter.generator = Generator()
    restored = FeatureDiscoveryTransformer.from_json(json.loads(json.dumps(adapter.to_json())), tmp_path)
    restored.generator = Generator()
    assert_frame_equal(adapter.transform(X), restored.transform(X))
    assert list(adapter.transform(X)) == list(training)
    (tmp_path / "generator.joblib").write_bytes(b"different fitted PCA")
    changed = FeatureDiscoveryTransformer.from_json(adapter.to_json(), tmp_path)
    with pytest.raises(AutoMLException, match="identity mismatch"):
        changed.fit_transform(X, y)


def test_selection_requires_matching_generator_and_original_identities(tmp_path):
    info, X, y = manifest(tmp_path)
    selection = dict(generator_id=info["generator_id"], dropped_input_identities=[input_identity("a")])
    params = dict(feature_discovery=dict(folds={"0:0": info}, feature_set="default"), drop_features=["a"], feature_selection_manifest=selection)
    prep = Preprocessing(params, k_fold=0, results_path=tmp_path)
    prep.fit_and_transform(X, y)
    saved = prep.to_json()
    saved["feature_selection_manifest"]["generator_id"] = "another-fit"
    with pytest.raises(AutoMLException, match="generator mismatch"):
        Preprocessing().from_json(saved, tmp_path)
    wrong = copy.deepcopy(params)
    wrong["feature_selection_manifest"] = dict(generator_id="wrong")
    with pytest.raises(AutoMLException, match="generator mismatch"):
        Preprocessing(wrong, k_fold=0, results_path=tmp_path).fit_and_transform(X, y)


@pytest.mark.parametrize("change", ["rename", "definition", "generator", "profile"])
def test_corrupted_identity_mapping_is_rejected(tmp_path, change):
    info, _, _ = manifest(tmp_path)
    if change == "rename":
        info["rename"]["multiply"] = info["rename"]["pca"]
    elif change == "definition":
        info["feature_identities"]["multiply"]["definition"]["operation"] = "div"
    elif change == "generator":
        info["generator_id"] = "another-fit"
    else:
        info["sets"]["default"].append("unknown")
    with pytest.raises(AutoMLException):
        FeatureDiscoveryTransformer.from_json(dict(info=info, feature_set="default"), tmp_path)


@pytest.mark.parametrize("name", ["feature_discovery_0", "feature_discovery_2_", "feature_discovery_" + "f" * 64])
def test_shared_generated_selection_is_blocked_in_new_training(name):
    with pytest.raises(AutoMLException, match="generated features"):
        reject_generated_drops(dict(drop_features=[name]))


def test_legacy_mapping_and_drops_replay_unchanged(tmp_path):
    info, X, y = manifest(tmp_path)
    for key in ["identity_schema_version", "generator_id", "generator_sha256", "fit_scope", "feature_identities"]:
        info.pop(key)
    info["rename"] = {"multiply": "feature_discovery_0", "pca": "feature_discovery_1"}
    saved = dict(feature_discovery=dict(info=info, feature_set="default"), drop_features=["feature_discovery_0"])
    prep = Preprocessing();prep.from_json(saved, tmp_path)
    class Generator:
        def transform(self, X, **kwargs):return pd.DataFrame({"multiply": X.a * X.b, "pca": X.a + 1})
    prep._feature_discovery.generator = Generator()
    transformed, _, _ = prep.transform(X, None)
    assert "feature_discovery_0" not in transformed
    np.testing.assert_array_equal(transformed.feature_discovery_1, X.a + 1)


@pytest.mark.parametrize("suffix", ["_importance.csv", "_shap_importance.csv"])
def test_importance_does_not_merge_local_slots_or_zero_fill(tmp_path, suffix):
    root = tmp_path / "model";root.mkdir()
    (tmp_path / "data_info.json").write_text(json.dumps({"columns": ["a", "b"]}))
    info, _, _ = manifest(root)
    legacy0 = dict(rename={"a_mul_b": "feature_discovery_0"})
    legacy1 = dict(rename={"PCA component 2": "feature_discovery_0"})
    desc = dict(learners=[{"name": "learner_fold_0"}, {"name": "learner_fold_1"}], preprocessing=[{"feature_discovery": {"info": legacy0}}, {"feature_discovery": {"info": legacy1}}])
    (root / "framework.json").write_text(json.dumps(desc))
    pd.DataFrame({"importance": [1., -.2, 0.]}, index=["a", "feature_discovery_0", "random_feature"]).to_csv(root / ("learner_fold_0" + suffix))
    pd.DataFrame({"importance": [2., -.1, 0., .1]}, index=["a", "feature_discovery_0", "random_feature", "b"]).to_csv(root / ("learner_fold_1" + suffix))
    frame, selectable = importance_matrix(root, 2, 1, suffix)
    assert len(frame) == 5 and selectable == {"a", "b", "random_feature"}
    generated = frame.loc[~frame.index.isin(selectable)]
    assert generated.notna().sum(axis=1).tolist() == [1, 1]
    assert pd.isna(frame.loc["b", "Learner 1"])
    if "shap" not in suffix:
        AdditionalMetrics.add_permutation_importance(io.StringIO(), str(root), 2, 1)
        assert json.loads((tmp_path / "drop_features.json").read_text()) == ["random_feature"]


def test_reports_use_cv_manifest_after_tfm_full_refit(tmp_path):
    root = tmp_path / "model";root.mkdir()
    first, _, _ = manifest(root, "fold0")
    second, _, _ = manifest(root, "fold1")
    full, _, _ = manifest(root, "full")
    validation = dict(learners=[{"name": "learner_fold_0"}, {"name": "learner_fold_1"}], preprocessing=[{"feature_discovery": {"info": first}}, {"feature_discovery": {"info": second}}])
    desc = dict(learners=[{"name": "learner_full"}], preprocessing=[{"feature_discovery": {"info": full}}], validation_models=validation)
    (root / "framework.json").write_text(json.dumps(desc))
    for k, info in enumerate([first, second]):
        pd.DataFrame({"importance": [1., 2.]}, index=list(info["rename"].values())).to_csv(root / f"learner_fold_{k}_importance.csv")
    frame, selectable = importance_matrix(root, 2, 1, "_importance.csv")
    assert len(frame) == 3 and not selectable
    assert sorted(frame.notna().sum(axis=1)) == [1, 1, 2]


def test_tuner_ignores_legacy_generated_drop_file(tmp_path):
    from supervised.tuner.mljar_tuner import MljarTuner
    tuner = MljarTuner.__new__(MljarTuner)
    tuner._data_info = {"columns_info": {"a": {}, "b": {}}}
    (tmp_path / "drop_features.json").write_text(json.dumps(["random_feature", "feature_discovery_0", "feature_discovery_abc", "a_mul_b"]))
    assert tuner.get_features_selection_params([], str(tmp_path), 60) is None
