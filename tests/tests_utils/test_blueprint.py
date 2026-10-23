import copy
import json
import xml.etree.ElementTree as ET

import pytest

from examples.blueprints.generate import examples, generate
from supervised.utils.blueprint import (
    FOOTER_TEXT,
    FOOTER_URL,
    build_blueprint,
    render_svg,
    save_blueprints,
)

SVG = "{http://www.w3.org/2000/svg}"


@pytest.mark.parametrize("name", list(examples()))
def test_examples_have_valid_graphs_and_clickable_footer(name):
    artifact = examples()[name]
    original = copy.deepcopy(artifact)
    graph = build_blueprint(artifact, ["LightGBM", "CatBoost"])
    svg = ET.fromstring(render_svg(graph))
    assert svg.tag == SVG + "svg"
    assert len(svg.findall(".//" + SVG + "a")) >= 1
    footer = [a for a in svg.findall(SVG + "a") if a.get("href") == FOOTER_URL]
    assert len(footer) == 1
    assert footer[0].find(SVG + "text").text == FOOTER_TEXT
    assert svg.find(SVG + "title").text.endswith(artifact["name"])
    ids = {n["id"] for n in graph["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in graph["edges"])
    assert artifact == original


def test_fold_grouping_preserves_different_encodings_and_learner_parameters():
    artifact = examples()["multiclass_cv"]
    graph = build_blueprint(artifact)
    groups = [n for n in graph["nodes"] if "learners" in n]
    assert [n["learners"] for n in groups] == [[0, 1, 2, 3], [4]]
    assert [e["weight"] for e in graph["edges"] if "weight" in e] == [0.8, 0.2]
    artifact["learners"][1]["params"]["max_depth"] = 12
    graph = build_blueprint(artifact)
    assert len([n for n in graph["nodes"] if "learners" in n]) == 3


def test_preprocessing_order_and_target_inverse_before_average():
    graph = build_blueprint(examples()["mixed_features"])
    labels = [n["label"] for n in graph["nodes"]]
    expected = [
        "Text TF-IDF",
        "IP features",
        "Fill missing values",
        "Fill new missing values",
        "Golden features",
        "K-means features",
        "Encode categories",
        "Datetime features",
        "Scale features",
        "Select features",
        "Clip numeric features",
        "CatBoost",
    ]
    assert [labels.index(label) for label in expected] == sorted(
        labels.index(label) for label in expected
    )
    graph = build_blueprint(examples()["regression"])
    nodes = {n["label"]: n["id"] for n in graph["nodes"]}
    assert {"source": nodes["Linear"], "target": nodes["Undo target scaling"]} in graph[
        "edges"
    ]
    assert {
        "source": nodes["Undo target scaling"],
        "target": nodes["Average learner predictions"],
    } in graph["edges"]


def test_ensemble_normalization_and_mixed_stacked_routing():
    graph = build_blueprint(examples()["binary_ensemble"])
    assert [n["weight"] for n in graph["nodes"] if "weight" in n] == [0.5, 1 / 3, 1 / 6]
    graph = build_blueprint(examples()["mixed_ensemble"], ["LightGBM", "CatBoost"])
    ids = {n["label"]: n["id"] for n in graph["nodes"]}
    assert (
        dict(source=ids["Input features"], target=ids["Regression_Linear"])
        in graph["edges"]
    )
    assert (
        dict(source=ids["Stacked features"], target=ids["Regression_Stacked"])
        in graph["edges"]
    )
    assert "out-of-fold" in " ".join(graph["notes"])


def test_xml_escaping_bounded_width_and_cycles():
    artifact = examples()["regression"]
    artifact["name"] = '<script>alert("example")</script> & model'
    graph = build_blueprint(artifact)
    svg = render_svg(graph)
    assert "<script>" not in svg
    assert ET.fromstring(svg).find(SVG + "title").text.endswith(artifact["name"])
    wide = copy.deepcopy(examples()["binary_ensemble"])
    wide["selected_models"] = [{"model": f"Model_{i}", "repeat": 1} for i in range(100)]
    assert int(ET.fromstring(render_svg(build_blueprint(wide))).get("width")) <= 1158
    graph["edges"].append(dict(source=graph["nodes"][-1]["id"], target="n0"))
    with pytest.raises(ValueError, match="cycle"):
        render_svg(graph)


def test_save_from_artifacts_idempotence_and_missing_metadata(tmp_path):
    name = "Example"
    directory = tmp_path / name
    directory.mkdir()
    artifact = examples()["regression"]
    artifact["name"] = name
    (directory / "framework.json").write_text(json.dumps(artifact))
    readme = directory / "README.md"
    readme.write_text("# Original report\n\nMetrics: unchanged.\n")
    save_blueprints(tmp_path, [name], {})
    first = readme.read_text()
    stamp = (directory / "blueprint.svg").stat().st_mtime_ns
    save_blueprints(tmp_path, [name], {})
    assert readme.read_text() == first
    assert (directory / "blueprint.svg").stat().st_mtime_ns == stamp
    assert first.count("## Model blueprint") == 1
    assert "![Model prediction pipeline](blueprint.svg)" in first
    assert FOOTER_URL not in first and "Metrics: unchanged." in first
    readme.write_text("# Rewritten report\n")
    save_blueprints(tmp_path, [name], {})
    assert "## Model blueprint" in readme.read_text()
    legacy = build_blueprint(
        {"name": "Legacy", "params": {"learner": {"ml_task": "regression"}}}
    )
    assert any(n["detail"] == "Fitted metadata unavailable" for n in legacy["nodes"])


def test_example_generation_reproducible(tmp_path):
    generate(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    generate(tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
