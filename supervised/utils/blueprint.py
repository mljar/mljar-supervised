"""Dependency-free prediction blueprints built from saved model metadata."""

import hashlib
import json
import math
from pathlib import Path
import textwrap
from urllib.parse import quote
import xml.etree.ElementTree as ET

FOOTER_URL = "https://mljar.com"
FOOTER_TEXT = "Trained with MLJAR AutoML"
_START = "<!-- mljar-blueprint:start -->"
_END = "<!-- mljar-blueprint:end -->"


def _columns_summary(columns):
    columns = [str(c) for c in columns]
    preview = ", ".join(columns[:3])
    if len(columns) > 3:
        preview += ", …"
    return f"{len(columns)} columns" + (f": {preview}" if preview else "")


def _steps(preprocessing):
    """Match Preprocessing.transform order, using compact fitted summaries."""
    result = []

    def add(title, columns=(), method=None):
        columns = list(columns)
        detail = str(method) if method else _columns_summary(columns)
        if method and columns:
            detail += "; " + _columns_summary(columns)
        result.append(
            dict(
                label=title,
                columns=list(map(str, columns)),
                detail=detail,
                kind="transform",
            )
        )

    p = preprocessing
    if p.get("remove_columns"):
        add("Remove unused columns", p["remove_columns"])
    for field, title in [
        ("text_transforms", "Text TF-IDF"),
        ("ip_transforms", "IP features"),
    ]:
        for transform in p.get(field, []):
            add(
                title,
                [transform.get("old_column", "unknown")],
                f"{len(transform.get('new_columns', []))} generated features",
            )
    for transform in p.get("missing_values", []):
        add(
            "Fill missing values",
            transform.get("fill_params", {}).keys(),
            transform.get("fill_method", "unknown method"),
        )
    if p.get("params", {}).get("columns_preprocessing"):
        add("Fill new missing values", method="Conditional fallback at prediction")
    if p.get("golden_features") is not None:
        add("Golden features", p["golden_features"].get("new_columns", []))
    if p.get("kmeans") is not None:
        add("K-means features", p["kmeans"].get("new_features", []))
    for transform in p.get("categorical", []):
        add(
            "Encode categories",
            transform.get("columns", []),
            transform.get("convert_method", "unknown method"),
        )
    for transform in p.get("datetime_transforms", []):
        add(
            "Datetime features",
            [transform.get("old_column", "unknown")],
            ", ".join(transform.get("transforms", [])),
        )
    for transform in p.get("scale", []):
        add(
            "Scale features",
            transform.get("columns", []),
            transform.get("scale_method", "unknown method"),
        )
    if p.get("add_random_feature"):
        add("Add random feature", method="Random values generated at prediction")
    if p.get("drop_features"):
        add("Select features", p["drop_features"], "Drop selected columns")
    add("Clip numeric features", method="Float32-safe numeric bounds")
    return result


def build_blueprint(artifact, stacked_models=()):
    """Build a versioned graph without loading estimators or training data."""
    name = artifact.get("name", "Model")
    task = artifact.get("ml_task") or artifact.get("params", {}).get("learner", {}).get(
        "ml_task", "unknown"
    )
    graph = dict(
        schema_version=1,
        name=name,
        task=task,
        view="prediction",
        nodes=[],
        edges=[],
        footer_url=FOOTER_URL,
        notes=["Generated from fitted model metadata."],
    )

    def node(label, kind="transform", detail="", **extra):
        identifier = f"n{len(graph['nodes'])}"
        graph["nodes"].append(
            dict(id=identifier, label=label, kind=kind, detail=detail, **extra)
        )
        return identifier

    def edge(source, target, **extra):
        graph["edges"].append(dict(source=source, target=target, **extra))

    raw = node("Input features", "input")
    stacked_input = None

    def stacking_input():
        nonlocal stacked_input
        if stacked_input is not None:
            return stacked_input
        predictions = []
        for base in stacked_models or ():
            ref = node(
                str(base),
                "model",
                "Base model prediction",
                href=f"../{quote(str(base), safe='')}/README.md",
            )
            edge(raw, ref)
            predictions.append(ref)
        if not predictions:
            ref = node(
                "Base model predictions", "model", "Dependency metadata unavailable"
            )
            edge(raw, ref)
            predictions.append(ref)
        stacked_input = node(
            "Stacked features", "aggregate", "Original features + predictions"
        )
        edge(raw, stacked_input)
        for ref in predictions:
            edge(ref, stacked_input)
        graph["notes"].append(
            "Stacking uses out-of-fold predictions during training and fitted base-model predictions at inference."
        )
        return stacked_input

    selected = artifact.get("selected_models")
    if selected is not None:
        total = sum(float(member["repeat"]) for member in selected)
        if not selected or not math.isfinite(total) or total <= 0:
            raise ValueError("Ensemble weights must have a positive finite total")
        aggregation = node(
            "Weighted average", "aggregate", "Combine member predictions"
        )
        for member in selected:
            repeat = float(member["repeat"])
            if not math.isfinite(repeat) or repeat <= 0:
                raise ValueError("Ensemble weights must be positive and finite")
            member_name = str(member["model"])
            ref = node(
                member_name,
                "model",
                f"Weight: {repeat / total:.3%}",
                weight=repeat / total,
                href=f"../{quote(member_name, safe='')}/README.md",
            )
            # Metadata supplied by the caller distinguishes mixed ensembles.
            is_stacked = member.get("is_stacked", artifact.get("is_stacked", False))
            edge(stacking_input() if is_stacked else raw, ref)
            edge(ref, aggregation)
    else:
        source = stacking_input() if artifact.get("is_stacked") else raw
        preprocessings = artifact.get("preprocessing", [])
        learners = artifact.get("learners", [])
        count = max(len(preprocessings), len(learners), 1)
        groups = {}
        for i in range(count):
            p = preprocessings[i] if i < len(preprocessings) else None
            learner = (
                learners[i]
                if i < len(learners)
                else artifact.get("params", {}).get("learner", {})
            )
            learner = learner.get("params", learner)
            stages = (
                _steps(p)
                if p is not None
                else [
                    dict(
                        label="Preprocessing",
                        kind="transform",
                        detail="Fitted metadata unavailable",
                        columns=[],
                    )
                ]
            )
            settings = {
                k: v
                for k, v in learner.items()
                if k not in ["seed", "name", "explain_level", "ml_task"]
                and (
                    isinstance(v, (str, int, float, bool, type(None)))
                    or (
                        isinstance(v, (list, tuple))
                        and len(v) <= 16
                        and all(isinstance(x, (str, int, float, bool)) for x in v)
                    )
                )
            }
            target = (p.get("scale_y") or {}).get("scale_method") if p else None
            key = json.dumps([stages, settings, target], sort_keys=True)
            groups.setdefault(
                key, dict(stages=stages, settings=settings, target=target, members=[])
            )["members"].append(i)
        ends = []
        for group in groups.values():
            members = group["members"]
            previous = node(
                "Fitted pipeline",
                "model",
                f"{len(members)} learner(s); separate fitted states",
                learners=members,
            )
            edge(source, previous)
            for stage in group["stages"]:
                current = node(**stage)
                edge(previous, current)
                previous = current
            settings = group["settings"]
            algorithm = settings.get("model_type", "Unknown learner")
            preview = "; ".join(
                f"{k}={v}" for k, v in settings.items() if k != "model_type"
            )
            current = node(algorithm, "learner", preview[:160], parameters=settings)
            edge(previous, current)
            if group["target"]:
                inverse = node("Undo target scaling", detail=group["target"])
                edge(current, inverse)
                current = inverse
            ends.append((current, len(members)))
        aggregation = node(
            "Average learner predictions",
            "aggregate",
            f"Equal weight across {count} fitted learner(s)",
        )
        for current, size in ends:
            if len(ends) > 1:
                edge(current, aggregation, weight=size / count, label=f"{size}/{count}")
            else:
                edge(current, aggregation)
        if count > 1:
            graph["notes"].append(
                "Matching pipeline structures are grouped; each fold/repeat retains its own fitted preprocessing and learner. Group size determines its contribution to the average."
            )
    output = node(
        "Prediction" if task == "regression" else "Class probabilities / labels",
        "output",
        "Target decoding" if task != "regression" else "Original target units",
    )
    edge(aggregation, output)
    return graph


def render_svg(graph):
    """Render a bounded-width layered graph using only the standard library."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    ranks = {key: 0 for key in nodes}
    for _ in range(len(nodes)):
        changed = False
        for e in graph["edges"]:
            value = ranks[e["source"]] + 1
            if ranks[e["target"]] < value:
                ranks[e["target"]] = value
                changed = True
        if not changed:
            break
    else:
        raise ValueError("Blueprint contains a cycle")
    levels = {}
    for identifier, rank in ranks.items():
        levels.setdefault(rank, []).append(identifier)
    columns = min(4, max(map(len, levels.values())))
    width = 56 + columns * 250 + (columns - 1) * 34
    positions = {}
    y = 90
    for rank in sorted(levels):
        ids = levels[rank]
        for i, identifier in enumerate(ids):
            row, col = divmod(i, 4)
            row_count = min(4, len(ids) - row * 4)
            left = (width - (row_count * 250 + (row_count - 1) * 34)) / 2
            positions[identifier] = (left + col * 284, y + row * 120)
        y += math.ceil(len(ids) / 4) * 120 + 24
    height = y + 55
    prefix = "bp" + hashlib.sha256(graph["name"].encode()).hexdigest()[:12]
    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        viewBox=f"0 0 {width} {height}",
        width=str(width),
        height=str(height),
        role="img",
        attrib={
            "aria-labelledby": prefix + "title",
            "style": "max-width:100%;height:auto",
        },
    )
    ET.SubElement(svg, "title", id=prefix + "title").text = (
        f"Prediction blueprint: {graph['name']}"
    )
    ET.SubElement(svg, "desc").text = " ".join(graph["notes"])
    defs = ET.SubElement(svg, "defs")
    marker = ET.SubElement(
        defs,
        "marker",
        id=prefix + "arrow",
        markerWidth="8",
        markerHeight="8",
        refX="7",
        refY="4",
        orient="auto",
    )
    ET.SubElement(marker, "path", d="M0,0 L8,4 L0,8 Z", fill="#64748b")
    ET.SubElement(svg, "rect", width="100%", height="100%", rx="12", fill="#f8fafc")

    def text(parent, x, y, value, size="13", color="#334155", weight="normal"):
        ET.SubElement(
            parent,
            "text",
            x=str(x),
            y=str(y),
            fill=color,
            attrib={
                "font-family": "Arial, sans-serif",
                "font-size": size,
                "font-weight": weight,
                "text-anchor": "middle",
            },
        ).text = value

    text(
        svg,
        width / 2,
        32,
        textwrap.shorten(graph["name"], 65, placeholder="…"),
        "20",
        "#0f172a",
        "bold",
    )
    text(svg, width / 2, 57, f"Prediction pipeline · {graph['task']}")
    for edge_index, e in enumerate(graph["edges"]):
        sx, sy = positions[e["source"]]
        tx, ty = positions[e["target"]]
        sx += 125
        tx += 125
        sy += 94
        middle = (sy + ty) / 2
        if ty - sy > 85:
            # Long edges travel outside the node grid, avoiding intermediate
            # boxes that would otherwise appear to be accidental dependencies.
            gutter = 8 + (edge_index % 4) * 4
            path = f"M {sx} {sy} V {sy + 10} H {gutter} V {ty - 12} H {tx} V {ty - 4}"
        else:
            path = f"M {sx} {sy} V {middle} H {tx} V {ty - 4}"
        ET.SubElement(
            svg,
            "path",
            d=path,
            fill="none",
            stroke="#64748b",
            attrib={"stroke-width": "1.5", "marker-end": f"url(#{prefix}arrow)"},
        )
        if e.get("label"):
            text(svg, (sx + tx) / 2, middle - 5, e["label"], "11")
    colors = {
        "input": "#dbeafe",
        "output": "#dcfce7",
        "learner": "#e0e7ff",
        "aggregate": "#fef3c7",
        "model": "#e0f2fe",
    }
    for identifier, n in nodes.items():
        x, y_node = positions[identifier]
        parent = ET.SubElement(svg, "a", href=n["href"]) if n.get("href") else svg
        ET.SubElement(
            parent,
            "rect",
            x=str(x),
            y=str(y_node),
            width="250",
            height="94",
            rx="10",
            fill=colors.get(n["kind"], "#ffffff"),
            stroke="#cbd5e1",
        )
        ET.SubElement(parent, "title").text = n["label"] + ": " + n["detail"]
        text(
            parent,
            x + 125,
            y_node + 23,
            textwrap.shorten(n["label"], 30, placeholder="…"),
            weight="bold",
        )
        for i, line in enumerate(textwrap.wrap(n["detail"], width=34)[:3]):
            text(parent, x + 125, y_node + 43 + i * 16, line, "12")
    ET.SubElement(
        svg,
        "line",
        x1="28",
        x2=str(width - 28),
        y1=str(height - 42),
        y2=str(height - 42),
        stroke="#cbd5e1",
    )
    footer = ET.SubElement(
        svg, "a", href=FOOTER_URL, target="_blank", rel="noopener noreferrer"
    )
    text(footer, width / 2, height - 18, FOOTER_TEXT, "13", "#0369a1", "bold")
    return ET.tostring(svg, encoding="unicode") + "\n"


def save_blueprints(results_path, model_names, run_params):
    """Save and embed all available solutions, including ensembles and stacking."""
    root = Path(results_path)
    saved = []
    for name in model_names:
        directory = root / name
        if directory.resolve().parent != root.resolve():
            raise ValueError("Model directory must be directly inside results_path")
        artifact_path = directory / "ensemble.json"
        if not artifact_path.exists():
            artifact_path = directory / "framework.json"
        if not artifact_path.exists() or not (directory / "README.md").exists():
            continue
        cache_key = dict(
            mtime=artifact_path.stat().st_mtime_ns,
            stacked=run_params.get("stacked", []),
            footer=FOOTER_TEXT,
        )
        graph_path = directory / "blueprint.json"
        graph = (
            json.loads(graph_path.read_text(encoding="utf-8"))
            if graph_path.exists()
            else None
        )
        if (
            graph is None
            or graph.get("source_key") != cache_key
            or not (directory / "blueprint.svg").exists()
        ):
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            for member in artifact.get("selected_models", []):
                framework = root / member["model"] / "framework.json"
                ensemble = root / member["model"] / "ensemble.json"
                source = framework if framework.exists() else ensemble
                if source.exists():
                    member["is_stacked"] = json.loads(
                        source.read_text(encoding="utf-8")
                    ).get("is_stacked", False)
            graph = build_blueprint(artifact, run_params.get("stacked", []))
            graph["source_key"] = cache_key
            graph_path.write_text(
                json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            (directory / "blueprint.svg").write_text(
                render_svg(graph), encoding="utf-8"
            )
        readme = directory / "README.md"
        content = readme.read_text(encoding="utf-8")
        section = (
            _START
            + "\n## Model blueprint\n\n![Model prediction pipeline](blueprint.svg)\n\n"
            + " ".join(graph["notes"])
            + "\n"
            + _END
        )
        if _START in content and _END in content:
            before, rest = content.split(_START, 1)
            _, after = rest.split(_END, 1)
            updated = before + section + after
        else:
            updated = content.rstrip() + "\n\n" + section + "\n"
        if updated != content:
            readme.write_text(updated, encoding="utf-8")
        saved.append(name)
    return saved
