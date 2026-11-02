"""Feature meaning is separate from a fitted pipeline's storage column names."""
import hashlib
import json
import re
from numbers import Real

from supervised.exceptions import AutoMLException

SCHEMA_VERSION = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def input_identity(name):
    return "input:" + digest({"type": type(name).__name__, "name": name})


def generated_column(name):
    # Includes old positional slots and new SHA-256 storage columns.
    return bool(re.match(r"^feature_discovery_", str(name)))


def build_identities(names, definitions, generator_id, input_columns, package_version):
    """Only audited stateless definitions can be shared between fitted generators.

    Goldmine 1.1 executes numeric formulas directly on raw inputs, with an
    optional learned output fill. All other definitions conservatively retain
    the identity of their serialized fitted generator (including PCA/rules/TE).
    Missing/incomplete upstream metadata is never inferred from a display name.
    """
    result, seen, storage_seen = {}, {}, set(input_columns)
    for name in names:
        desc = definitions.get(name, {})
        op, inputs = desc.get("formula_name"), desc.get("source_columns", [])
        stateless = (desc.get("feature_type") == "numeric_formula"
                     and op in {"mul", "div", "sub", "absdiff"} and len(inputs) == 2
                     and all(c in input_columns for c in inputs)
                     and "output_fill" in desc and desc["output_fill"] is None)
        if stateless:
            operands = [input_identity(c) for c in inputs]
            if op in {"mul", "absdiff"}:
                operands.sort()
            definition = {"kind": "stateless", "operation": op, "inputs": operands,
                          "parameters": {"denominator_epsilon": 1e-9} if op == "div" else {},
                          "implementation": "features-goldmine:" + package_version}
        else:
            definition = {"kind": "fitted", "generator_id": generator_id,
                          "component": str(name), "implementation": "features-goldmine:" + package_version,
                          "feature_type": desc.get("feature_type", "unknown"),
                          "operation": op, "inputs": [input_identity(c) for c in inputs]}
            if desc.get("feature_type") == "binary_rule":
                conditions = desc.get("metadata", {}).get("conditions", [])
                if all(isinstance(c, dict) and {"feature", "op", "threshold"} <= c.keys()
                       and isinstance(c["threshold"], Real) for c in conditions):
                    definition["conditions"] = [
                        {"input": input_identity(c["feature"]), "operation": c["op"],
                         "threshold": float(c["threshold"])} for c in conditions
                    ]
            # State is bound by the generator's content hash; no lossy JSON
            # conversion of numpy arrays or category-keyed target mappings.
        identity = "discovery:" + digest(definition)
        if identity in seen:
            if seen[identity] != canonical(definition):
                raise AutoMLException("Feature identity hash collision")
            raise AutoMLException("Duplicate equivalent discovery outputs")
        seen[identity] = canonical(definition)
        column = "feature_discovery_" + identity.split(":", 1)[1]
        if column in storage_seen:
            raise AutoMLException("Feature discovery storage column collision")
        storage_seen.add(column)
        result[name] = {"identity": identity, "storage_column": column,
                        "display_label": str(name), "definition": definition}
    return result


def validate_manifest(info):
    if "identity_schema_version" not in info:
        return  # Preserve old saved mappings verbatim; never infer equivalence.
    if info["identity_schema_version"] != SCHEMA_VERSION:
        raise AutoMLException("Unsupported feature identity schema")
    records = info["feature_identities"]
    if info["generator_id"] != digest({"fit_scope": info["fit_scope"], "state": info["generator_sha256"]}):
        raise AutoMLException("Feature identity fitted scope mismatch")
    if set(records) != set(info["rename"]):
        raise AutoMLException("Feature identity mapping does not match storage columns")
    ids, columns = set(), set(info["input_columns"])
    for name, record in records.items():
        definition = record["definition"]
        if record["identity"] != "discovery:" + digest(definition):
            raise AutoMLException("Feature identity definition mismatch")
        if definition["kind"] == "fitted" and definition["generator_id"] != info["generator_id"]:
            raise AutoMLException("Feature identity generator mismatch")
        column = record["storage_column"]
        if column != "feature_discovery_" + record["identity"].split(":", 1)[1]:
            raise AutoMLException("Feature identity storage mismatch")
        if info["rename"][name] != column or column in columns or record["identity"] in ids:
            raise AutoMLException("Feature identity collision or storage mapping mismatch")
        ids.add(record["identity"])
        columns.add(column)
    if any(name not in records for names in info["sets"].values() for name in names):
        raise AutoMLException("Feature profile references an unknown identity")


def reject_generated_drops(params, original_columns=None):
    """New training must not apply shared selection to generated columns."""
    drops = params.get("drop_features", [])
    generated = set()
    for info in params.get("feature_discovery", {}).get("folds", {}).values():
        generated.update(info.get("rename", {}).values())
    invalid = [c for c in drops if c in generated or generated_column(c)
               or (original_columns is not None and c != "random_feature" and c not in original_columns)]
    if invalid:
        raise AutoMLException("Shared selection of generated features is disabled: " + ", ".join(map(str, invalid)))
