"""Align importance only when fitted preprocessing establishes equivalence."""
import json
from pathlib import Path

import pandas as pd

from supervised.preprocessing.feature_identity import digest, generated_column, input_identity, validate_manifest
from supervised.utils.common import construct_learner_name


def importance_matrix(model_path, fold_cnt, repeat_cnt, suffix):
    root = Path(model_path)
    description = json.loads((root / "framework.json").read_text()) if (root / "framework.json").exists() else {}
    validation = description.get("validation_models", description)
    by_learner = {l["name"]: p for l, p in zip(validation.get("learners", []), validation.get("preprocessing", []))}
    inputs_file = root.parent / "data_info.json"
    original = set(json.loads(inputs_file.read_text()).get("columns", [])) if inputs_file.exists() else None
    frames, records, labels, generated = [], [], {}, set()
    for repeat in range(repeat_cnt):
        for fold in range(fold_cnt):
            learner = construct_learner_name(fold, repeat, repeat_cnt)
            path = root / (learner + suffix)
            if not path.exists():
                continue
            frame = pd.read_csv(path, index_col=0)
            title = f"Learner {fold+1}" + (f", Repeat {repeat+1}" if repeat_cnt > 1 else "")
            frame.columns = [title]
            info = by_learner.get(learner, {}).get("feature_discovery", {}).get("info", {})
            validate_manifest(info)
            mapping = {r["storage_column"]: r for r in info.get("feature_identities", {}).values()}
            legacy = {column: name for name, column in info.get("rename", {}).items()}
            indices = []
            for column, value in frame[title].items():
                record = mapping.get(column)
                if record:
                    identity, display = record["identity"], record["display_label"]
                    scoped = record["definition"]["kind"] == "fitted"
                elif column in legacy or generated_column(column) or (original is not None and column not in original and column != "random_feature"):
                    identity = "local:" + digest({"learner": learner, "column": column, "info": info})
                    display, scoped = str(legacy.get(column, column)), True
                else:
                    indices.append(column)
                    records.append(dict(learner=learner, storage_column=column, identity=input_identity(column), display_label=str(column), importance=value))
                    continue
                if identity not in labels:
                    label = f"{display} [{title if scoped else 'shared'}, {identity[-12:]}]"
                    if label in labels.values():
                        label = f"{display} [{identity}]"
                    labels[identity] = label
                label = labels[identity]
                indices.append(label)
                generated.add(label)
                records.append(dict(learner=learner, storage_column=column, identity=identity, display_label=display, importance=value))
            frame.index = indices
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), set()
    matrix = pd.concat(frames, axis=1)  # Missing features remain NaN, never zero.
    pd.DataFrame(records).to_csv(root / ("shap_importance_identities.csv" if "shap" in suffix else "permutation_importance_identities.csv"), index=False)
    selectable = set(matrix.index) - generated
    if original is not None:
        selectable &= original | {"random_feature"}
    return matrix, selectable
