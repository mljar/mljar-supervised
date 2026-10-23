"""Generate dependency-free examples: python -m examples.blueprints.generate."""

import copy
import json
from pathlib import Path

from supervised.utils.blueprint import build_blueprint, render_svg


def examples():
    preprocessing = {
        "missing_values": [
            {"fill_method": "fill_na_median", "fill_params": {"age": 42}}
        ],
        "categorical": [
            {"convert_method": "categorical_to_onehot", "columns": ["city"]}
        ],
        "scale": [{"scale_method": "scale_normal", "columns": ["age", "income"]}],
        "params": {"columns_preprocessing": {"age": ["missing_values", "scale"]}},
    }
    regression = dict(
        name="Regression_Linear",
        params={"learner": {"model_type": "Linear", "ml_task": "regression"}},
        preprocessing=[
            dict(preprocessing, scale_y={"scale_method": "scale_log_and_normal"})
        ],
        learners=[{"params": {"model_type": "Linear", "fit_intercept": True}}],
    )
    cv = dict(
        name="Multiclass_RandomForest_CV",
        params={"learner": {"ml_task": "multiclass_classification"}},
        preprocessing=[preprocessing] * 5,
        learners=[
            {"params": {"model_type": "Random Forest", "max_depth": 6, "seed": i}}
            for i in range(5)
        ],
    )
    # A fold with a genuinely different encoding must remain a separate branch.
    cv["preprocessing"] = [copy.deepcopy(p) for p in cv["preprocessing"]]
    cv["preprocessing"][4]["categorical"][0]["convert_method"] = "categorical_to_int"
    ensemble = dict(
        name="Binary_Ensemble",
        ml_task="binary_classification",
        selected_models=[
            {"model": "LightGBM", "repeat": 3},
            {"model": "CatBoost", "repeat": 2},
            {"model": "NeuralNetwork", "repeat": 1},
        ],
    )
    stacked = dict(
        name="Regression_Stacked",
        params={"learner": {"ml_task": "regression"}},
        is_stacked=True,
        preprocessing=[preprocessing],
        learners=[{"params": {"model_type": "Xgboost", "max_depth": 3}}],
    )
    mixed = dict(
        name="Binary_Text_IP_Datetime",
        params={"learner": {"ml_task": "binary_classification"}},
        preprocessing=[
            dict(
                preprocessing,
                text_transforms=[
                    {"old_column": "description", "new_columns": ["tfidf_0", "tfidf_1"]}
                ],
                ip_transforms=[
                    {
                        "old_column": "client_ip",
                        "new_columns": ["ip_version", "ip_private"],
                    }
                ],
                datetime_transforms=[
                    {
                        "old_column": "timestamp",
                        "transforms": ["year", "month", "weekday"],
                    }
                ],
                golden_features={"new_columns": ["age_ratio_income"]},
                kmeans={"new_features": ["cluster_0", "cluster_1"]},
                drop_features=["unused_feature"],
            )
        ],
        learners=[{"params": {"model_type": "CatBoost", "depth": 4}}],
    )
    mixed_ensemble = dict(
        name="Mixed_Stacked_Ensemble",
        ml_task="regression",
        is_stacked=True,
        selected_models=[
            {"model": "Regression_Linear", "repeat": 2, "is_stacked": False},
            {"model": "Regression_Stacked", "repeat": 3, "is_stacked": True},
        ],
    )
    return dict(
        regression=regression,
        multiclass_cv=cv,
        binary_ensemble=ensemble,
        stacked_regression=stacked,
        mixed_features=mixed,
        mixed_ensemble=mixed_ensemble,
    )


def generate(output=None):
    output = Path(output or Path(__file__).parent)
    output.mkdir(parents=True, exist_ok=True)
    gallery = [
        "# Model blueprint examples",
        "",
        "Set `explain_level=2` to embed fitted model blueprints in AutoML model READMEs.",
        "",
        "These six diagrams use illustrative fitted metadata. Integration tests also train real models, including ensembles and stacking.",
        "",
        "Every SVG includes a clickable ‘Trained with MLJAR AutoML’ footer in standalone SVGs and inline HTML reports.",
        "",
        "Regenerate with `venv/bin/python -m examples.blueprints.generate`.",
        "",
    ]
    for name, artifact in examples().items():
        graph = build_blueprint(artifact, ["LightGBM", "CatBoost"])
        (output / f"{name}.json").write_text(
            json.dumps(graph, indent=2) + "\n", encoding="utf-8"
        )
        (output / f"{name}.svg").write_text(render_svg(graph), encoding="utf-8")
        gallery.extend(
            [
                f"## {name.replace('_', ' ').title()}",
                "",
                f"![{artifact['name']}]({name}.svg)",
                "",
            ]
        )
    (output / "README.md").write_text("\n".join(gallery), encoding="utf-8")
    return output


if __name__ == "__main__":
    print(generate())
