import os


class FairnessReport:
    """Saves information about fairness in the report."""

    @staticmethod
    def save_classification(fairness_metrics, fout, model_path, is_multi=False):

        for k, v in fairness_metrics.items():
            if k == "fairness_optimization":
                continue

            if is_multi:
                a = k.split("__", maxsplit=1)
                feature, class_name = a

            if is_multi:
                fout.write(f"\n\n## Fairness metrics for {feature} feature and {class_name} class\n\n")
            else:
                fout.write(f"\n\n## Fairness metrics for {k} feature\n\n")

            fout.write(v["metrics"].to_markdown())
            fout.write("\n\n")
            fout.write(v["stats"].to_markdown())
            fout.write("\n\n")

            if is_multi:
                fout.write(f"\n\n## Is model fair for {feature} feature and {class_name} class?\n")
            else:
                fout.write(f"\n\n## Is model fair for {k} feature?\n")
            fair_str = "fair" if v["is_fair"] else "unfair"
            fairness_threshold = fairness_metrics.get("fairness_optimization", {}).get(
                "fairness_threshold"
            )
            fairness_threshold_str = ""
            if fairness_threshold is not None:
                if "ratio" in v["fairness_metric_name"].lower():
                    fairness_threshold_str = (
                        f"It should be higher than {fairness_threshold}."
                    )
                else:
                    fairness_threshold_str = (
                        f"It should be lower than {fairness_threshold}."
                    )

            if is_multi:
                fout.write(f"Model is {fair_str} for {feature} feature and {class_name} class.\n")
            else:
                fout.write(f"Model is {fair_str} for {k} feature.\n")
            fout.write(
                f'The {v["fairness_metric_name"]} is {v["fairness_metric_value"]}. {fairness_threshold_str}\n'
            )
            if not v["is_fair"]:
                # display information about privileged and underprivileged groups
                # for unfair models
                if v.get("underprivileged_value") is not None:
                    fout.write(
                        f'Underprivileged value is {v["underprivileged_value"]}.\n'
                    )
                if v.get("privileged_value") is not None:
                    fout.write(f'Privileged value is {v["privileged_value"]}.\n')

            for figure in v["figures"]:
                fout.write(f"\n\n### {figure['title']}\n\n")
                figure["figure"].savefig(os.path.join(model_path, figure["fname"]))
                fout.write(f"\n![]({figure['fname']})\n\n")

    @staticmethod
    def regression(fairness_metrics, fout, model_path):

        for k, v in fairness_metrics.items():
            if k == "fairness_optimization":
                continue
            fout.write(f"\n\n## Fairness metrics for {k} feature\n\n")

            fout.write(v["metrics"].to_markdown())
            fout.write("\n\n")

            fout.write(f'Privileged value: {v["privileged_value"]}\n\n')
            fout.write(f'Underprivileged value: {v["underprivileged_value"]}\n\n\n')
            fout.write(f'Fairness metric: {v["fairness_metric_name"]}\n\n')
            fout.write(f'{v["metric_name"]} Difference: {v["diff"]}\n\n')
            fout.write(f'{v["metric_name"]} Ratio: {v["ratio"]}\n\n')

            for figure in v["figures"]:
                fout.write(f"\n\n### {figure['title']}\n\n")
                figure["figure"].savefig(os.path.join(model_path, figure["fname"]))
                fout.write(f"\n![]({figure['fname']})\n\n")
