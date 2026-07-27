import matplotlib.pyplot as plt
import pandas as pd

from supervised.utils.automl_plots import AutoMLPlots


class MockModel:
    def __init__(self, name, predictions):
        self._name = name
        self._predictions = predictions

    def get_name(self):
        return self._name

    def get_out_of_folds(self):
        return pd.DataFrame({"prediction": self._predictions})


def test_plot_feature_heatmap(tmp_path):
    plot_path = tmp_path / AutoMLPlots.features_heatmap_fname
    data = pd.DataFrame(
        {"model_1": [0.2, 0.8], "model_2": [0.4, 0.6]},
        index=["feature_1", "feature_2"],
    )

    AutoMLPlots._plot_feature_heatmap(
        data_df=data,
        title="Feature Importance",
        plot_path=plot_path,
    )

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert plt.get_fignums() == []


def test_models_correlation(tmp_path):
    models = [
        MockModel("model_1", [0.1, 0.4, 0.2, 0.8]),
        MockModel("model_2", [0.2, 0.3, 0.5, 0.7]),
    ]

    AutoMLPlots.models_correlation(tmp_path, models)

    plot_path = tmp_path / AutoMLPlots.correlation_heatmap_fname
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert plt.get_fignums() == []
