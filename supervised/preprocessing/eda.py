import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from wordcloud import STOPWORDS
from collections import defaultdict

from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.registry import REGRESSION
from supervised.exceptions import AutoMLException
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils

BLUE = "#007cf2"

class EDA:
    @staticmethod
    def compute(X_train, y_train, eda_path):

        inform = defaultdict(list)

        if isinstance(y_train, pd.Series):

            if PreprocessingUtils.get_type(y_train) in ("categorical"):

                plt.figure(figsize=(5, 5))
                sns.countplot(y_train, color=BLUE)
                plt.title("Target class distribution")
                plt.tight_layout(pad=2.0)
                plot_path = os.path.join(eda_path, "target.png")
                plt.savefig(plot_path)
                plt.close("all")

            else:

                plt.figure(figsize=(5, 5))
                sns.distplot(y_train, color=BLUE)
                plt.title("Target class distribution")
                plt.tight_layout(pad=2.0)
                plot_path = os.path.join(eda_path, "target.png")
                plt.savefig(plot_path)
                plt.close("all")

            inform["missing"].append(pd.isnull(y_train).sum() / y_train.shape[0])
            inform["unique"].append(y_train.nunique())
            inform["feature_type"].append(PreprocessingUtils.get_type(y_train))
            inform["plot"].append("![](target.png)")
            inform["feature"].append("target")
            inform["desc"].append(y_train.describe().to_dict())

        for col in X_train.columns:

            inform["feature_type"].append(PreprocessingUtils.get_type(X_train[col]))

            if PreprocessingUtils.get_type(X_train[col]) in ("categorical", "discrete"):

                plt.figure(figsize=(5, 5))
                chart = sns.countplot(
                    X_train[col],
                    order=X_train[col].value_counts().iloc[:10].index,
                    color=BLUE,
                )
                chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
                plt.title(f"{col} class distribution")
                plt.tight_layout(pad=2.0)
                plot_path = os.path.join(eda_path, f"{col}.png")
                plt.savefig(plot_path)
                plt.close("all")

            elif PreprocessingUtils.get_type(X_train[col]) in ("CONTINUOUS"):

                plt.figure(figsize=(5, 5))
                sns.distplot(X_train[col], color=BLUE)
                plt.title(f"{col} value distribution")
                plt.tight_layout(pad=2.0)
                plot_path = os.path.join(eda_path, f"{col}.png")
                plt.savefig(plot_path)
                plt.close("all")

            elif PreprocessingUtils.get_type(X_train[col]) in ("text"):

                plt.figure(figsize=(10, 10), dpi=70)
                word_string = " ".join(X_train[col].str.lower())
                wordcloud = WordCloud(
                    width=500,
                    height=500,
                    stopwords=STOPWORDS,
                    background_color="white",
                    max_words=400,
                    max_font_size=None,
                ).generate(word_string)

                plt.imshow(wordcloud, aspect="auto", interpolation="nearest")
                plt.axis("off")
                plot_path = os.path.join(eda_path, f"{col}.png")
                plt.savefig(plot_path)

            elif PreprocessingUtils.get_type(X_train[col]) in ("datetime"):

                plt.figure(figsize=(5, 5))
                pd.to_datetime(X_train[col]).plot(grid="True", color=BLUE)
                plt.tight_layout(pad=2.0)
                plot_path = os.path.join(eda_path, f"{col}.png")
                plt.savefig(plot_path)
                plt.close("all")

            inform["missing"].append(
                pd.isnull(X_train[col]).sum() * 100 / X_train.shape[0]
            )

            inform["unique"].append(int(X_train[col].nunique()))
            inform["plot"].append(f"![]({col}.png)")
            inform["feature"].append(str(col))
            inform["desc"].append(X_train[col].describe().to_dict())

        df = pd.DataFrame(inform)

        with open(os.path.join(eda_path, "README.md"), "w") as fout:

            for i, row in df.iterrows():

                fout.write(f"## Feature : {row['feature']}\n")
                fout.write(f"- **Feature type** : {row['feature_type']}\n")
                fout.write(f"- **Missing** : {row['missing']}%\n")
                fout.write(f"- **Unique** : {row['unique']}\n")

                for key in row["desc"].keys():

                    if key in ("25%", "50%", "75%"):

                        fout.write(
                            f"- **{key.capitalize()}th Percentile** : {row['desc'][key]}\n"
                        )
                    else:

                        fout.write(f"- **{key.capitalize()}** :{row['desc'][key]}\n")

                fout.write(f"- {row['plot']}\n")

        fout.close()
