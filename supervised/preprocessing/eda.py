import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from wordcloud import STOPWORDS
from collections import defaultdict

from supervised.utils.config import LOG_LEVEL
from supervised.exceptions import AutoMLException
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# color used in the plots
BLUE = "#007cf2"
COLS = 14


class EDA:
    @staticmethod
    def compute(X, y, eda_path):

        # Check for empty dataframes in params
        if X.empty:
            raise ValueError("X is empty")
        if y.empty:
            raise ValueError("y is empty")
        try:
            # check if exists
            if os.path.exists(eda_path):
                # probably the EDA analysis is already done
                # skip from here
                return
            else:
                # need to create directory for EDA analysis
                os.mkdir(eda_path)

            inform = defaultdict(list)

            if isinstance(y, pd.Series):

                if PreprocessingUtils.get_type(y) in ("categorical"):

                    plt.figure(figsize=(5, 5))
                    sns.countplot(y, color=BLUE)
                    plt.title("Target class distribution")
                    plt.tight_layout(pad=2.0)
                    plot_path = os.path.join(eda_path, "target.png")
                    plt.savefig(plot_path)
                    plt.close("all")

                else:

                    plt.figure(figsize=(5, 5))
                    sns.distplot(y, color=BLUE)
                    plt.title("Target class distribution")
                    plt.tight_layout(pad=2.0)
                    plot_path = os.path.join(eda_path, "target.png")
                    plt.savefig(plot_path)
                    plt.close("all")

                inform["missing"].append(pd.isnull(y).sum() / y.shape[0])
                inform["unique"].append(y.nunique())
                inform["feature_type"].append(PreprocessingUtils.get_type(y))
                inform["plot"].append("![](target.png)")
                inform["feature"].append("target")
                inform["desc"].append(y.describe().to_dict())
            for col in X.columns:
                inform["feature_type"].append(PreprocessingUtils.get_type(X[col]))

                if PreprocessingUtils.get_type(X[col]) in ("categorical", "discrete",):

                    plt.figure(figsize=(5, 5))
                    chart = sns.countplot(
                        X[col], order=X[col].value_counts().iloc[:10].index, color=BLUE,
                    )
                    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
                    plt.title(f"{col} class distribution")
                    plt.tight_layout(pad=2.0)
                    plot_path = os.path.join(eda_path, f"{col}.png")
                    plt.savefig(plot_path)
                    plt.close("all")

                elif PreprocessingUtils.get_type(X[col]) in ("continous"):

                    plt.figure(figsize=(5, 5))
                    sns.distplot(X[col], color=BLUE)
                    plt.title(f"{col} value distribution")
                    plt.tight_layout(pad=2.0)
                    plot_path = os.path.join(eda_path, f"{col}.png")
                    plt.savefig(plot_path)
                    plt.close("all")

                elif PreprocessingUtils.get_type(X[col]) in ("text"):

                    plt.figure(figsize=(10, 10), dpi=70)
                    word_string = " ".join(X[col].str.lower())
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

                elif PreprocessingUtils.get_type(X[col]) in ("datetime"):

                    plt.figure(figsize=(5, 5))
                    pd.to_datetime(X[col]).plot(grid="True", color=BLUE)
                    plt.tight_layout(pad=2.0)
                    plot_path = os.path.join(eda_path, f"{col}.png")
                    plt.savefig(plot_path)
                    plt.close("all")

                inform["missing"].append(pd.isnull(X[col]).sum() * 100 / X.shape[0])

                inform["unique"].append(int(X[col].nunique()))
                inform["plot"].append(f"![]({col}.png)")
                inform["feature"].append(str(col))
                inform["desc"].append(X[col].describe().to_dict())

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

                            fout.write(
                                f"- **{key.capitalize()}** :{row['desc'][key]}\n"
                            )

                    fout.write(f"- {row['plot']}\n")

            fout.close()
        except Exception as e:
            logger.error(f"There was an issue when running EDA. {str(e)}")

    @staticmethod
    def extensive_eda(X, y,save_path):

        X = X.copy(deep=True)
        # Check for empty dataframes in params
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe")
        if X.shape[0] != len(y):
            raise ValueError("X and y should have same number of samples")

        if save_path:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        else:
            raise ValueError("provide a valid path to save plots") 


        plt.style.use("ggplot")
        try:

            if PreprocessingUtils.get_type(y) in ("categorical", "discrete"):

                for col in X.columns:

                    if PreprocessingUtils.get_type(X[col]) == "continous":

                        plt.figure(figsize=(5, 5))
                        for i in np.unique(y):
                            sns.kdeplot(
                                X.iloc[np.where(y == i)[0]][col],
                                label=f"class {i}",
                                shade=True,
                            )
                        plt.legend()
                        plt.gca().set_title(
                            f"Distribution of {col} for each class",
                            fontsize=11,
                            weight="bold",
                            alpha=0.75,
                        )
                        plt.savefig(os.path.join(save_path,f"{col}_target"))

                    elif PreprocessingUtils.get_type(X[col]) in (
                        "categorical",
                        "discrete",
                    ):

                        if np.nunique(X[col]) <= 7:

                            plt.figure(figsize=(5, 5))
                            sns.countplot(x=X[col], hue=y)
                            plt.gca().set_title(
                                f"Count plot of each {col}",
                                fontsize=11,
                                weight="bold",
                                alpha=0.75,
                            )
                            plt.savefig(os.path.join(save_path,f"{col}_target"))


            elif PreprocessingUtils.get_type(y) == "continous":
                for col in X.columns:

                    if PreprocessingUtils.get_type(X[col]) == "continous":

                        plt.figure(figsize=(5, 5))
                        plt.scatter(X[col].values, y)
                        plt.gca().set_xlabel(f"{col}")
                        plt.gca().set_ylabel("target")
                        plt.gca().set_title(
                            f"Scatter plot of {col} vs target",
                            fontsize=11,
                            weight="bold",
                            alpha=0.75,
                        )

                        plt.savefig(os.path.join(save_path,f"{col}_target"))

                    elif PreprocessingUtils.get_type(X[col]) in (
                        "categorical",
                        "discrete",
                    ):
                        if X[col].nunique() <= 7:

                            plt.figure(figsize=(5, 5))
                            for i in X[col].unique():
                                sns.kdeplot(
                                    y[X[X[col] == i].index],
                                    shade=True,
                                    label=f"{col}_{i}",
                                )
                            plt.gca().set_title(
                                f"Distribution of target for each {col}",
                                fontsize=11,
                                weight="bold",
                                alpha=0.75,
                            )
                            plt.legend()

                            plt.savefig(os.path.join(save_path,f"{col}_target"))


                    elif PreprocessingUtils.get_type(X[col]) == "datetime":

                        plt.figure(figsize=(5, 5))
                        plt.plot(X[col], y)
                        plt.gca().set_xticklabels(X[col].dt.date, rotation="45")
                        plt.gca().set_title(
                            f"Distribution of target over time",
                            fontsize=11,
                            weight="bold",
                            alpha=0.75,
                        )
                        plt.savefig(os.path.join(save_path,f"{col}_target"))


            cols = [
                col
                for col in X.columns
                if PreprocessingUtils.get_type(X[col]) == "continous"
            ][:COLS]
            X["target"] = y
            plt.figure(figsize=(10, 10))
            sns.heatmap(X[cols + ["target"]].corr())
            plt.gca().set_title(
                "Heatmap", fontsize=11, weight="bold", alpha=0.75,
            )

            plt.savefig(os.path.join(save_path,"heatmap"))


        except Exception as e:
            AutoMLException(e)
