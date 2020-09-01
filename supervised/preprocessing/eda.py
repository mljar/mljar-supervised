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
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# color used in the plots
BLUE = "#007cf2"


class EDA:
    @staticmethod
    def compute(X, y, eda_path):

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

                if PreprocessingUtils.get_type(X[col]) in (
                    "categorical",
                    "discrete",
                ):

                    plt.figure(figsize=(5, 5))
                    chart = sns.countplot(
                        X[col],
                        order=X[col].value_counts().iloc[:10].index,
                        color=BLUE,
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
