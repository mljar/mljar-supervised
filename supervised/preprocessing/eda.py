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

import string
import base64


class EDA:
    """ Creates plots for Automated Exploratory Data Analysis. """

    @staticmethod
    def prepare(column):
        """ Prepare the column to be used as file name. """
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        valid_chars = frozenset(valid_chars)
        col = "".join(c for c in column if c in valid_chars)
        if not len(col):
            col = base64.urlsafe_b64encode(column.encode("utf-8"))
            col = col.decode("utf-8")
        return col.replace(" ", "_")

    @staticmethod
    def plot_fname(column):
        """ Returns file name for the plot based on the column name. """
        return EDA.prepare(column) + ".png"

    @staticmethod
    def plot_path(eda_path, column):
        """ Returns full path for the plot based on the column name. """
        fname = os.path.join(eda_path, EDA.plot_fname(column))
        return fname

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

                plt.figure(figsize=(5, 5))
                if PreprocessingUtils.get_type(y) in ("categorical"):
                    sns.countplot(y, color=BLUE)
                else:
                    sns.distplot(y, color=BLUE)
                plt.title("Target class distribution")
                plt.tight_layout(pad=2.0)
                plt.savefig(EDA.plot_path(eda_path, "target"))
                plt.close("all")

                inform["missing"].append(pd.isnull(y).sum() / y.shape[0])
                inform["unique"].append(y.nunique())
                inform["feature_type"].append(PreprocessingUtils.get_type(y))
                inform["plot"].append(f"![]({EDA.plot_fname('target')})")
                inform["feature"].append("target")
                inform["desc"].append(y.describe().to_dict())
            for col in X.columns:
                inform["feature_type"].append(PreprocessingUtils.get_type(X[col]))

                if PreprocessingUtils.get_type(X[col]) in ("categorical", "discrete"):

                    plt.figure(figsize=(5, 5))
                    chart = sns.countplot(
                        X[col], order=X[col].value_counts().iloc[:10].index, color=BLUE
                    )
                    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
                    plt.title(f"{col} class distribution")
                    plt.tight_layout(pad=2.0)

                elif PreprocessingUtils.get_type(X[col]) in ("continous"):

                    plt.figure(figsize=(5, 5))
                    sns.distplot(X[col], color=BLUE)
                    plt.title(f"{col} value distribution")
                    plt.tight_layout(pad=2.0)

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

                elif PreprocessingUtils.get_type(X[col]) in ("datetime"):

                    plt.figure(figsize=(5, 5))
                    pd.to_datetime(X[col]).plot(grid="True", color=BLUE)
                    plt.tight_layout(pad=2.0)

                plt.savefig(EDA.plot_path(eda_path, col))
                plt.close("all")

                inform["missing"].append(pd.isnull(X[col]).sum() * 100 / X.shape[0])
                inform["unique"].append(int(X[col].nunique()))
                inform["plot"].append(f"![]({EDA.plot_fname(col)})")
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
