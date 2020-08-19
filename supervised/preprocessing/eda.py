import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from wordcloud import STOPWORDS

from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.registry import REGRESSION
from supervised.exceptions import AutoMLException
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils


class EDA:


    def __init__(self,X,y,ml_task, eda_path ):

        self.X_train = X
        self.y_train = y
        self._ml_task = ml_task
        self.eda_path = eda_path


    
    def compute(self,X_train,y_train):

        inform = {}

        if self._ml_task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:

            plt.figure(figsize=(5, 5))
            sns.countplot(y_train, color='blue')
            plt.title("Target class distribution")
            plt.tight_layout(pad=2.0)
            plot_path = os.path.join(self.eda_path, "target.png")
            plt.savefig(plot_path)
            plt.close("all")

          
        elif self._ml_task == REGRESSION:

            plt.figure(figsize=(5, 5))
            sns.distplot(y_train, color='blue')
            plt.title("Target class distribution")
            plt.tight_layout(pad=2.0)
            plot_path = os.path.join(self.eda_path, "target.png")
            plt.savefig(plot_path)
            plt.close("all")


        inform["missing"] = [pd.isnull(y_train).sum() / y_train.shape[0]]
        inform["unique"] = [y_train.nunique()]
        inform['feature_type'] = [PreprocessingUtils.get_type(y_train)]
        inform['plot'] = ['![](target.png)']
        inform['feature'] = ["target"]
        inform['desc'] = [y_train.describe().to_dict()]

        for col in X_train.columns:

            inform['feature_type'] += [
                PreprocessingUtils.get_type(
                    X_train[col])]

            if PreprocessingUtils.get_type(
                    X_train[col]) in (
                    "categorical",
                    "discrete"):

                plt.figure(figsize=(5, 5))
                sns.countplot(
                    X_train[col], order=X_train[col].value_counts().iloc[:10].index, color='blue')
                plt.title(f"{col} class distribution")
                plt.tight_layout(pad=2.0)
                plot_path = os.path.join(self.eda_path, f"{col}.png")
                plt.savefig(plot_path)
                plt.close("all")


            elif PreprocessingUtils.get_type(X_train[col]) in ("continous"):

                plt.figure(figsize=(5, 5))
                sns.distplot(X_train[col], color='blue')
                plt.title(f"{col} value distribution")
                plt.tight_layout(pad=2.0)
                plot_path = os.path.join(self.eda_path, f"{col}.png")
                plt.savefig(plot_path)
                plt.close("all")


            elif PreprocessingUtils.get_type(X_train[col]) in ('text'):

                plt.figure(figsize=(5, 5), dpi=70)
                word_string = " ".join(X_train[col].str.lower())
                wordcloud = WordCloud(
                    width=500,
                    height=500,
                    stopwords=STOPWORDS,
                    background_color='white',
                    max_words=400,
                    max_font_size=None,
                ).generate(word_string)

                plt.imshow(wordcloud, aspect="auto", interpolation="nearest")
                plt.axis('off')
                plot_path = os.path.join(self.eda_path, f"{col}.png")
                plt.savefig(plot_path)

            elif PreprocessingUtils.get_type(X_train[col]) in ('datetime'):

                plt.figure(figsize=(5, 5))
                pd.to_datetime(df[col]).plot(grid='True', color='blue')
                plt.tight_layout(pad=2.0)
                plot_path = os.path.join(self.eda_path, f"{col}.png")
                plt.savefig(plot_path)
                plt.close("all")



            inform["missing"] += [pd.isnull(X_train[col]).sum()
                                      * 100 / X_train.shape[0]]

            inform["unique"] += [int(X_train[col].nunique())]
            inform['plot'] += [f'![]({col}.png)']
            inform['feature'] += [str(col)]
            inform['desc'] += [X_train[col].describe().to_dict()]

        df = pd.DataFrame(inform)

        with open(os.path.join(self.eda_path, "Readme.md"), "w") as fout:

            for i, row in df.iterrows():

                fout.write(f"## Feature : {row['feature']}\n")
                fout.write(f"- **Feature type** : {row['feature_type']}\n")
                fout.write(f"- **Missing** : {row['missing']}%\n")
                fout.write(f"- **Unique** : {row['unique']}\n")

                for key in row['desc'].keys():

                    if key in ("25%", "50%", "75%"):

                        fout.write(
                            f"- **{key.capitalize()}th Percentile** : {row['desc'][key]}\n")
                    else:

                        fout.write(
                            f"- **{key.capitalize()}** :{row['desc'][key]}\n")

                fout.write(f"- {row['plot']}\n")

        fout.close()

