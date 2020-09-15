import os
import re
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil
from sklearn import datasets

from supervised import AutoML
from supervised.preprocessing.eda import EDA


class EDATest(unittest.TestCase):

    automl_dir = "automl_tests"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_explain_default(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Baseline"],
            train_ensemble=False,
            explain_level=2,
        )

        X, y = datasets.make_classification(n_samples=100, n_features=5)
        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="class")

        a.fit(X, y)

        result_files = os.listdir(os.path.join(a._results_path, "EDA"))

        for col in X.columns:
            self.assertTrue(f"{col}.png" in result_files)
        self.assertTrue("target.png" in result_files)
        self.assertTrue("README.md" in result_files)

    def test_column_name_to_filename(self):
        """ Valid feature name should be untouched """
        col = "feature_1"
        self.assertEqual(EDA.prepare(col), col)

        self.tearDown()

    def test_extensive_eda(self):

        X, y = datasets.make_regression(n_samples=100, n_features=5)

        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="class")

        results_path = "EDA"
        EDA.extensive_eda(X, y, results_path)
        result_files = os.listdir(results_path)

        for col in X.columns:

            produced = True
            if "_target".join((col, ".png")) not in result_files:
                produced = False
                break
        self.assertTrue(produced)

        X, y = datasets.make_classification(n_samples=100, n_features=5)

        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="class")

        results_path = "EDA"
        EDA.extensive_eda(X, y, results_path)
        result_files = os.listdir(results_path)

        for col in X.columns:

            produced = True
            if "_target".join((col, ".png")) not in result_files:
                produced = False
                break
        self.assertTrue(produced)

        produced = True
        if "heatmap.png" not in result_files:
            produced = False
        self.assertTrue(produced)

        produced = True
        if "Extensive_EDA.md" not in result_files:
            produced = False
        self.assertTrue(produced)

        self.tearDown()

    def test_extensive_eda_missing(self):

        X, y = datasets.make_regression(n_samples=100, n_features=5)

        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="class")

        ##add some nan values
        X.loc[np.random.randint(0, 100, 20), "f_0"] = np.nan

        results_path = "EDA"
        EDA.extensive_eda(X, y, results_path)
        result_files = os.listdir(results_path)

        for col in X.columns:

            produced = True
            if "_target".join((col, ".png")) not in result_files:
                produced = False
                break
        self.assertTrue(produced)

        X, y = datasets.make_regression(n_samples=100, n_features=5)

        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="class")

        ##add some nan values
        X.loc[np.random.randint(0, 100, 20), "f_0"] = np.nan

        results_path = "EDA"
        EDA.extensive_eda(X, y, results_path)
        result_files = os.listdir(results_path)

        for col in X.columns:

            produced = True
            if "_target".join((col, ".png")) not in result_files:
                produced = False
                break
        self.assertTrue(produced)

        produced = True
        if "heatmap.png" not in result_files:
            produced = False
        self.assertTrue(produced)

        produced = True
        if "Extensive_EDA.md" not in result_files:
            produced = False
        self.assertTrue(produced)

        self.tearDown()

    def test_symbol_feature(self):

        X, y = datasets.make_regression(n_samples=100, n_features=5)

        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        X.rename({"f_0": "ff*", "f_1": "fg/"}, axis=1, inplace=True)
        y = pd.Series(y, name="class")

        results_path = "EDA"
        EDA.extensive_eda(X, y, results_path)
        result_files = os.listdir(results_path)

        ## replace  symbol for filenames
        X.columns = [re.sub(r'[\\/*?:"<>|]', "_", x) for x in X.columns]

        for col in X.columns:

            produced = True
            if "_target".join((col, ".png")) not in result_files:
                produced = False
                break
        self.assertTrue(produced)

        produced = True
        if "Extensive_EDA.md" not in result_files:
            produced = False
        self.assertTrue(produced)

        self.tearDown()

    def test_naughty_column_name_to_filename(self):
        """ Test with naughty strings.
            String from https://github.com/minimaxir/big-list-of-naughty-strings """
        os.mkdir(self.automl_dir)
        naughty_columns = [
            "feature_1",
            "*",
            "😍",
            "¯\_(ツ)_/¯",
            "表",
            "𠜎𠜱𠝹𠱓",
            "عاملة بولندا",
            "Ṱ̺̺̕o͞ ̷" "🇸🇦🇫🇦🇲",
            "⁰⁴⁵",
            "∆˚¬…æ",
            "!@#$%^&*()`~",
            "onfocus=JaVaSCript:alert(123) autofocus",
            "`\"'><img src=xxx:x \x20onerror=javascript:alert(1)>",
            'System("ls -al /")',
            'Kernel.exec("ls -al /")',
            "لُلُصّبُلُل" "{% print 'x' * 64 * 1024**3 %}",
            '{{ "".__class__.__mro__[2].__subclasses__()[40]("/etc/passwd").read() }}',
            "ÜBER Über German Umlaut",
            "影師嗎",
            "C'est déjà l'été." "Nín hǎo. Wǒ shì zhōng guó rén",
            "Компьютер",
            "jaja---lol-méméméoo--a",
        ]
        for col in naughty_columns:
            fname = EDA.plot_path(self.automl_dir, col)
            with open(fname, "w") as fout:
                fout.write("ok")
                
