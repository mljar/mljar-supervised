import json
import unittest
import tempfile
import numpy as np
import pandas as pd

from preprocessing_missing import PreprocessingMissingValues
from preprocessing_categorical import PreprocessingCategorical
from preprocessing_box import PreprocessingBox


class PreprocessingBoxTest(unittest.TestCase):

    def test_constructor_preprocessing_box(self):

        preprocess = PreprocessingBox(PreprocessingMissingValues.FILL_NA_MEDIAN, PreprocessingCategorical.CONVERT_INTEGER)
        self.assertEqual(preprocess._missing_values_method, PreprocessingMissingValues.FILL_NA_MEDIAN)
        self.assertTrue(preprocess._missing_values is None)
        self.assertEqual(preprocess._categorical_method, PreprocessingCategorical.CONVERT_INTEGER)
        self.assertTrue(preprocess._categorical is None)


    def test_fit_median_integer(self):
        # training data
        d = {
                'col1': [1, 1, np.nan, 3],
                'col2': ['a', 'a', np.nan, 'a'],
                'col3': [1, 1, 1, 3],
                'col4': ['a', 'a', 'b', 'c']
            }
        df = pd.DataFrame(data=d)
        preprocess = PreprocessingBox(PreprocessingMissingValues.FILL_NA_MEDIAN, PreprocessingCategorical.CONVERT_INTEGER)
        preprocess.fit(df)

        self.assertTrue('col1' in preprocess._missing_values._na_fill_params)
        self.assertTrue('col2' in preprocess._missing_values._na_fill_params)
        self.assertTrue('col3' not in preprocess._missing_values._na_fill_params)
        self.assertTrue('col4' not in preprocess._missing_values._na_fill_params)

        self.assertEqual(1, preprocess._missing_values._na_fill_params['col1'])
        self.assertEqual('a', preprocess._missing_values._na_fill_params['col2'])

        self.assertTrue('col1' not in preprocess._categorical._convert_params)
        self.assertTrue('col2' in preprocess._categorical._convert_params)
        self.assertTrue('col3' not in preprocess._categorical._convert_params)
        self.assertTrue('col4' in preprocess._categorical._convert_params)

        self.assertTrue('a' in preprocess._categorical._convert_params['col2'])
        self.assertEqual(1, len(preprocess._categorical._convert_params['col2']))

        # check if data frame is untouched after preprocessing
        self.assertTrue(np.isnan(df['col1'][2]))
        self.assertEqual(df['col2'][0], 'a')

    def test_fit_median_one_hot(self):
        # training data
        d = {
                'col1': [1, 1, np.nan, 3],
                'col2': ['a', 'a', np.nan, 'a'],
                'col3': [1, 1, 1, 3],
                'col4': ['a', 'a', 'b', 'c']
            }
        df = pd.DataFrame(data=d)
        preprocess = PreprocessingBox(PreprocessingMissingValues.FILL_NA_MEDIAN, PreprocessingCategorical.CONVERT_ONE_HOT)
        preprocess.fit(df)

        self.assertTrue('col1' in preprocess._missing_values._na_fill_params)
        self.assertTrue('col2' in preprocess._missing_values._na_fill_params)
        self.assertTrue('col3' not in preprocess._missing_values._na_fill_params)
        self.assertTrue('col4' not in preprocess._missing_values._na_fill_params)

        self.assertEqual(1, preprocess._missing_values._na_fill_params['col1'])
        self.assertEqual('a', preprocess._missing_values._na_fill_params['col2'])

        self.assertTrue('col1' not in preprocess._categorical._convert_params)
        self.assertTrue('col2' in preprocess._categorical._convert_params)
        self.assertTrue('col3' not in preprocess._categorical._convert_params)
        self.assertTrue('col4' in preprocess._categorical._convert_params)

        self.assertTrue('col2_a' in preprocess._categorical._convert_params['col2']['new_columns'])
        self.assertTrue('col4_a' in preprocess._categorical._convert_params['col4']['new_columns'])
        self.assertTrue('col4_b' in preprocess._categorical._convert_params['col4']['new_columns'])
        self.assertTrue('col4_c' in preprocess._categorical._convert_params['col4']['new_columns'])

        # check if data frame is untouched after preprocessing
        self.assertTrue(np.isnan(df['col1'][2]))
        self.assertEqual(df['col2'][0], 'a')


    def test_fit_transform_median_integer(self):
        # training data
        d = {
                'col1': [1, 1, np.nan, 3],
                'col2': ['a', 'a', np.nan, 'a'],
                'col3': [1, 1, 1, 3],
                'col4': ['a', 'a', 'b', 'c']
            }
        df = pd.DataFrame(data=d)
        preprocess = PreprocessingBox(PreprocessingMissingValues.FILL_NA_MEDIAN, PreprocessingCategorical.CONVERT_INTEGER)
        preprocess.fit(df)
        # check if data frame is untouched after preprocessing fit
        self.assertTrue(np.isnan(df['col1'][2]))
        self.assertEqual(df['col2'][0], 'a')
        # apply transform
        df = preprocess.transform(df)

        for col in ['col1', 'col2', 'col3', 'col4']:
            self.assertTrue(col in df.columns)
        self.assertEqual(1, df['col1'][2])
        self.assertEqual(0, df['col2'][0])
        self.assertEqual(0, df['col2'][2])
        self.assertEqual(0, df['col4'][0])
        self.assertEqual(0, df['col4'][1])
        self.assertEqual(1, df['col4'][2])
        self.assertEqual(2, df['col4'][3])

    def test_fit_transform_median_integer_new_values(self):
        # training data
        d = {
                'col1': [1, 1, np.nan, 3],
                'col2': ['a', 'a', np.nan, 'a'],
                'col3': [1, 1, 1, 3],
                'col4': ['a', 'a', 'b', 'c']
            }
        df = pd.DataFrame(data=d)
        preprocess = PreprocessingBox(PreprocessingMissingValues.FILL_NA_MEDIAN, PreprocessingCategorical.CONVERT_INTEGER)
        preprocess.fit(df)
        # check if data frame is untouched after preprocessing fit
        self.assertTrue(np.isnan(df['col1'][2]))
        self.assertEqual(df['col2'][0], 'a')
        # apply transform on new data
        d_test = {
                'col1': [1, 2, np.nan, 1],
                'col2': ['a', 'b', np.nan, 'a'],
                'col3': [1, 1, 1, 3],
                'col4': ['a', 'e', 'f', 'c']
            }
        df_test = pd.DataFrame(data=d_test)

        df_test = preprocess.transform(df_test)

        for col in ['col1', 'col2', 'col3', 'col4']:
            self.assertTrue(col in df_test.columns)

        self.assertEqual(1, df_test['col1'][2])
        self.assertEqual(0, df_test['col2'][0])
        self.assertEqual(1, df_test['col2'][1]) # new value set to higher number
        self.assertEqual(0, df_test['col2'][2])
        self.assertEqual(0, df_test['col4'][0])
        self.assertEqual(3, df_test['col4'][1])
        self.assertEqual(4, df_test['col4'][2])
        self.assertEqual(2, df_test['col4'][3])


    def test_fit_transform_median_one_hot(self):
        # training data
        d = {
                'col1': [1, 1, np.nan, 3],
                'col2': ['a', 'a', np.nan, 'a'],
                'col3': [1, 1, 1, 3],
                'col4': ['a', 'a', 'b', 'c']
            }
        df = pd.DataFrame(data=d)
        preprocess = PreprocessingBox(PreprocessingMissingValues.FILL_NA_MEDIAN, PreprocessingCategorical.CONVERT_ONE_HOT)
        preprocess.fit(df)
        # check if data frame is untouched after preprocessing fit
        self.assertTrue(np.isnan(df['col1'][2]))
        self.assertEqual(df['col2'][0], 'a')
        # apply transform
        df = preprocess.transform(df)

        for col in ['col1', 'col2_a', 'col3', 'col4_a', 'col4_b', 'col4_c']:
            self.assertTrue(col in df.columns)
        self.assertEqual(1, df['col1'][2])
        self.assertEqual(1, df['col2_a'][0])
        self.assertEqual(1, df['col2_a'][2])
        self.assertEqual(2, np.sum(df['col4_a']))
        self.assertEqual(1, np.sum(df['col4_b']))
        self.assertEqual(1, np.sum(df['col4_c']))
        self.assertEqual(1, df['col4_a'][1])
        self.assertEqual(1, df['col4_b'][2])
        self.assertEqual(1, df['col4_c'][3])


    def test_fit_transform_median_one_hot_new_values(self):
        # training data
        d = {
                'col1': [1, 1, np.nan, 3],
                'col2': ['a', 'a', np.nan, 'a'],
                'col3': [1, 1, 1, 3],
                'col4': ['a', 'a', 'b', 'c']
            }
        df = pd.DataFrame(data=d)
        preprocess = PreprocessingBox(PreprocessingMissingValues.FILL_NA_MEDIAN, PreprocessingCategorical.CONVERT_ONE_HOT)
        preprocess.fit(df)
        # check if data frame is untouched after preprocessing fit
        self.assertTrue(np.isnan(df['col1'][2]))
        self.assertEqual(df['col2'][0], 'a')
        # apply transform on new data
        d_test = {
                'col1': [1, 1, np.nan, 3],
                'col2': ['a', 'b', np.nan, 'a'],
                'col3': [1, 1, 1, 3],
                'col4': ['a', 'e', 'f', 'c']
            }
        df_test = pd.DataFrame(data=d_test)

        df_test = preprocess.transform(df_test)

        for col in ['col1', 'col2_a', 'col3', 'col4_a', 'col4_b', 'col4_c']:
            self.assertTrue(col in df_test.columns)

        self.assertEqual(1, df_test['col1'][2])
        self.assertEqual(1, df_test['col2_a'][0])
        self.assertEqual(0, df_test['col2_a'][1]) # new value set to 0
        self.assertEqual(1, df_test['col2_a'][2])
        self.assertEqual(3, np.sum(df_test['col2_a']))

        self.assertEqual(1, np.sum(df_test['col4_a']))
        self.assertEqual(0, np.sum(df_test['col4_b']))
        self.assertEqual(1, np.sum(df_test['col4_c']))


    def test_save_and_load_median_integer(self):
        # training data
        d = {
                'col1': [1, 1, np.nan, 3],
                'col2': ['a', 'a', np.nan, 'a'],
                'col3': [1, 1, 1, 3],
                'col4': ['a', 'a', 'b', 'c']
            }
        df = pd.DataFrame(data=d)
        preprocess = PreprocessingBox(PreprocessingMissingValues.FILL_NA_MEDIAN, PreprocessingCategorical.CONVERT_INTEGER)
        preprocess.fit(df)
        # check if data frame is untouched after preprocessing fit
        self.assertTrue(np.isnan(df['col1'][2]))
        self.assertEqual(df['col2'][0], 'a')

        preprocess2 = PreprocessingBox()
        with tempfile.NamedTemporaryFile() as temp:
            preprocess.save(temp.name)
            preprocess2.load(temp.name)

        # apply transform
        df = preprocess2.transform(df)

        for col in ['col1', 'col2', 'col3', 'col4']:
            self.assertTrue(col in df.columns)
        self.assertEqual(1, df['col1'][2])
        self.assertEqual(0, df['col2'][0])
        self.assertEqual(0, df['col2'][2])
        self.assertEqual(0, df['col4'][0])
        self.assertEqual(0, df['col4'][1])
        self.assertEqual(1, df['col4'][2])
        self.assertEqual(2, df['col4'][3])


    def test_save_and_load_median_one_hot(self):
        # training data
        d = {
                'col1': [1, 1, np.nan, 3],
                'col2': ['a', 'a', np.nan, 'a'],
                'col3': [1, 1, 1, 3],
                'col4': ['a', 'a', 'b', 'c']
            }
        df = pd.DataFrame(data=d)
        preprocess = PreprocessingBox(PreprocessingMissingValues.FILL_NA_MEDIAN, PreprocessingCategorical.CONVERT_ONE_HOT)
        preprocess.fit(df)
        # check if data frame is untouched after preprocessing fit
        self.assertTrue(np.isnan(df['col1'][2]))
        self.assertEqual(df['col2'][0], 'a')
        # create second object
        preprocess2 = PreprocessingBox()
        with tempfile.NamedTemporaryFile() as temp:
            preprocess.save(temp.name)
            preprocess2.load(temp.name)

        # apply transform
        df = preprocess2.transform(df)

        for col in ['col1', 'col2_a', 'col3', 'col4_a', 'col4_b', 'col4_c']:
            self.assertTrue(col in df.columns)
        self.assertEqual(1, df['col1'][2])
        self.assertEqual(1, df['col2_a'][0])
        self.assertEqual(1, df['col2_a'][2])
        self.assertEqual(2, np.sum(df['col4_a']))
        self.assertEqual(1, np.sum(df['col4_b']))
        self.assertEqual(1, np.sum(df['col4_c']))
        self.assertEqual(1, df['col4_a'][1])
        self.assertEqual(1, df['col4_b'][2])
        self.assertEqual(1, df['col4_c'][3])




if __name__ == '__main__':
    unittest.main()
