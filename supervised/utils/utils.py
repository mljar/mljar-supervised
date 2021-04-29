import copy
import numpy as np
import pandas as pd


class Store:
    data = {}

    def set(self, key, value):
        Store.data[key] = value

    def get(self, key):
        return copy.deepcopy(Store.data[key])


def dump_data(file_path, df):
    store = Store()
    store.set(file_path, df)
    # try:
    #    df.to_parquet(file_path, index=False)
    # except Exception as e:
    #    df.to_csv(file_path, index=False)


def load_data(file_path):
    store = Store()
    return store.get(file_path)
    # try:
    #    return pd.read_parquet(file_path)
    # except Exception as e:
    #    return pd.read_csv(file_path)
