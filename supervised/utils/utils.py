import numpy as np
import pandas as pd


def dump_data(file_path, df):
    try:
        df.to_parquet(file_path, index=False)
    except Exception as e:
        df.to_csv(file_path, index=False)


def load_data(file_path):
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        return pd.read_csv(file_path)
