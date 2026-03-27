import pandas as pd
import os

def load_data(path, **kwargs):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.csv', '.tsv']:
        return pd.read_csv(path, **kwargs)
    elif ext in ['.xls', '.xlsx', '.ods']:
        return pd.read_excel(path, **kwargs)
    elif ext == '.json':
        return pd.read_json(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
