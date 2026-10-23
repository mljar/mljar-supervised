import ipaddress

import numpy as np
import pandas as pd


def parse_ip(value):
    """Parse plain address strings using the same rules at fit and prediction."""
    if not isinstance(value, str):
        raise ValueError("IP addresses must be strings")
    value = value.strip()
    if "%" in value:
        raise ValueError("IPv6 zone identifiers are not supported")
    return ipaddress.ip_address(value)


class IPTransformer:
    VERSION = 2

    def __init__(self):
        self._old_column = None
        self._new_columns = []
        self._active_indices = []

    def fit(self, X, column):
        self._old_column = column
        suffixes = ["Version"]
        suffixes += [f"IPv4_{i}" for i in range(1, 5)]
        suffixes += [f"IPv6_{i}" for i in range(1, 9)]
        suffixes += ["Missing"]
        names = [f"{column}_IP_{suffix}" for suffix in suffixes]
        values = self._encode(X)
        if not len(values):
            raise ValueError("Cannot fit IP transformation on empty data")
        # Select on training data only; validation and prediction reuse this mask.
        self._active_indices = np.flatnonzero(
            np.any(values != values[0], axis=0)
        ).tolist()
        self._new_columns = [names[i] for i in self._active_indices]
        self._check_collisions(X)

    def _check_collisions(self, X):
        collisions = [name for name in self._new_columns if name in X.columns]
        if collisions:
            raise ValueError(f"IP feature names already exist: {collisions}")

    def _encode(self, X):
        values = np.zeros((len(X), 14), dtype=np.float64)
        for position, (row, value) in enumerate(X[self._old_column].items()):
            if pd.isna(value):
                values[position, -1] = 1
                continue
            try:
                address = parse_ip(value)
            except ValueError as error:
                raise ValueError(
                    f"Invalid IP address in column {self._old_column!r}, row {row!r}: {value!r}"
                ) from error
            values[position, 0] = address.version
            if address.version == 4:
                values[position, 1:5] = list(address.packed)
            else:
                packed = address.packed
                values[position, 5:13] = [
                    int.from_bytes(packed[i : i + 2], "big") for i in range(0, 16, 2)
                ]
        return values

    def transform(self, X):
        self._check_collisions(X)
        values = self._encode(X)
        if self._new_columns:
            X[self._new_columns] = values[:, self._active_indices]
        X.drop(self._old_column, axis=1, inplace=True)
        return X

    def to_json(self):
        return {
            "version": self.VERSION,
            "old_column": self._old_column,
            "new_columns": list(self._new_columns),
            "active_indices": list(self._active_indices),
        }

    def from_json(self, data_json):
        if data_json["version"] not in (1, self.VERSION):
            raise ValueError("Unsupported IP transformation version")
        self._old_column = data_json["old_column"]
        self._new_columns = list(data_json["new_columns"])
        self._active_indices = (
            list(range(14))
            if data_json["version"] == 1
            else list(data_json["active_indices"])
        )
