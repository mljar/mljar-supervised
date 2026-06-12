import pandas as pd


def filter_oof_to_original_rows(oof_df, original_rows):
    """Keep only out-of-fold predictions for original (non-upsampled) rows."""
    if original_rows is None or oof_df is None or oof_df.empty:
        return oof_df
    if oof_df.shape[0] <= original_rows:
        return oof_df
    return oof_df[oof_df.index < original_rows].copy(deep=True)


def filter_oofs_dict(oofs, original_rows):
    if original_rows is None:
        return oofs
    return {
        name: filter_oof_to_original_rows(oof, original_rows)
        for name, oof in oofs.items()
    }


def filter_target_to_original_rows(target, original_rows):
    if original_rows is None or target is None:
        return target
    if isinstance(target, pd.DataFrame):
        if target.shape[0] <= original_rows:
            return target
        return target[target.index < original_rows].copy(deep=True)
    if isinstance(target, pd.Series):
        if target.shape[0] <= original_rows:
            return target
        return target[target.index < original_rows].copy(deep=True)
    return target
