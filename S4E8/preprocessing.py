import warnings

import numpy as np
import pandas as pd


def convert_cols(df, cont_cols, cat_cols):
    df = df.fillna(np.nan)

    for c in cont_cols:
        df[c] = df[c].astype("float32")

    for c in cat_cols:
        try:
            df[c] = df[c].astype("category")
        except KeyError:
            warnings.warn(f"'{c}' not found in dataframe")

    return df

def null_all_non_original_categories(df, orig, cat_cols):
    for c in cat_cols:
        try:
            orig_categories = pd.unique(orig[c].dropna())
            cat_dtype = pd.CategoricalDtype(categories=orig_categories, ordered=False)

            df.loc[~df[c].isin(orig_categories), c] = np.nan

            df[c] = df[c].astype(cat_dtype)
        except KeyError:
            warnings.warn(f"'{c}' not found in dataframe")

    return df