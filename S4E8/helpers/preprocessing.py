import warnings

import numpy as np
import pandas as pd


def convert_cols(df, cont_cols, cat_cols, treat_nan_as_cat=False):
    df = df.fillna(np.nan)

    for c in cont_cols:
        df[c] = df[c].astype("float32")

    for c in cat_cols:
        try:
            df[c] = df[c].astype("category")
            if treat_nan_as_cat:
                df[c] = df[c].cat.add_categories(["NaN_cat"])
        except KeyError:
            warnings.warn(f"'{c}' not found in dataframe")

    return df


def null_all_non_original_categories(df, orig, cat_cols, treat_nan_as_cat=False):
    for c in cat_cols:
        try:
            orig_categories = pd.unique(orig[c].dropna())
            if treat_nan_as_cat:
                orig_categories = np.append(orig_categories, ["NaN_cat"])
            cat_dtype = pd.CategoricalDtype(categories=orig_categories, ordered=False)

            if treat_nan_as_cat:
                df.loc[~df[c].isin(orig_categories), c] = "NaN_cat"
            else:
                df.loc[~df[c].isin(orig_categories), c] = np.nan

            df[c] = df[c].astype(cat_dtype)
        except KeyError:
            warnings.warn(f"'{c}' not found in dataframe")

    return df

def fix_that_one_mushroom_in_test(test_df):
    # There is a one mushroom with a cap diameter of 607 in the test dataset
    # Orig and train have diameter maxes at 62 and 81 respectively
    # I presume this one mushroom in test is "accidentally" scaled by 10, so this method is to fix that
    test_df.loc[3755132, "cap-diameter"] = test_df.loc[3755132, "cap-diameter"] / 10
    return test_df

if __name__ == "__main__":
    train_df = pd.read_csv("../data/train.csv", index_col="id")
    orig_df = pd.read_csv("../data/orig.csv", index_col="id")

    CONT_FEATS = ["cap-diameter", "stem-height", "stem-width"]
    CAT_FEATS = [c for c in train_df.columns if c not in CONT_FEATS]
    RESPONSE_COL = "class"

    train_df = convert_cols(train_df, CONT_FEATS, CAT_FEATS)
    train_df = null_all_non_original_categories(train_df, orig_df, CAT_FEATS)
