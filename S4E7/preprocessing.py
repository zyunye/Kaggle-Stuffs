from enum import Enum
import warnings

import category_encoders as ce
import pandas as pd
from sklearn.preprocessing import TargetEncoder

categorical_cols = [
    "Gender",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Policy_Sales_Channel",
    "Response",
]


def convert_cols(df):
    for col in categorical_cols:
        try:
            df[col] = df[col].astype("category")
        except KeyError:
            warnings.warn(f"{col} was not found in dataframe")
            continue

    # df["Age"] = df["Age"].astype("int8")
    # df["Region_Code"] = df["Region_Code"].astype("int8")
    # # df["Region_Code"] = df["Region_Code"].astype("category")
    # df["Annual_Premium"] = df["Annual_Premium"].astype("int32")
    # df["Policy_Sales_Channel"] = df["Policy_Sales_Channel"].astype("int16")
    # # df["Policy_Sales_Channel"] = df["Policy_Sales_Channel"].astype("category")
    # df["Vintage"] = df["Vintage"].astype("int16")

    # df["Gender"] = df["Gender"].cat.rename_categories({"Female": 0, "Male": 1})
    # df["Vehicle_Age"] = df["Vehicle_Age"].cat.rename_categories(
    #     {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}
    # )
    # df["Vehicle_Damage"] = df["Vehicle_Damage"].cat.rename_categories(
    #     {"No": 0, "Yes": 1}
    # )
    df["Gender"] = (
        df["Gender"].cat.rename_categories({"Female": 0, "Male": 1}).astype("uint8")
    )
    df["Age"] = df["Age"].astype("uint8")
    df["Driving_License"] = df["Driving_License"].astype("uint8")
    df["Region_Code"] = df["Region_Code"].astype("uint8")
    df["Previously_Insured"] = df["Previously_Insured"].astype("uint8")
    df["Vehicle_Age"] = (
        df["Vehicle_Age"]
        .cat.rename_categories({"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2})
        .astype("uint8")
    )
    df["Vehicle_Damage"] = (
        df["Vehicle_Damage"].cat.rename_categories({"No": 0, "Yes": 1}).astype("uint8")
    )
    df["Annual_Premium"] = df["Annual_Premium"].astype("uint32")
    df["Policy_Sales_Channel"] = df["Policy_Sales_Channel"].astype("int16")
    df["Vintage"] = df["Vintage"].astype("int16")
    try:
        df["Response"] = df["Response"].astype("uint8")
    except KeyError:
        pass

    return df


def downsample(df):
    zeroes = df[df["Response"] == 0]
    ones = df[df["Response"] == 1]
    undersampled_zeroes = zeroes.sample(len(ones))

    downsampled_df = pd.concat([ones, undersampled_zeroes])

    return downsampled_df


class FE_OPTS(Enum):
    Not_Insured_and_Damaged = 1
    Age_Discriminator = 2


def feature_engineer(
    df, opts=[FE_OPTS.Not_Insured_and_Damaged, FE_OPTS.Age_Discriminator]
):
    if FE_OPTS.Not_Insured_and_Damaged in opts:
        df["Not_Insured_and_Damaged"] = (df["Previously_Insured"] == 0) & (
            df["Vehicle_Damage"] == 1
        )
        df["Not_Insured_and_Damaged"] = df["Not_Insured_and_Damaged"].astype("uint8")

    if FE_OPTS.Age_Discriminator:
        # Magic number from running a KMeans over the data and seeing where the separation line was. See eda.py
        df["Young"] = 0
        df.loc[df["Age"] <= 39, "Young"] = 1
        df["Young"] = df["Young"].astype("uint8")

    return df


te_cols = [
    "Region_Code",
    "Policy_Sales_Channel",
]


def target_encode(train_df, val_df, test_df, cols, target_col="Response"):
    target_mean = train_df[target_col].mean()
    for col in cols:
        encoding = train_df[[col, target_col]].groupby(col).mean().reset_index()
        encoding = encoding.rename(columns={target_col: f"{col}_tgt_enc"})

        train_df = train_df.merge(encoding, on=col, how="left")
        train_df = train_df.fillna(target_mean)

        val_df = val_df.merge(encoding, on=col, how="left")
        val_df = val_df.fillna(target_mean)

        test_df = test_df.merge(encoding, on=col, how="left")
        test_df = test_df.fillna(target_mean)

    return train_df, val_df, test_df
