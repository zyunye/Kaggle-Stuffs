import pandas as pd
from sklearn.cluster import KMeans


def convert_cols(df, day_as_cat=False):
    try:
        df["rainfall"] = df["rainfall"].astype(pd.UInt8Dtype())
    except KeyError:
        print("WARN: No 'rainfall' column found")
    # float_cols = ["pressure", "maxtemp", "temperature", "mintemp", "dewpoint", "humidity", "cloud", "sunshine", "winddirection", "windspeed"]

    if day_as_cat:
        df["day"] = df["day"].astype("categorical")

    df = df.rename(columns={"temparature": "temperature"})

    return df


def train_kmeans(df, n_clusters=3, random_state=0):
    return KMeans(n_clusters=n_clusters, random_state=random_state).fit(df)
