# Feature engineering logic
import numpy as np

def add_time_features(df):
    df["hour"] = df["hour_timestamp"].dt.hour
    df["dayofweek"] = df["hour_timestamp"].dt.dayofweek
    df["month"] = df["hour_timestamp"].dt.month
    df["day"] = df["hour_timestamp"].dt.day
    df["weekofyear"] = df["hour_timestamp"].dt.isocalendar().week.astype(int)

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"] = np.sin(2*np.pi*df["dayofweek"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dayofweek"]/7)

    df["station_encoded"] = df["stationID"].astype("category").cat.codes

    return df


def add_lag_features(df):
    df = df.sort_values(["stationID","hour_timestamp"])
    
    df["lag_1"] = df.groupby("stationID")["total_kWh"].shift(1)
    df["rolling_3h"] = df.groupby("stationID")["total_kWh"].transform(lambda x: x.rolling(3).mean())
    df["rolling_24h"] = df.groupby("stationID")["total_kWh"].transform(lambda x: x.rolling(24).mean())

    df.dropna(inplace=True)
    return df
