# Preprocessing logic
import pandas as pd
import json

def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data["_items"])


def clean_data(df):
    df = df.dropna(subset=["connectionTime","disconnectTime","kWhDelivered","stationID"])
    
    df["connectionTime"] = pd.to_datetime(df["connectionTime"], errors="coerce")
    df["disconnectTime"] = pd.to_datetime(df["disconnectTime"], errors="coerce")

    df = df.dropna(subset=["connectionTime","disconnectTime"])
    df = df[df["kWhDelivered"] > 0]
    df = df[df["disconnectTime"] > df["connectionTime"]]

    return df


def aggregate_hourly(df):
    df["hour_timestamp"] = df["connectionTime"].dt.floor("h")

    hourly = (
        df.groupby(["stationID","hour_timestamp"])["kWhDelivered"]
        .sum()
        .reset_index()
    )

    hourly.rename(columns={"kWhDelivered":"total_kWh"}, inplace=True)
    return hourly
