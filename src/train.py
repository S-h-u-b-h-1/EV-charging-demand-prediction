# Training script
import pandas as pd
from src.preprocessing import load_data, clean_data, aggregate_hourly
from src.feature_engineering import add_time_features, add_lag_features
from src.model import get_models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

def train_pipeline(json_path):

    df = load_data(json_path)
    df = clean_data(df)
    hourly = aggregate_hourly(df)

    hourly = add_time_features(hourly)
    hourly = add_lag_features(hourly)

    split_index = int(len(hourly)*0.8)
    split_date = hourly["hour_timestamp"].sort_values().iloc[split_index]

    train = hourly[hourly["hour_timestamp"] < split_date]
    test  = hourly[hourly["hour_timestamp"] >= split_date]

    features = [
        "station_encoded","hour","dayofweek","month","day","weekofyear",
        "hour_sin","hour_cos","dow_sin","dow_cos",
        "lag_1","rolling_3h","rolling_24h"
    ]

    X_train, y_train = train[features], train["total_kWh"]
    X_test, y_test = test[features], test["total_kWh"]

    models = get_models()

    results = []

    for name, model in models.items():
        model.fit(X_train,y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test,preds)
        rmse = np.sqrt(mean_squared_error(y_test,preds))
        r2 = r2_score(y_test,preds)

        results.append((name,rmse))

    best_model_name = sorted(results,key=lambda x: x[1])[0][0]
    best_model = models[best_model_name]

    joblib.dump(best_model,"models/trained_model.pkl")

    return best_model_name
