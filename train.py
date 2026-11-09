import os
import pandas as pd
import numpy as np
import yaml
import json
import mlflow
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configurar tracking remoto
mlflow.set_tracking_uri("http://44.198.230.92:5000/")
mlflow.set_experiment("Taller_Demanda_Electrica_XM")

with open("params.yaml", "r") as f_yaml:
    params = yaml.safe_load(f_yaml)

df_train = pd.read_csv("train.csv", parse_dates=[0], index_col=0)
df_test = pd.read_csv("test.csv", parse_dates=[0], index_col=0)

# Baseline
with mlflow.start_run(run_name="Baseline_Naive"):
    mlflow.log_param("model_type", "Naive_Seasonal")
    mlflow.log_param("test_hours", params["test_hours"])
    df_test_baseline = df_train.iloc[-24:].copy()
    y_baseline = np.tile(df_test_baseline["Demanda_MWh"].values, 2)[:params["test_hours"]]
    rmse_b = np.sqrt(mean_squared_error(df_test["Demanda_MWh"], y_baseline))
    mae_b = mean_absolute_error(df_test["Demanda_MWh"], y_baseline)
    mlflow.log_metric("RMSE", rmse_b)
    mlflow.log_metric("MAE", mae_b)
    print(f"[MLflow] Baseline registrado. RMSE: {rmse_b:.2f}")

# SARIMAX
with mlflow.start_run(run_name="SARIMAX_Model"):
    mlflow.log_param("model_type", "SARIMAX")
    mlflow.log_param("order", str(params["order"]))
    mlflow.log_param("seasonal_order", str(params["seasonal_order"]))
    mlflow.log_param("days_train", params["days_train"])
    mlflow.log_param("usar_exogena", params["usar_exogena"])

    max_train_rows = params["days_train"] * 24
    df_train_subset = df_train.iloc[-max_train_rows:]

    model = SARIMAX(
        df_train_subset["Demanda_MWh"],
        exog=df_train_subset[["Temperatura_C"]],
        order=tuple(params["order"]),
        seasonal_order=tuple(params["seasonal_order"]),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)
    predictions = model_fit.get_forecast(steps=params["test_hours"], exog=df_test[["Temperatura_C"]]).predicted_mean

    rmse_s = np.sqrt(mean_squared_error(df_test["Demanda_MWh"], predictions))
    mae_s = mean_absolute_error(df_test["Demanda_MWh"], predictions)
    mlflow.log_metric("RMSE", rmse_s)
    mlflow.log_metric("MAE", mae_s)

    # Guardar gráfico comparativo
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_test.index, df_test["Demanda_MWh"], label="Real")
    ax.plot(df_test.index, predictions, label="SARIMAX", linestyle="--")
    ax.plot(df_test.index, y_baseline, label="Baseline", linestyle=":")
    ax.set_title("Comparativa Real vs Predicciones")
    ax.legend()
    fig.savefig("comparacion_pronostico.png")
    mlflow.log_artifact("comparacion_pronostico.png")
    plt.close(fig)
    print(f"[MLflow] SARIMAX registrado. RMSE: {rmse_s:.2f}")

# Exportar a JSON
with open("metrics.json", "w") as f_m:
    json.dump({"baseline": {"RMSE": rmse_b, "MAE": mae_b}, "sarimax": {"RMSE": rmse_s, "MAE": mae_s}}, f_m, indent=4)
