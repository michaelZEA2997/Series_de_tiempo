import pandas as pd
import yaml

# Leer parámetros
with open("params.yaml", "r") as f_yaml:
    params = yaml.safe_load(f_yaml)

# Leer datos
df = pd.read_csv("xm_demanda.csv", parse_dates=[0], index_col=0).sort_index()

# Calcular corte temporal
test_hours = params["test_hours"]

# Segmentar train/test sin mezclar fechas
df_train = df.iloc[:-test_hours]
df_test = df.iloc[-test_hours:]

# Guardar sets procesados
df_train.to_csv("train.csv")
df_test.to_csv("test.csv")

print(f"Preparación de datos lista. Train: {len(df_train)} filas, Test: {len(df_test)} filas.")
