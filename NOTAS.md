# Notas de Entrega - Taller Semana 1 (Pronóstico de Demanda Eléctrica)

Este archivo recopila las verificaciones de calidad, metodologías y evidencias del pipeline de datos, entrenamiento y trazabilidad de acuerdo con las directrices del taller.

---

## 1. Control de Calidad y Verificaciones Obligatorias

### A. Prevención de Fuga de Información (Data Leakage)
* **Verificación**: Comprobamos que el corte de datos entre entrenamiento y prueba se realiza de manera estrictamente temporal/cronológica empleando índices temporales ordenados.
* **Evidencia**: En el script `prepare.py`, la división utiliza segmentación indexada con `.iloc[:-test_hours]` y `.iloc[-test_hours:]` tras asegurar el orden de la serie temporal mediante `.sort_index()`. Ninguna función de escalado, imputación o cálculo de parámetros utiliza datos del tramo de prueba (el cual corresponde exactamente a las últimas 48 horas de la serie).

### B. Recálculo de Métricas a Mano (Consistencia de MLflow)
* **Verificación**: Recalculamos el error cuadrático medio (RMSE) de forma manual utilizando Python/Numpy para certificar que el valor subido al servidor remoto de MLflow coincide plenamente con los cálculos del script.
* **Evidencia**: 
  * **Fórmula de cálculo**: $RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$
  * **Valores manuales calculados**:
    * **Baseline Naive (Estacional 24h)**: RMSE = `8169.83` | MAE = `8096.34`
    * **Modelo SARIMAX**: RMSE = `7949.46` | MAE = `7891.07`
  * Ambos resultados coinciden de forma exacta (con precisión de decimales) con el archivo de salida `metrics.json` y el panel de control remoto de MLflow.

### C. Trazabilidad e Inserción del Conjunto de Datos
* **Verificación**: Garantizar que los conjuntos de datos de gran tamaño no entren al repositorio de Git, quedando delegados al control de versiones de datos.
* **Evidencia**: El archivo `.gitignore` local fue configurado para proteger y omitir de forma estricta los archivos `xm_demanda.csv`, `train.csv` y `test.csv`, permitiendo que solo los artefactos y códigos fuente de ejecución (`prepare.py`, `train.py`, `params.yaml`, `metrics.json`) sean controlados por Git, preservando así la integridad del almacenamiento del repositorio.

---

## 2. Versiones de Datos Registrados (DVC Hashes)

* **Dataset Versión 1 (Snapshot Inicial de XM)**: 
  * MD5 / DVC Hash: `b3644fa91509b1bfa73333b18fccb703` 
* **Dataset Versión 2 (Con actualización de Temperatura Exógena Open-Meteo)**:
  * MD5 / DVC Hash: `86ee63a3dcfaf29b3e655fb20177c8bc` 
