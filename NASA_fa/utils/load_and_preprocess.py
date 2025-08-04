import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Expected columns from your cleaned file
EXPECTED_COLUMNS = [
    'engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_measurement_2', 'sensor_measurement_3', 'sensor_measurement_4',
    'sensor_measurement_6', 'sensor_measurement_7', 'sensor_measurement_8',
    'sensor_measurement_9', 'sensor_measurement_11', 'sensor_measurement_12',
    'sensor_measurement_13', 'sensor_measurement_14', 'sensor_measurement_15',
    'sensor_measurement_16', 'sensor_measurement_17', 'sensor_measurement_18',
    'sensor_measurement_19', 'sensor_measurement_20', 'sensor_measurement_21',
    'RUL'
]

def load_and_preprocess_data(df):
    # Check for correct columns
    if list(df.columns) != EXPECTED_COLUMNS:
        print("❌ Uploaded columns do not match expected format.")
        print("📊 Uploaded CSV Columns:")
        print(list(df.columns))
        return None

    # Copy the DataFrame
    df_copy = df.copy()

    # Features to normalize (excluding engine_id, cycle, RUL)
    features_to_scale = df_copy.columns.difference(['engine_id', 'cycle', 'RUL'])

    # Normalize features
    scaler = MinMaxScaler()
    df_copy[features_to_scale] = scaler.fit_transform(df_copy[features_to_scale])

    return df_copy
