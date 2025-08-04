# train_xgboost.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# --- Load Your Processed Data ---
df = pd.read_csv("data/processed_engine_data.csv")  
X = df.drop("RUL", axis=1)
y = df["RUL"]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train XGBoost Model ---
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)

# --- Evaluate Model ---
y_pred = xgb.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"XGBoost RMSE: {rmse:.2f}")
print(f"XGBoost MAE: {mae:.2f}")
print(f"XGBoost R² Score: {r2:.2f}")

# --- Save Model ---
joblib.dump(xgb, "models/xgb_model.pkl")
print("✅ XGBoost model saved to models/xgb_model.pkl")
