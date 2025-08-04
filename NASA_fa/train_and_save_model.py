import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import joblib

# Load cleaned data
df = pd.read_csv(r"C:\Project\NASAp\data\train_FD001_cleaned.csv")

# Drop unnecessary columns and split features/target
X = df.drop(['RUL', 'engine_id', 'cycle'], axis=1)
y = df['RUL']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define hyperparameter grid
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize base model
base_model = RandomForestRegressor(random_state=42)

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    verbose=2,
    n_jobs=-1,
    scoring='neg_mean_squared_error'
)

# Train the model
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_
print(f"[INFO] Best Parameters: {random_search.best_params_}")

# Evaluate model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"[INFO] Tuned Model MSE: {mse:.2f}")

# Save model
joblib.dump(best_model, "models/rul_model.pkl")
print("[INFO] Best model saved to models/rul_model.pkl")
