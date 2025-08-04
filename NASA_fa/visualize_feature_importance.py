import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Load model and data
model = joblib.load("models/rul_model.pkl")
data = pd.read_csv("C:/Project/NASA_fa/train_FD001_cleaned.csv")

# Extract features (remove target columns if present)
features = data.drop(columns=["RUL", "engine_id", "cycle"], errors="ignore")

# Get feature importances
importances = model.feature_importances_
feature_names = features.columns

# Create DataFrame for sorting and plotting
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Plot top 10 features
top_n = 10
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:top_n][::-1], importance_df['Importance'][:top_n][::-1])
plt.xlabel("Importance")
plt.title(f"Top {top_n} Important Features")
plt.tight_layout()

# ✅ Ensure 'assets' folder exists before saving
os.makedirs("assets", exist_ok=True)

# Save the figure
plt.savefig("assets/feature_importance.png")
plt.show()
