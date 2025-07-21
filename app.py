import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="NASA Engine Monitor", layout="centered")

# --- Title ---
st.markdown("""
<style>
h1 {
    text-align: center;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 NASA Turbofan Engine Health Monitor")

st.markdown("""
Welcome to the NASA Engine Health Monitor — powered by Random Forest and XGBoost.  
This tool analyzes turbofan sensor data to detect potential anomalies and predict engine condition.
""")


# --- Tabs ---
tab1, tab2 = st.tabs(["🧠 Predict & Detect", "📊 Feature Importance"])

# --- Load Models ---
@st.cache_resource
def load_models():
    rf = joblib.load("random_forest_model.pkl")
    xgb = joblib.load("xgboost_model.pkl")
    return rf, xgb

rf_model, xgb_model = load_models()

# --- Feature Names (same as training) ---
features = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_6', 'sensor_7',
            'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13']

# ---------------- TAB 1 ------------------
with tab1:
    # --- Model Selection ---
    model_choice = st.selectbox("Select a model", ["Random Forest", "XGBoost"])
    model = rf_model if model_choice == "Random Forest" else xgb_model

    # --- File Upload ---
    st.header("📂 Upload Engine Sensor Data (CSV)")
    uploaded_file = st.file_uploader("Choose a file", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully.")
            st.dataframe(data.head())

            # --- Predict Button ---
            if st.button("🔍 Detect Anomalies"):
                predictions = model.predict(data)
                data["Prediction"] = predictions

                # 📊 Summary Metrics
                avg_pred = data["Prediction"].mean()

                col1, col2 = st.columns(2)
                col1.metric("🔮 Avg Anomaly Score", f"{avg_pred:.2f}")
                if avg_pred < 0.3:
                    col2.success("🟢 Healthy")
                elif avg_pred < 0.7:
                    col2.warning("🟠 Warning")
                else:
                    col2.error("🔴 Critical")

                # 📄 Detailed Table
                st.markdown("### 📋 Prediction Results")
                st.dataframe(data)


                # --- Prediction Summary Chart ---
                prediction_counts = data["Prediction"].value_counts()
                st.subheader("📊 Anomaly Prediction Summary")
                fig, ax = plt.subplots()
                ax.bar(prediction_counts.index.astype(str), prediction_counts.values, color=["green", "red"])
                ax.set_xlabel("Prediction")
                ax.set_ylabel("Count")
                ax.set_title("Normal (0) vs Anomalies (1)")
                st.pyplot(fig)

                # --- Download Button ---
                csv = data.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv,
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )

                # --- Accuracy Info ---
                if model_choice == "Random Forest":
                    st.info("✅ Random Forest Accuracy: 100% (CV)")
                elif model_choice == "XGBoost":
                    st.info("✅ XGBoost Accuracy: 99.87% (CV)")

                # --- Health Status (Based on Mean Prediction) ---
                if "Prediction" in data.columns:
                    avg_pred = data["Prediction"].mean()
                    st.subheader("🩺 Engine Health Status")
                    if avg_pred < 0.3:
                        st.success("🟢 Engine Health: Healthy")
                    elif avg_pred < 0.7:
                        st.warning("🟠 Engine Health: Warning")
                    else:
                        st.error("🔴 Engine Health: Critical")

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

    # --- Manual Input Placeholder ---
    st.header("✍️ Or Enter Sensor Readings Manually (Coming Soon)")
    st.info("Manual entry will be added in the next version.")

# ---------------- TAB 2 ------------------
with tab2:
    st.header("📊 Feature Importance")

    model_option = st.radio("Choose a model to view feature importance", ["Random Forest", "XGBoost"])
    importances = rf_model.feature_importances_ if model_option == "Random Forest" else xgb_model.feature_importances_

    # 🔍 Dynamically generate feature names to match model input
    feature_names = [f"Feature {i+1}" for i in range(len(importances))]

    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by="Importance", ascending=False)

    # 📊 Plot
    fig, ax = plt.subplots()
    ax.barh(feature_df['Feature'], feature_df['Importance'], color='skyblue')
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_option} Feature Importance")
    ax.invert_yaxis()
    st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.caption("Made    by Lipsha Dash | [GitHub](https://github.com/Lipsha-Dash) | Dataset: NASA CMAPS")
