# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="Sales Prediction Dashboard", page_icon="📈", layout="centered")

# Load saved models
@st.cache_resource
def load_artifacts():
    with open("artifacts/sales_model.pkl", "rb") as m_file:
        model = pickle.load(m_file)
    with open("artifacts/scaler.pkl", "rb") as s_file:
        scaler = pickle.load(s_file)
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("❌ Model files not found! Please run `python train.py` first to generate model artifacts.")
    st.stop()

# Header Section
st.title("📈 Advertising Sales Prediction App")
st.markdown("Predict product units sales based on advertising expenditure across different platforms.")

st.markdown("---")

# Layout Configuration: Input columns
st.subheader("🛠️ Step 1: Enter Advertising Budgets (in \$1,000s)")
col1, col2 = st.columns(2)

with col1:
    tv = st.number_input("TV Advertising Budget", min_value=0.0, max_value=500.0, value=150.0, step=5.0)
    radio = st.number_input("Radio Advertising Budget", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
    newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0, max_value=150.0, value=30.0, step=2.0)

with col2:
    digital = st.number_input("Digital Ads Budget", min_value=0.0, max_value=300.0, value=100.0, step=5.0)
    social_media = st.number_input("Social Media Budget", min_value=0.0, max_value=200.0, value=50.0, step=5.0)

# Build summary dataframe
input_data = pd.DataFrame({
    'TV': [tv],
    'Radio': [radio],
    'Newspaper': [newspaper],
    'Digital': [digital],
    'Social_Media': [social_media]
})

# Display Input Summary Table
st.markdown("### 📋 Input Budget Summary")
st.dataframe(input_data, use_container_width=True)

# Calculation Trigger Button
st.markdown("---")
if st.button("🔮 Predict Sales Performance", type="primary"):
    
    # Preprocess inputs using the fitted scaler
    scaled_inputs = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(scaled_inputs)[0]
    
    # Output metrics
    st.subheader("🚀 Predicted Target Outcome")
    st.metric(label="Estimated Sales Volume (Units)", value=f"{max(0.0, prediction):.2f}K units")
    
    # Quick alert indicator
    if prediction > 15.0:
        st.success("🎉 High performance yield predicted based on this budget distribution scheme!")
    else:
        st.warning("⚠️ Conservative sales output predicted. Consider reallocating budget to high-performing channels.")
