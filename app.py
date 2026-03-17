import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="",
    layout="wide"
)

# ── Load model ────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("churn_model.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

model, feature_names = load_model()

# ── Header ────────────────────────────────────────────
st.title(" Customer Churn Predictor")
st.markdown("**Predict whether a customer is likely to churn based on their profile.**")
st.markdown("*Model: XGBoost | ROC-AUC: 0.8427 | Dataset: Telco Customer Churn (7,043 records)*")
st.divider()

# ── Sidebar inputs ────────────────────────────────────
st.sidebar.header("Customer Profile")
st.sidebar.markdown("Fill in the customer details below:")

gender           = st.sidebar.selectbox("Gender",           ["Male", "Female"])
senior_citizen   = st.sidebar.selectbox("Senior Citizen",   ["No", "Yes"])
partner          = st.sidebar.selectbox("Has Partner",       ["No", "Yes"])
dependents       = st.sidebar.selectbox("Has Dependents",    ["No", "Yes"])
tenure           = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service    = st.sidebar.selectbox("Phone Service",     ["No", "Yes"])
multiple_lines   = st.sidebar.selectbox("Multiple Lines",    ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service",  ["DSL", "Fiber optic", "No"])
online_security  = st.sidebar.selectbox("Online Security",   ["No", "Yes", "No internet service"])
online_backup    = st.sidebar.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
device_protection= st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support     = st.sidebar.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
streaming_tv     = st.sidebar.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])
contract         = st.sidebar.selectbox("Contract Type",     ["Month-to-month", "One year", "Two year"])
paperless        = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method   = st.sidebar.selectbox("Payment Method",    [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges  = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
total_charges    = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, monthly_charges * tenure)

# ── Encode inputs ─────────────────────────────────────
def encode(val, mapping):
    return mapping.get(val, 0)

yes_no     = {"No": 0, "Yes": 1}
gender_map = {"Female": 0, "Male": 1}

input_data = {
    "gender":            encode(gender, gender_map),
    "SeniorCitizen":     encode(senior_citizen, yes_no),
    "Partner":           encode(partner, yes_no),
    "Dependents":        encode(dependents, yes_no),
    "tenure":            tenure,
    "PhoneService":      encode(phone_service, yes_no),
    "MultipleLines":     encode(multiple_lines, {"No": 0, "No phone service": 1, "Yes": 2}),
    "InternetService":   encode(internet_service, {"DSL": 0, "Fiber optic": 1, "No": 2}),
    "OnlineSecurity":    encode(online_security, {"No": 0, "No internet service": 1, "Yes": 2}),
    "OnlineBackup":      encode(online_backup, {"No": 0, "No internet service": 1, "Yes": 2}),
    "DeviceProtection":  encode(device_protection, {"No": 0, "No internet service": 1, "Yes": 2}),
    "TechSupport":       encode(tech_support, {"No": 0, "No internet service": 1, "Yes": 2}),
    "StreamingTV":       encode(streaming_tv, {"No": 0, "No internet service": 1, "Yes": 2}),
    "StreamingMovies":   encode(streaming_movies, {"No": 0, "No internet service": 1, "Yes": 2}),
    "Contract":          encode(contract, {"Month-to-month": 0, "One year": 1, "Two year": 2}),
    "PaperlessBilling":  encode(paperless, yes_no),
    "PaymentMethod":     encode(payment_method, {
        "Bank transfer (automatic)": 0, "Credit card (automatic)": 1,
        "Electronic check": 2, "Mailed check": 3
    }),
    "MonthlyCharges":    monthly_charges,
    "TotalCharges":      total_charges,
}

input_df = pd.DataFrame([input_data])[feature_names]

# ── Prediction ────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer Summary")
    st.write(f"**Contract:** {contract}")
    st.write(f"**Tenure:** {tenure} months")
    st.write(f"**Monthly Charges:** ${monthly_charges:.2f}")
    st.write(f"**Internet Service:** {internet_service}")
    st.write(f"**Tech Support:** {tech_support}")

with col2:
    st.subheader("Prediction")
    prob        = model.predict_proba(input_df)[0][1]
    prediction  = model.predict(input_df)[0]

    if prediction == 1:
        st.error(f" **HIGH CHURN RISK**")
        st.metric("Churn Probability", f"{prob*100:.1f}%")
        st.markdown("This customer is **likely to churn**. Consider retention strategies.")
    else:
        st.success(f" **LOW CHURN RISK**")
        st.metric("Churn Probability", f"{prob*100:.1f}%")
        st.markdown("This customer is **likely to stay**.")

    st.progress(float(prob))

with col3:
    st.subheader("Key Risk Factors")
    if contract == "Month-to-month":
        st.warning(" Month-to-month contract — highest churn risk (42.7%)")
    if internet_service == "Fiber optic":
        st.warning(" Fiber optic customer — elevated churn risk (41.9%)")
    if tech_support == "No":
        st.warning(" No tech support — increases churn likelihood")
    if tenure < 12:
        st.warning(" Low tenure — new customers churn more")
    if monthly_charges > 80:
        st.warning(" High monthly charges — financial pressure risk")

# ── Model info ────────────────────────────────────────
st.divider()
st.markdown("### Model Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Best Model",  "XGBoost")
m2.metric("ROC-AUC",     "0.8427")
m3.metric("Accuracy",    "74.88%")
m4.metric("Models Compared", "3")