import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Preprocessing constants ───────────────────────────────────────────────────
LOG_COLS = ['annual_income', 'savings_assets', 'current_debt', 'loan_amount']
CAT_COLS = ['occupation_status', 'product_type', 'loan_intent']

def preprocess(data: dict, feature_columns: list) -> pd.DataFrame:
    df = pd.DataFrame([data])
    for col in LOG_COLS:
        df[col] = np.log1p(df[col])
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

# ── Load model + feature columns ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not os.path.exists('xgb_model.pkl'):
        return None, None
    if not os.path.exists('feature_columns.pkl'):
        return None, None
    model   = joblib.load('xgb_model.pkl')
    columns = joblib.load('feature_columns.pkl')
    return model, columns

model, feature_columns = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏦 Loan Approval Predictor")
st.caption("XGBoost · Loan Approval 2025 Dataset")
st.divider()

if model is None or feature_columns is None:
    st.error("""
    **Missing required files.** Make sure both are in your GitHub repo:
    - `xgb_model.pkl`
    - `feature_columns.pkl`
    """)
    st.stop()

# ── Form ──────────────────────────────────────────────────────────────────────
st.subheader("① Personal Information")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
with col2:
    years_employed = st.number_input("Years Employed", min_value=0, max_value=50, value=5)
with col3:
    credit_history_years = st.number_input("Credit History (years)", min_value=0, max_value=50, value=8)

occupation_status = st.selectbox("Occupation Status",
    ["Full-Time", "Part-Time", "Self-Employed", "Student", "Unemployed"])

st.subheader("② Financial Profile")
col1, col2 = st.columns(2)
with col1:
    annual_income        = st.number_input("Annual Income",       min_value=0.0,  value=450000.0, step=1000.0)
    savings_assets       = st.number_input("Savings & Assets",    min_value=0.0,  value=120000.0, step=1000.0)
    credit_score         = st.number_input("Credit Score",        min_value=300,  max_value=850,  value=680)
with col2:
    current_debt         = st.number_input("Current Debt",        min_value=0.0,  value=50000.0,  step=1000.0)
    derogatory_marks     = st.number_input("Derogatory Marks",    min_value=0,    max_value=10,   value=0)
    delinquencies        = st.number_input("Delinquencies (last 2 yrs)", min_value=0, max_value=20, value=0)

defaults_on_file = st.selectbox("Defaults on File", ["No", "Yes"])

st.subheader("③ Loan Details")
col1, col2 = st.columns(2)
with col1:
    loan_amount   = st.number_input("Loan Amount",        min_value=1.0,   value=200000.0, step=1000.0)
    interest_rate = st.number_input("Interest Rate (%)",  min_value=0.0,   max_value=100.0, value=12.5, step=0.1)
with col2:
    product_type  = st.selectbox("Loan Product", ["Home Loan", "Line of Credit", "Personal Loan"])

loan_intent = st.selectbox("Loan Purpose",
    ["Home Improvement", "Debt Consolidation", "Education", "Medical", "Personal", "Venture"])

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("RUN PREDICTION", use_container_width=True, type="primary"):
    dti = current_debt / annual_income if annual_income > 0 else 0
    lti = loan_amount  / annual_income if annual_income > 0 else 0

    data = {
        'age':                   age,
        'years_employed':        years_employed,
        'annual_income':         annual_income,
        'credit_score':          credit_score,
        'credit_history_years':  credit_history_years,
        'savings_assets':        savings_assets,
        'current_debt':          current_debt,
        'defaults_on_file':      1 if defaults_on_file == "Yes" else 0,
        'delinquencies_last_2yrs': delinquencies,
        'derogatory_marks':      derogatory_marks,
        'loan_amount':           loan_amount,
        'interest_rate':         interest_rate,
        'debt_to_income_ratio':  dti,
        'loan_to_income_ratio':  lti,
        'occupation_status':     occupation_status,
        'product_type':          product_type,
        'loan_intent':           loan_intent,
    }

    try:
        X    = preprocess(data, feature_columns)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]
        conf = max(prob, 1 - prob)
        conf_label = "HIGH" if conf > 0.80 else "MODERATE" if conf > 0.60 else "LOW"

        if pred == 1:
            st.success("## ✅ APPROVED")
        else:
            st.error("## ❌ REJECTED")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Approval Prob",  f"{prob*100:.1f}%")
        col2.metric("Confidence",     f"{conf*100:.1f}% ({conf_label})")
        col3.metric("Debt-to-Income", f"{dti:.2f}")
        col4.metric("Loan-to-Income", f"{lti:.2f}")

        st.progress(float(prob), text=f"Approval probability: {prob*100:.1f}%")

        st.subheader("Risk Factors")
        if defaults_on_file == "Yes":
            st.error("⚠ Defaults on file — strong rejection signal")
        if credit_score < 620:
            st.error(f"⚠ Low credit score ({credit_score}) — below 620 threshold")
        elif credit_score >= 700:
            st.success(f"✔ Strong credit score ({credit_score})")
        if dti > 0.35:
            st.error(f"⚠ High debt-to-income ratio ({dti:.2f}) — above 0.35")
        elif dti < 0.25:
            st.success(f"✔ Healthy debt-to-income ratio ({dti:.2f})")
        else:
            st.warning(f"· Moderate debt-to-income ratio ({dti:.2f})")
        if derogatory_marks > 2:
            st.warning(f"⚠ {derogatory_marks} derogatory marks on record")
        if delinquencies > 0:
            st.warning(f"⚠ {delinquencies} delinquencies in last 2 years")
        if loan_intent == "Debt Consolidation":
            st.warning("· Debt Consolidation has the lowest approval rate (36.6%)")
        elif loan_intent == "Education":
            st.success("✔ Education loans have the highest approval rate (67.5%)")

        st.caption("Model: XGBoost (tuned via RandomizedSearchCV) · Loan Approval 2025")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info("Check that feature_columns.pkl matches the model's training columns.")
