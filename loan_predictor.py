import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.stApp { background-color: #0d0f14; color: #e8e6e0; }

.app-header { text-align:center; padding:2.5rem 0 1.5rem; border-bottom:1px solid #1e2330; margin-bottom:2rem; }
.app-header h1 { font-size:2.2rem; font-weight:800; color:#f0ede6; letter-spacing:-0.03em; margin:0; }
.app-header p  { color:#5a6070; font-size:0.78rem; margin-top:0.4rem; letter-spacing:0.08em; text-transform:uppercase; }

.section-label { font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700; letter-spacing:0.15em;
    text-transform:uppercase; color:#4a9eff; margin:1.8rem 0 0.6rem; padding-bottom:0.4rem; border-bottom:1px solid #1e2330; }

.result-approved { background:linear-gradient(135deg,#0a2818,#0d1f12); border:1px solid #1a5c32;
    border-left:4px solid #22c55e; border-radius:8px; padding:1.8rem 2rem; margin:1.5rem 0; }
.result-rejected { background:linear-gradient(135deg,#1f0a0a,#1a0d0d); border:1px solid #5c1a1a;
    border-left:4px solid #ef4444; border-radius:8px; padding:1.8rem 2rem; margin:1.5rem 0; }

.result-verdict { font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800; letter-spacing:-0.02em; margin:0 0 0.3rem; }
.result-approved .result-verdict { color:#22c55e; }
.result-rejected .result-verdict { color:#ef4444; }
.result-prob { font-size:0.82rem; color:#8a8f9a; margin:0; letter-spacing:0.04em; }

.prob-bar-wrap { margin:1.2rem 0 0.4rem; background:#1a1e28; border-radius:4px; height:6px; overflow:hidden; }
.prob-bar-fill-approved { height:100%; border-radius:4px; background:linear-gradient(90deg,#16a34a,#22c55e); }
.prob-bar-fill-rejected { height:100%; border-radius:4px; background:linear-gradient(90deg,#b91c1c,#ef4444); }
.prob-label { display:flex; justify-content:space-between; font-size:0.72rem; color:#5a6070; margin-top:0.3rem; }

.metric-strip { display:flex; gap:1rem; margin:1.2rem 0 0; }
.metric-box { flex:1; background:#12151e; border:1px solid #1e2330; border-radius:6px; padding:0.8rem 1rem; text-align:center; }
.metric-box .val { font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; color:#f0ede6; }
.metric-box .lbl { font-size:0.65rem; color:#5a6070; text-transform:uppercase; letter-spacing:0.08em; margin-top:0.15rem; }

.flag-row { display:flex; align-items:flex-start; gap:0.5rem; font-size:0.8rem;
    margin:0.35rem 0; padding:0.4rem 0.6rem; border-radius:4px; }
.flag-danger  { background:#1f0e0e; color:#f87171; }
.flag-warning { background:#1a1506; color:#fbbf24; }
.flag-good    { background:#0a1a10; color:#4ade80; }

.stButton > button { background:#4a9eff !important; color:#0d0f14 !important;
    font-family:'Syne',sans-serif !important; font-weight:700 !important; font-size:0.9rem !important;
    letter-spacing:0.05em !important; border:none !important; border-radius:6px !important;
    padding:0.7rem 2.5rem !important; width:100% !important; margin-top:1.2rem !important; }
.stButton > button:hover { opacity:0.85 !important; }

footer { visibility:hidden; }
#MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Preprocessing constants ───────────────────────────────────────────────────
LOG_COLS = ['annual_income', 'savings_assets', 'current_debt', 'loan_amount']
CAT_COLS = ['occupation_status', 'product_type', 'loan_intent']
TRAIN_COLUMNS = [
    'age', 'annual_income', 'credit_score', 'loan_amount',
    'loan_term', 'interest_rate', 'debt_to_income_ratio',
    'loan_to_income_ratio', 'savings_assets', 'current_debt',
    'defaults_on_file', 'derogatory_marks',
    'occupation_status_Part-Time', 'occupation_status_Self-Employed',
    'occupation_status_Unemployed',
    'product_type_Personal Loan', 'product_type_Student Loan',
    'loan_intent_Debt Consolidation', 'loan_intent_Education',
    'loan_intent_Home Improvement', 'loan_intent_Medical',
    'loan_intent_Venture'
]

def preprocess(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    for col in LOG_COLS:
        df[col] = np.log1p(df[col])
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)
    df = df.reindex(columns=TRAIN_COLUMNS, fill_value=0)
    return df

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists('xgb_model.pkl'):
        return None
    return joblib.load('xgb_model.pkl')

model = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🏦 Loan Approval Predictor</h1>
    <p>XGBoost · Loan Approval 2025 Dataset</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("**xgb_model.pkl not found.** Run Section 20a in the notebook first, then place `xgb_model.pkl` in the same folder as this script.")
    st.stop()

# ── Form ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">① Personal Information</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
with col2:
    occupation_status = st.selectbox("Occupation Status", ["Full-Time", "Part-Time", "Self-Employed", "Unemployed"])

st.markdown('<div class="section-label">② Financial Profile</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    annual_income  = st.number_input("Annual Income",    min_value=0.0, value=450000.0, step=1000.0)
    savings_assets = st.number_input("Savings & Assets", min_value=0.0, value=120000.0, step=1000.0)
    credit_score   = st.number_input("Credit Score",     min_value=300, max_value=850,  value=680)
with col2:
    current_debt     = st.number_input("Current Debt",      min_value=0.0, value=50000.0, step=1000.0)
    derogatory_marks = st.number_input("Derogatory Marks",  min_value=0,   max_value=10,  value=0)
    defaults_on_file = st.selectbox("Defaults on File",     ["No", "Yes"])

st.markdown('<div class="section-label">③ Loan Details</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    loan_amount   = st.number_input("Loan Amount",        min_value=1.0,  value=200000.0, step=1000.0)
    loan_term     = st.number_input("Loan Term (months)", min_value=1,    max_value=360,  value=36)
with col2:
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=12.5, step=0.1)
    product_type  = st.selectbox("Loan Product", ["Home Loan", "Personal Loan", "Student Loan"])

loan_intent = st.selectbox("Loan Purpose",
    ["Personal", "Home Improvement", "Debt Consolidation", "Education", "Medical", "Venture"])

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("RUN PREDICTION"):
    dti = current_debt / annual_income if annual_income > 0 else 0
    lti = loan_amount  / annual_income if annual_income > 0 else 0

    data = {
        'age': age, 'annual_income': annual_income, 'credit_score': credit_score,
        'loan_amount': loan_amount, 'loan_term': loan_term, 'interest_rate': interest_rate,
        'debt_to_income_ratio': dti, 'loan_to_income_ratio': lti,
        'savings_assets': savings_assets, 'current_debt': current_debt,
        'defaults_on_file': 1 if defaults_on_file == "Yes" else 0,
        'derogatory_marks': derogatory_marks,
        'occupation_status': occupation_status,
        'product_type': product_type,
        'loan_intent': loan_intent,
    }

    X    = preprocess(data)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    conf = max(prob, 1 - prob)
    conf_label = "HIGH" if conf > 0.80 else "MODERATE" if conf > 0.60 else "LOW"
    approved   = pred == 1

    card_class = "result-approved" if approved else "result-rejected"
    verdict    = "✅ APPROVED"     if approved else "❌ REJECTED"
    bar_class  = "prob-bar-fill-approved" if approved else "prob-bar-fill-rejected"

    st.markdown(f"""
    <div class="{card_class}">
        <p class="result-verdict">{verdict}</p>
        <p class="result-prob">Approval probability · {prob*100:.1f}%</p>
        <div class="prob-bar-wrap">
            <div class="{bar_class}" style="width:{prob*100:.1f}%"></div>
        </div>
        <div class="prob-label"><span>0%</span><span>50%</span><span>100%</span></div>
        <div class="metric-strip">
            <div class="metric-box"><div class="val">{prob*100:.1f}%</div><div class="lbl">Approval Prob</div></div>
            <div class="metric-box"><div class="val">{conf*100:.1f}%</div><div class="lbl">Confidence · {conf_label}</div></div>
            <div class="metric-box"><div class="val">{dti:.2f}</div><div class="lbl">Debt-to-Income</div></div>
            <div class="metric-box"><div class="val">{lti:.2f}</div><div class="lbl">Loan-to-Income</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Risk Factors</div>', unsafe_allow_html=True)
    flags = []
    if defaults_on_file == "Yes":
        flags.append(("danger",  "⚠", "Defaults on file — strong rejection signal"))
    if credit_score < 620:
        flags.append(("danger",  "⚠", f"Low credit score ({credit_score}) — below 620 threshold"))
    elif credit_score >= 700:
        flags.append(("good",    "✔", f"Strong credit score ({credit_score})"))
    if dti > 0.35:
        flags.append(("danger",  "⚠", f"High debt-to-income ratio ({dti:.2f}) — above 0.35"))
    elif dti < 0.25:
        flags.append(("good",    "✔", f"Healthy debt-to-income ratio ({dti:.2f})"))
    else:
        flags.append(("warning", "·", f"Moderate debt-to-income ratio ({dti:.2f})"))
    if derogatory_marks > 2:
        flags.append(("warning", "⚠", f"{derogatory_marks} derogatory marks on record"))
    if loan_intent == "Debt Consolidation":
        flags.append(("warning", "·", "Debt Consolidation has the lowest approval rate (36.6%)"))
    elif loan_intent == "Education":
        flags.append(("good",    "✔", "Education loans have the highest approval rate (67.5%)"))
    if not flags:
        flags.append(("good", "✔", "No major risk flags detected"))

    for kind, icon, msg in flags:
        st.markdown(f'<div class="flag-row flag-{kind}">{icon}&nbsp;&nbsp;{msg}</div>', unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size:0.68rem;color:#3a3f4a;margin-top:1.5rem;text-align:center;">
        Model: XGBoost (tuned via RandomizedSearchCV) &nbsp;·&nbsp; Loan Approval 2025
    </p>""", unsafe_allow_html=True)
