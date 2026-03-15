"""
╔══════════════════════════════════════════════════════╗
║       🏦  Loan Approval Predictor — XGBoost          ║
║       Trained on Loan_approval_data_2025.csv         ║
╚══════════════════════════════════════════════════════╝

Run this script AFTER executing the notebook to use the
trained XGBoost model. It replicates the exact same
preprocessing pipeline used during training.

Usage:
    python loan_predictor.py
"""

import sys
import os
import numpy as np
import pandas as pd

# ── Terminal colours ───────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    BG_GREEN = "\033[42m"
    BG_RED   = "\033[41m"
    BG_YELLOW = "\033[43m"

def clear(): os.system('cls' if os.name == 'nt' else 'clear')

def banner():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════════╗
║          🏦  LOAN APPROVAL PREDICTOR  ·  XGBoost             ║
║          Trained on Loan Approval 2025 Dataset               ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
""")

def divider(char="─", width=64):
    print(f"{C.DIM}{char * width}{C.RESET}")

def section(title):
    print(f"\n{C.CYAN}{C.BOLD}  {title}{C.RESET}")
    divider()

def prompt(label, hint=""):
    hint_str = f" {C.DIM}({hint}){C.RESET}" if hint else ""
    return input(f"  {C.WHITE}{label}{C.RESET}{hint_str}: ").strip()

def error(msg):
    print(f"  {C.RED}✖  {msg}{C.RESET}")

def success(msg):
    print(f"  {C.GREEN}✔  {msg}{C.RESET}")

# ── Preprocessing — mirrors the notebook pipeline exactly ─────────────────────
LOG_COLS = ['annual_income', 'savings_assets', 'current_debt', 'loan_amount']
CAT_COLS = ['occupation_status', 'product_type', 'loan_intent']

# These are the exact OHE columns produced by pd.get_dummies(..., drop_first=True)
# on the training set — must match X_train_enc.columns
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
    """Apply the same preprocessing pipeline as the notebook."""
    df = pd.DataFrame([data])

    # 1. Log1p transform skewed features
    for col in LOG_COLS:
        df[col] = np.log1p(df[col])

    # 2. One-hot encode categorical columns
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)

    # 3. Align to training columns (fill any missing OHE cols with 0)
    df = df.reindex(columns=TRAIN_COLUMNS, fill_value=0)

    return df

# ── Input helpers ─────────────────────────────────────────────────────────────

def get_float(label, hint="", min_val=None, max_val=None):
    while True:
        try:
            val = float(prompt(label, hint))
            if min_val is not None and val < min_val:
                error(f"Must be ≥ {min_val}"); continue
            if max_val is not None and val > max_val:
                error(f"Must be ≤ {max_val}"); continue
            return val
        except ValueError:
            error("Please enter a valid number.")

def get_int(label, hint="", min_val=None, max_val=None):
    while True:
        try:
            val = int(prompt(label, hint))
            if min_val is not None and val < min_val:
                error(f"Must be ≥ {min_val}"); continue
            if max_val is not None and val > max_val:
                error(f"Must be ≤ {max_val}"); continue
            return val
        except ValueError:
            error("Please enter a whole number.")

def get_choice(label, options: list):
    """Display numbered menu and return chosen value."""
    print(f"\n  {C.WHITE}{label}{C.RESET}")
    for i, opt in enumerate(options, 1):
        print(f"    {C.CYAN}{i}{C.RESET}. {opt}")
    while True:
        try:
            choice = int(prompt("Enter number"))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            error(f"Enter a number between 1 and {len(options)}.")
        except ValueError:
            error("Please enter a number.")

def get_binary(label):
    """Return 1 or 0 for yes/no questions."""
    while True:
        val = prompt(label, "y/n").lower()
        if val in ('y', 'yes', '1'): return 1
        if val in ('n', 'no',  '0'): return 0
        error("Enter y or n.")

# ── Collect applicant data ────────────────────────────────────────────────────

def collect_inputs():
    data = {}

    section("① PERSONAL INFORMATION")
    data['age'] = get_int("Age", "18–80", 18, 80)

    data['occupation_status'] = get_choice(
        "Occupation Status",
        ["Full-Time", "Part-Time", "Self-Employed", "Unemployed"]
    )

    section("② FINANCIAL PROFILE")
    data['annual_income']  = get_float("Annual Income (₱ or local currency)", "e.g. 450000", 0)
    data['savings_assets'] = get_float("Savings & Assets",                    "e.g. 120000", 0)
    data['current_debt']   = get_float("Current Debt",                        "e.g. 50000",  0)
    data['credit_score']   = get_int(  "Credit Score",                        "300–850", 300, 850)
    data['defaults_on_file']   = get_binary("Any defaults on file?")
    data['derogatory_marks']   = get_int("Number of derogatory marks", "0–10", 0, 10)

    section("③ LOAN DETAILS")
    data['loan_amount']  = get_float("Loan Amount Requested", "e.g. 200000", 1)
    data['loan_term']    = get_int(  "Loan Term",             "months, e.g. 36", 1, 360)
    data['interest_rate']= get_float("Interest Rate (%)",     "e.g. 12.5", 0, 100)

    data['product_type'] = get_choice(
        "Loan Product Type",
        ["Home Loan", "Personal Loan", "Student Loan"]
    )
    data['loan_intent'] = get_choice(
        "Loan Purpose / Intent",
        ["Home Improvement", "Debt Consolidation", "Education",
         "Medical", "Personal", "Venture"]
    )

    section("④ DERIVED RATIOS  (auto-calculated)")
    # Compute ratios — same as notebook features
    monthly_income  = data['annual_income'] / 12
    monthly_payment = (data['loan_amount'] * (data['interest_rate']/100/12)) / \
                      (1 - (1 + data['interest_rate']/100/12) ** (-data['loan_term'])) \
                      if data['interest_rate'] > 0 else data['loan_amount'] / data['loan_term']

    data['debt_to_income_ratio']  = data['current_debt'] / data['annual_income'] \
                                    if data['annual_income'] > 0 else 0
    data['loan_to_income_ratio']  = data['loan_amount']  / data['annual_income'] \
                                    if data['annual_income'] > 0 else 0

    print(f"\n  {C.DIM}Debt-to-Income Ratio :  {data['debt_to_income_ratio']:.4f}")
    print(f"  Loan-to-Income Ratio :  {data['loan_to_income_ratio']:.4f}{C.RESET}")

    return data

# ── Result display ─────────────────────────────────────────────────────────────

def show_result(prediction, probability, data):
    approved   = prediction == 1
    conf       = probability if approved else (1 - probability)
    prob_pct   = probability * 100

    print(f"\n{C.BOLD}")
    divider("═")
    print(f"  🏦  LOAN DECISION RESULT")
    divider("═")
    print(C.RESET)

    if approved:
        verdict = f"{C.BG_GREEN}{C.BOLD}  ✅  APPROVED  {C.RESET}"
        conf_color = C.GREEN
    else:
        verdict = f"{C.BG_RED}{C.BOLD}  ❌  REJECTED  {C.RESET}"
        conf_color = C.RED

    print(f"  Verdict      :  {verdict}")
    print(f"  Approval Prob:  {conf_color}{C.BOLD}{prob_pct:.1f}%{C.RESET}")
    print(f"  Confidence   :  {conf_color}{conf*100:.1f}%{C.RESET}  ({'HIGH' if conf > 0.8 else 'MODERATE' if conf > 0.6 else 'LOW'})")

    # Risk factors
    print(f"\n  {C.BOLD}Key factors in this decision:{C.RESET}")
    flags = []
    if data['defaults_on_file']:
        flags.append(f"  {C.RED}⚠  Defaults on file detected{C.RESET}")
    if data['credit_score'] < 620:
        flags.append(f"  {C.RED}⚠  Credit score below 620 ({data['credit_score']}){C.RESET}")
    elif data['credit_score'] >= 700:
        flags.append(f"  {C.GREEN}✔  Strong credit score ({data['credit_score']}){C.RESET}")
    dti = data['debt_to_income_ratio']
    if dti > 0.35:
        flags.append(f"  {C.RED}⚠  High debt-to-income ratio ({dti:.2f}){C.RESET}")
    elif dti < 0.25:
        flags.append(f"  {C.GREEN}✔  Healthy debt-to-income ratio ({dti:.2f}){C.RESET}")
    if data['derogatory_marks'] > 2:
        flags.append(f"  {C.YELLOW}⚠  {data['derogatory_marks']} derogatory marks{C.RESET}")
    if not flags:
        flags.append(f"  {C.DIM}No major risk flags detected.{C.RESET}")

    for f in flags:
        print(f)

    divider("═")
    print(f"  {C.DIM}Model: XGBoost (tuned) · Loan Approval 2025{C.RESET}\n")

# ── Load model ────────────────────────────────────────────────────────────────

def load_model():
    """Try to load xgb_model from a saved .pkl, else guide user to export it."""
    try:
        import joblib
        model = joblib.load('xgb_model.pkl')
        success("XGBoost model loaded from xgb_model.pkl")
        return model
    except FileNotFoundError:
        pass

    # Try pickle
    try:
        import pickle
        with open('xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        success("XGBoost model loaded.")
        return model
    except FileNotFoundError:
        pass

    # Model file not found — guide user
    print(f"""
{C.YELLOW}{C.BOLD}  ⚠  xgb_model.pkl not found.{C.RESET}

  Add this cell at the end of your notebook and run it once:

{C.CYAN}  import joblib
  joblib.dump(xgb_model, 'xgb_model.pkl')
  print("✅ Model saved to xgb_model.pkl"){C.RESET}

  Then place xgb_model.pkl in the same folder as this script.
""")
    sys.exit(1)

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    clear()
    banner()
    print(f"  {C.DIM}This console uses the XGBoost model trained in your notebook.")
    print(f"  Preprocessing mirrors the notebook pipeline exactly.{C.RESET}\n")

    model = load_model()

    while True:
        try:
            data    = collect_inputs()
            X       = preprocess(data)
            pred    = model.predict(X)[0]
            prob    = model.predict_proba(X)[0][1]
            show_result(pred, prob, data)

        except KeyboardInterrupt:
            print(f"\n\n  {C.DIM}Exiting. Goodbye.{C.RESET}\n")
            break

        print(f"\n  {C.CYAN}Run another prediction?{C.RESET}")
        again = prompt("Enter y to continue, n to exit", "y/n").lower()
        if again not in ('y', 'yes'):
            print(f"\n  {C.DIM}Goodbye.{C.RESET}\n")
            break
        clear()
        banner()

if __name__ == '__main__':
    main()
