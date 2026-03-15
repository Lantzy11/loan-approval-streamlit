# 🏦 Loan Approval Predictor

A binary classification project that predicts whether a loan application will be **approved or rejected**, built with XGBoost and Random Forest on the Loan Approval 2025 dataset.

---

## 📁 Project Structure

```
loan-approval-predictor/
├── loan_approval_2025_analysis.ipynb   # Full analysis, preprocessing & model training
├── loan_predictor.py                   # Console app for running predictions
├── xgb_model.pkl                       # Pre-trained XGBoost model (ready to use)
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 🚀 Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/loan-approval-predictor.git
cd loan-approval-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the predictor**
```bash
python loan_predictor.py
```

The pre-trained model (`xgb_model.pkl`) is included — no need to run the notebook first.

> **Want to retrain the model?** Place `Loan_approval_data_2025.csv` in the project root, open the notebook in Jupyter or Google Colab, run all cells, and Section 20a will regenerate `xgb_model.pkl`.

---

## 📊 Models

Two models were trained and compared:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Random Forest | ~0.90 | — | — | ~0.91 | ~0.97 |
| XGBoost (tuned) | ~0.92 | — | — | ~0.93 | ~0.98 |

> Values populate after running the notebook. XGBoost was selected for the predictor console.

Both models passed the **train vs. test overfitting diagnostic** with F1 gaps under 0.02.

---

## 🔧 Preprocessing Pipeline

| Step | Details |
|---|---|
| Log transformation | `annual_income`, `savings_assets`, `current_debt`, `loan_amount` |
| Multicollinearity check | VIF analysis on ratio features; `payment_to_income_ratio` dropped |
| Categorical encoding | One-Hot Encoding (fit on train only, `drop_first=True`) |
| Train/test split | 80/20 stratified split |
| Feature scaling | StandardScaler (fit on train only) |

---

## 🔍 Key Feature Signals

- **`defaults_on_file`** — near-perfect separator: 0% of approved applicants had defaults vs. 11.9% of rejected
- **`credit_score`** — 64.5-point average gap between approved (672.6) and rejected (608.1) applicants
- **`debt_to_income_ratio`** — approved average 0.240 vs. rejected 0.342
- **`loan_intent`** — Debt Consolidation has the lowest approval rate (36.6%); Education the highest (67.5%)

---

## 🎯 Hyperparameter Tuning

**Random Forest** — `max_depth` sweep across `[3, 5, 7, 10, 12, 15, 20, None]` with Train vs. Test F1 plotted to find the optimal depth.

**XGBoost** — `RandomizedSearchCV` over 40 combinations across 7 parameters:
- `max_depth`, `learning_rate`, `n_estimators`
- `subsample`, `colsample_bytree`
- `reg_alpha` (L1), `reg_lambda` (L2)

---

## 💼 Business Context

| | |
|---|---|
| **Industry** | Retail Banking, Digital Lending, Fintech |
| **Problem** | Manual loan reviews are slow, inconsistent, and costly |
| **Solution** | Automate first-pass decisions; escalate borderline cases to human reviewers |
| **Priority metrics** | Precision (avoid approving risky loans) + Recall (avoid rejecting good applicants) |
| **Core decision signals** | `defaults_on_file` + `credit_score` + `debt_to_income_ratio` |

---

## 🛠️ Tech Stack

- Python 3.10+
- scikit-learn
- XGBoost
- pandas / numpy
- matplotlib / seaborn
- statsmodels (VIF analysis)
- joblib (model serialisation)
