[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-risk-model-pd.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/chathurab1120/Data-Science-Projects-New/tree/main/Credit_Risk_Modeling)

# Credit Risk Modeling — Retail Loan Default Prediction

**Project folder:** [`Credit_Risk_Modeling/`](https://github.com/chathurab1120/Data-Science-Projects-New/tree/main/Credit_Risk_Modeling) — same layout as the other portfolio projects (`scripts/`, `app/`, `outputs/`, `configs/`, etc.).

### Project Overview
A production-grade Probability of Default (PD) model built on LendingClub loan data (2007–2018, ~2.26M records). The model estimates the likelihood a borrower will default within 12 months of loan origination.

**Business Objective:** Support credit decisioning by scoring applicants at origination.

### Dataset
- **Source:** [LendingClub Loan Data — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **File:** `accepted_2007_to_2018Q4.csv` (~1.3 GB compressed)
- **Records:** ~2.26 million loans
- **Target:** `default_flag` — 1 = Default/Charged Off, 0 = Fully Paid

### Methodology
1. Exploratory Data Analysis (EDA) with Information Value ranking
2. Data Cleaning — leakage removal, imputation, outlier capping
3. Feature Engineering — credit history, loan-to-income ratio, FICO midpoint
4. Chronological Train/Test Split (train: 2007–2015, test: 2016–2018)
5. Baseline: Logistic Regression
6. Advanced: XGBoost + LightGBM with Optuna hyperparameter tuning
7. Full validation: AUC, KS, Gini, PSI, Lift chart
8. SHAP explainability (global + local)
9. Points-based Scorecard

### Results
| Model | AUC | KS | Gini |
|---|---|---|---|
| Logistic Regression | 0.7029 | 0.2935 | 0.4058 |
| XGBoost (Champion) | 0.7131 | 0.3087 | 0.4263 |
| LightGBM | 0.7112 | 0.3054 | 0.4223 |

### Key Findings
- **Top predictors:** grade, interest rate, FICO score, loan-to-income ratio (confirmed by both IV table and SHAP)
- **Out-of-time validation:** Train 2007–2015, Test 2016–2018 — no data leakage
- **PSI = 0.0057:** Score distribution is highly stable between train and test
- **Top decile lift: 2.18x** — targeting top 10% of predicted defaulters captures 2.18x more actual defaults than random
- **Scorecard:** Logistic regression converted to points-based scorecard (PDO=20, Base Score=600)

### Interview Talking Points
- Chronological split chosen over random split to simulate real deployment conditions
- class_weight='balanced' used instead of SMOTE — preserves data distribution
- Threshold set to 0.35 (not 0.5) — missing a default costs more than a false alarm
- LightGBM trains 4x faster than XGBoost with only 0.0019 AUC difference — strong production candidate
- SHAP values used over gain-based importance — accounts for feature interactions

### Project Structure (Credit Risk)
```
Credit_Risk_Modeling/
├── scripts/          # Numbered .py scripts (# %% cell structured)
├── src/              # Importable helper modules
├── configs/          # config.yaml — central configuration
├── outputs/          # Charts, models, results
│   ├── charts/
│   ├── models/
│   └── results/
├── notebooks/        # Optional: jupytext exports
├── app/              # Streamlit dashboard
├── data/             # Raw data (gitignored — download from Kaggle)
├── requirements.txt
└── runtime.txt
```

### How to Run (Credit Risk)
```bash
cd Credit_Risk_Modeling
pip install -r requirements.txt
# Download dataset from Kaggle and place in data/ (under project root or Credit_Risk_Modeling/data/)
python scripts/01_data_loading.py
# ... run pipeline in order ...
streamlit run app/streamlit_app.py
```

### Deployment (Credit Risk)
- **Interactive Dashboard:** [Streamlit Cloud](https://credit-risk-model-pd.streamlit.app/)
- **Model Demo:** [Hugging Face Spaces](TBD)

### Tech Stack (Credit Risk)
Python 3.11 | XGBoost | LightGBM | SHAP | Streamlit | Gradio | Optuna

---
*Repository snapshot: March 2026.*
