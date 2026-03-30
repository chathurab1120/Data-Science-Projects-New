# 💳 Credit Card Fraud Detection
### Production-ready ML pipeline for real-time fraud risk assessment

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-Best_Model-success)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange)
![License](https://img.shields.io/badge/License-MIT-green)

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Streamlit_App-FF4B4B?style=for-the-badge)](YOUR_STREAMLIT_URL)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/chathurab1120/Data-Science-Projects-New)

## 📌 Project Overview

Credit card fraud is rare (0.1727%) but extremely costly for financial institutions and consumers. The central machine
learning challenge is severe class imbalance, where standard models can appear accurate while still missing the fraud
class. This project builds a full production pipeline from EDA and preprocessing to training, explainability, and
deployment. SHAP explainability makes each prediction transparent and auditable for real-world fintech use.

## 🏆 Key Results

| Metric | Value |
|--------|-------|
| Best Model | LightGBM |
| PR-AUC | 0.8795 |
| ROC-AUC | 0.9794 |
| Recall (Fraud) | 86.7% |
| Precision | 73.9% |
| F1 Score | 0.798 |
| Fraud Cases Caught | 85 / 98 |
| False Alarms | 30 / 56,864 |

### All Models Comparison

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|----|---------|--------|
| **LightGBM** ⭐ | 0.739 | 0.867 | 0.798 | 0.979 | **0.880** |
| XGBoost | 0.759 | 0.837 | 0.796 | 0.980 | 0.858 |
| Random Forest | 0.430 | 0.878 | 0.577 | 0.982 | 0.825 |
| Logistic Regression | 0.052 | 0.908 | 0.098 | 0.975 | 0.758 |

## 📊 Dataset

- Source: ULB Machine Learning Group — Kaggle
- Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284,807 transactions over 2 days
- 492 fraud cases (0.1727%)
- Features: 28 anonymized PCA components (V1-V28) + Time + Amount
- Target: Class (0 = Legitimate, 1 = Fraud)

Data is not included in this repository due to size. Download from Kaggle and place at data/raw/creditcard.csv

## 🗂️ Project Structure

Credit-Card-Fraud-Detection/
├── app/
│   └── streamlit_app.py        # Streamlit dashboard (4 pages)
├── src/
│   ├── data/
│   │   ├── eda.py              # Exploratory data analysis (10 cells)
│   │   └── preprocessing.py   # Feature engineering + SMOTE (10 cells)
│   ├── models/
│   │   └── train.py            # Model training + evaluation (11 cells)
│   └── visualization/
│       └── shap_analysis.py   # SHAP explainability (11 cells)
├── data/
│   ├── raw/                    # creditcard.csv (not tracked)
│   └── processed/             # Parquet files (not tracked)
├── outputs/
│   ├── models/                 # Trained model artifacts
│   ├── figures/                # All visualizations (12 figures)
│   └── reports/               # CSVs and HTML force plots
├── notebooks/                  # Jupyter exports for stakeholders
├── tests/
├── config.py                   # Central path + constants config
├── requirements.txt
└── README.md

## ⚙️ ML Pipeline

1. 🔍 EDA — Class imbalance analysis, amount/time distributions,
   KDE plots for all 28 PCA features, correlation matrix,
   KS-test to rank most discriminative features

2. 🔧 Preprocessing — 10 engineered features including cyclical
   time encoding, PCA interaction terms, log-transformed amount.
   RobustScaler fitted on train only. SMOTE balancing:
   394 → 227,451 fraud samples

3. 🤖 Training — 4 models with stratified CV. LightGBM and XGBoost
   trained on raw imbalanced data (native class weighting).
   Logistic Regression and Random Forest trained on SMOTE-resampled data.
   Best model selected by PR-AUC

4. 🔬 SHAP Analysis — TreeExplainer for LightGBM. Summary bar,
   beeswarm, waterfall plots for individual predictions,
   dependence plots for top 6 features, HTML force plots

5. 🌐 Deployment — Streamlit 4-page dashboard deployed to
   Streamlit Community Cloud. Real-time fraud scoring with
   on-the-fly SHAP explanations

## 🔍 SHAP Explainability

SHAP is essential in fraud detection because high-stakes model decisions must be transparent and defensible. It enables
regulatory-friendly explanations, auditable decisions for analysts, and faster debugging of feature behavior in
production.

| Rank | Feature | Mean \|SHAP\| | Description |
|------|---------|-------------|-------------|
| 1 | V4 | 0.785 | PCA component — strongest fraud signal |
| 2 | V8 | 0.243 | PCA component |
| 3 | amount_v14 | 0.229 | Engineered: Amount × V14 interaction |
| 4 | V14 | 0.160 | PCA component — key fraud indicator |
| 5 | time_sin | 0.157 | Engineered: cyclical time encoding |
| 6 | V1 | 0.146 | PCA component |
| 7 | V18 | 0.141 | PCA component |
| 8 | V15 | 0.126 | PCA component |

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/chathurab1120/Data-Science-Projects-New.git
cd Data-Science-Projects-New/Credit-Card-Fraud-Detection

# Create virtual environment with Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place at:
# data/raw/creditcard.csv
```

## 🚀 Usage

```bash
# Run EDA
python src/data/eda.py

# Run Preprocessing
python src/data/preprocessing.py

# Train Models (~9 minutes)
python src/models/train.py

# Run SHAP Analysis
python src/visualization/shap_analysis.py

# Launch Streamlit App locally
streamlit run app/streamlit_app.py
```

Each .py file uses # %% cell markers and can be run
cell-by-cell in VS Code Python Interactive or exported to
Jupyter notebook for stakeholder sharing

## 🌐 Streamlit App

- 🏠 Overview — Project metrics, dataset summary, model leaderboard
- 🔍 Fraud Predictor — Real-time transaction scoring with SHAP explanation
- 📊 Model Performance — ROC/PR curves, confusion matrix, all model metrics
- 🔬 SHAP Explainability — Global and local SHAP visualizations

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Streamlit_App-FF4B4B?style=for-the-badge)](YOUR_STREAMLIT_URL)

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| ML Models | LightGBM, XGBoost, Random Forest, Logistic Regression |
| Imbalanced Learning | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |
| Code Quality | Black, isort, loguru, type hints |
| Version Control | Git + GitHub |

## 📄 License
MIT License

## 🙏 Acknowledgements
- Dataset: ULB Machine Learning Group
- SHAP library: Lundberg & Lee (2017)
- Deployed via Streamlit Community Cloud
