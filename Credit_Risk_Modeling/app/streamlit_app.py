"""
App: streamlit_app.py
Purpose: 4-page Streamlit dashboard for the Credit Risk Modeling project.
Pages:
  1. Project Overview  — summary, dataset stats, key results
  2. EDA              — all 5 EDA charts + IV table
  3. Model Results    — model comparison, ROC, KS, lift, confusion matrix
  4. SHAP & Scorecard — SHAP plots + scorecard band table + loan scorer
"""

# %% [0] Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import joblib
import yaml
from pathlib import Path

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Model | Portfolio",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load config ─────────────────────────────────────────────
# Resolve config path relative to this file's location — works both locally
# and on Streamlit Cloud regardless of working directory
import os
_APP_DIR = Path(__file__).parent          # app/
_PROJECT_DIR = _APP_DIR.parent            # Credit_Risk_Modeling/
_CONFIG_PATH = _PROJECT_DIR / "configs" / "config.yaml"

with open(_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

CHARTS  = _PROJECT_DIR / config["paths"]["outputs_charts"]
RESULTS = _PROJECT_DIR / config["paths"]["outputs_results"]
MODELS  = _PROJECT_DIR / config["paths"]["outputs_models"]

# ── Helper: load image ──────────────────────────────────────
def show_chart(filename, caption="", width=None):
    path = CHARTS / filename
    if path.exists():
        if width:
            st.image(str(path), caption=caption, width=width)
        else:
            st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Chart not found: {filename}")

# ── Helper: load CSV ────────────────────────────────────────
@st.cache_data
def load_csv(filename):
    path = RESULTS / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

# ── Helper: load model ──────────────────────────────────────
@st.cache_resource
def load_model(filename):
    path = MODELS / filename
    if path.exists():
        return joblib.load(path)
    return None

# ── Sidebar navigation ──────────────────────────────────────
st.sidebar.title("💳 Credit Risk Model")
st.sidebar.markdown("**Retail Loan Default Prediction**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Project Overview", "Exploratory Data Analysis",
     "Model Results", "SHAP & Scorecard"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.markdown("LendingClub 2007–2018")
st.sidebar.markdown("~1.34M loans after filtering")
st.sidebar.markdown("---")
st.sidebar.markdown("**Champion Model**")
st.sidebar.markdown("XGBoost (Optuna tuned)")
st.sidebar.markdown("Test AUC: **0.7131**")
st.sidebar.markdown("KS Stat: **0.3087**")
st.sidebar.markdown("Gini: **0.4263**")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)]"
    "(https://github.com/chathurab1120/Data-Science-Projects-New)"
)


# ════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "Project Overview":
    st.title("💳 Credit Risk Modeling")
    st.subheader("Retail Loan Default Prediction — Production-Grade PD Model")
    st.markdown("""
    This project builds an end-to-end **Probability of Default (PD) model** on
    LendingClub loan data (2007–2018). The model estimates the likelihood a borrower
    will default within 12 months of loan origination — the core problem in retail
    credit risk.
    """)

    st.markdown("---")

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Dataset Size",    "1.34M loans")
    col2.metric("Default Rate",    "19.96%")
    col3.metric("Champion AUC",    "0.7131")
    col4.metric("KS Statistic",    "0.3087")
    col5.metric("PSI (Stability)", "0.0057")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Business Problem")
        st.markdown("""
        **Objective:** Score loan applicants at origination to estimate their
        probability of defaulting within 12 months.

        **Why it matters:**
        - A 1% improvement in default prediction on a $1B portfolio = $10M in avoided losses
        - Mis-scoring good borrowers = lost revenue
        - Mis-scoring bad borrowers = credit losses

        **Approach:**
        1. Exploratory Data Analysis + Information Value ranking
        2. Chronological train/test split (out-of-time validation)
        3. Logistic Regression baseline → XGBoost champion
        4. SHAP explainability + traditional scorecard
        """)

    with col_right:
        st.subheader("Model Results Summary")
        final_metrics = load_csv("final_metrics.csv")
        comparison    = load_csv("model_comparison.csv")
        if not comparison.empty:
            st.dataframe(
                comparison[["model", "test_auc", "ks", "gini"]].style.format({
                    "test_auc": "{:.4f}",
                    "ks":       "{:.4f}",
                    "gini":     "{:.4f}"
                }).highlight_max(subset=["test_auc", "ks", "gini"], color="#d4edda"),
                use_container_width=True
            )
        if not final_metrics.empty:
            st.markdown(f"""
            **Champion: {final_metrics['model'].values[0]}**
            - Top Decile Lift: **{final_metrics['top_decile_lift'].values[0]:.2f}x**
            - Average Precision: **{final_metrics['avg_precision'].values[0]:.4f}**
            - PSI (Train vs Test): **{final_metrics['psi'].values[0]:.4f}** (Stable)
            """)

    st.markdown("---")
    st.subheader("Project Pipeline")
    pipeline_cols = st.columns(5)
    stages = [
        ("📥", "Data Loading", "2.26M rows\n151 columns"),
        ("🔍", "EDA + IV", "5 charts\nIV ranking"),
        ("🔧", "Feature Eng.", "6 new features\n43 total"),
        ("🤖", "Modelling", "LR + XGB\n+ LightGBM"),
        ("📊", "Validation", "AUC/KS/PSI\nSHAP + Scorecard")
    ]
    for col, (icon, title, desc) in zip(pipeline_cols, stages):
        col.markdown(f"""
        <div style='text-align:center; padding:12px; background:#f8f9fa;
                    border-radius:8px; border:1px solid #dee2e6'>
            <div style='font-size:28px'>{icon}</div>
            <div style='font-weight:bold; margin-top:6px'>{title}</div>
            <div style='font-size:12px; color:#666; white-space:pre-line'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Tech Stack")
    tech_cols = st.columns(6)
    techs = ["Python 3.11", "XGBoost", "LightGBM", "SHAP", "Optuna", "Streamlit"]
    for col, tech in zip(tech_cols, techs):
        col.markdown(f"<div style='text-align:center; padding:8px; background:#e9ecef; "
                     f"border-radius:6px; font-weight:bold'>{tech}</div>",
                     unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATORY DATA ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("Deep dive into the LendingClub dataset to understand default drivers.")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Default by Grade", "FICO Distribution", "Default by Purpose",
        "Missing Values", "DTI vs Income", "Information Value"
    ])

    with tab1:
        st.subheader("Default Rate by Loan Grade")
        st.markdown("""
        LendingClub assigns grades A–G based on internal risk assessment.
        Grade A is the safest, Grade G is the riskiest.
        This chart validates the grading system has real predictive power.
        """)
        show_chart("01_default_rate_by_grade.png")
        st.info("**Key finding:** Grade G loans default at ~50% vs Grade A at ~6%. "
                "The 8x spread confirms grade is a strong predictor (IV > 0.3).")

    with tab2:
        st.subheader("FICO Score Distribution by Outcome")
        st.markdown("""
        Overlapping KDE plot shows how FICO separates defaulters from non-defaulters.
        The distributions overlap significantly — FICO alone is not sufficient.
        """)
        show_chart("02_fico_distribution_by_outcome.png")
        st.info("**Key finding:** Good loans average FICO 700.3 vs bad loans 689.9. "
                "Separation exists but distributions overlap — FICO needs other features.")

    with tab3:
        st.subheader("Default Rate by Loan Purpose")
        st.markdown("""
        Loan purpose captures the borrower's intended use.
        Some purposes (small business, renewable energy) have structurally higher default rates.
        """)
        show_chart("03_default_rate_by_purpose.png")
        st.info("**Key finding:** Small business loans have highest default rate — "
                "business income is more volatile than salary income.")

    with tab4:
        st.subheader("Missing Value Pattern")
        st.markdown("""
        The missing value heatmap reveals which columns have systematic missingness.
        Systematic patterns (columns missing together) reveal data structure.
        58 columns had >40% missing and were excluded from modelling.
        """)
        show_chart("04_missing_value_heatmap.png")
        st.info("**Key finding:** Many hardship-related columns are missing for "
                "borrowers with no hardship history — this is informative, not random.")

    with tab5:
        st.subheader("DTI vs Annual Income")
        st.markdown("""
        High debt-to-income ratio combined with low income is a strong default signal.
        Income axis is log-scaled due to heavy right skew.
        """)
        show_chart("05_dti_vs_income_scatter.png")
        st.info("**Key finding:** Defaulters cluster in the high-DTI, low-income zone. "
                "The combination matters more than either feature alone.")

    with tab6:
        st.subheader("Information Value Table")
        st.markdown("""
        IV measures each feature's predictive power. Features with IV > 0.3 are strong predictors.
        IV < 0.02 are useless and dropped.
        """)
        iv_df = load_csv("information_value_table.csv")
        if not iv_df.empty:
            def color_strength(val):
                colors = {"Strong": "#d4edda", "Medium": "#fff3cd",
                          "Weak": "#f8d7da", "Useless": "#f5f5f5"}
                return f"background-color: {colors.get(val, 'white')}"
            st.dataframe(
                iv_df.style.format({"iv": "{:.4f}"})
                     .map(color_strength, subset=["strength"]),
                use_container_width=True
            )
        st.info("**Key finding:** sub_grade, grade, int_rate have IV > 0.3 (Strong). "
                "fico_midpoint is Medium (IV=0.12). These are the model's top features.")


# ════════════════════════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ════════════════════════════════════════════════════════════
elif page == "Model Results":
    st.title("🤖 Model Results")
    st.markdown("Comparison of all three models with full validation metrics.")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Comparison", "ROC & KS Curves", "Lift & Calibration", "Confusion Matrix"
    ])

    with tab1:
        st.subheader("Model Comparison: LR vs XGBoost vs LightGBM")
        comparison = load_csv("model_comparison.csv")
        if not comparison.empty:
            col1, col2, col3 = st.columns(3)
            xgb_row = comparison[comparison["model"] == "XGBoost"].iloc[0]
            lgb_row = comparison[comparison["model"] == "LightGBM"].iloc[0]
            lr_row  = comparison[comparison["model"] == "Logistic Regression"].iloc[0]
            col1.metric("LR Baseline AUC",  f"{lr_row['test_auc']:.4f}")
            col2.metric("XGBoost AUC",      f"{xgb_row['test_auc']:.4f}",
                        delta=f"+{xgb_row['test_auc']-lr_row['test_auc']:.4f}")
            col3.metric("LightGBM AUC",     f"{lgb_row['test_auc']:.4f}",
                        delta=f"+{lgb_row['test_auc']-lr_row['test_auc']:.4f}")
            st.markdown("---")
            st.dataframe(comparison.style.format({
                "test_auc": "{:.4f}", "train_auc": "{:.4f}",
                "ks": "{:.4f}", "gini": "{:.4f}"
            }), use_container_width=True)
        show_chart("09_model_comparison.png")
        st.markdown("""
        **Why XGBoost won:**
        - Highest test AUC (0.7131) on out-of-time test set
        - Minimal overfit gap (train 0.7282 vs test 0.7131)
        - Optuna-tuned with 50 trials on stratified 20% sample

        **LightGBM trade-off:**
        - Comparable AUC (0.7112) at 4x faster training (6.6s vs 25.7s)
        - Strong candidate for production where latency matters
        """)

    with tab2:
        st.subheader("ROC Curve — Baseline vs Champion")
        col1, col2 = st.columns(2)
        with col1:
            show_chart("06_roc_curve_baseline.png", "Logistic Regression (Baseline)")
        with col2:
            show_chart("07_ks_chart_baseline.png", "KS Chart — Baseline")
        st.info("**KS Statistic = 0.3087** — The maximum separation between the "
                "cumulative good and bad distributions. Industry target is KS > 0.35.")

    with tab3:
        st.subheader("Lift Chart & Calibration")
        col1, col2 = st.columns(2)
        with col1:
            show_chart("11_lift_chart.png", "Lift Chart by Decile")
            lift_df = load_csv("lift_table.csv")
            if not lift_df.empty:
                st.dataframe(
                    lift_df[["decile", "total", "defaults", "default_rate", "lift"]]
                    .style.format({"default_rate": "{:.4f}", "lift": "{:.2f}"}),
                    use_container_width=True
                )
        with col2:
            show_chart("12_calibration_curve.png", "Calibration Curve")
            st.markdown("""
            **Lift Chart:** Top decile captures **2.18x** more defaults than random.
            Targeting the top 10% of predictions catches
            disproportionately more real defaults.

            **Calibration:** The model tends to compress predictions
            toward the centre. Isotonic regression calibration
            could improve this in production.
            """)

    with tab4:
        st.subheader("Confusion Matrix & Precision-Recall")
        col1, col2 = st.columns(2)
        with col1:
            show_chart("10_confusion_matrix.png", "Confusion Matrix (threshold=0.35)")
        with col2:
            show_chart("13_precision_recall_curve.png", "Precision-Recall Curve")
        final_metrics = load_csv("final_metrics.csv")
        if not final_metrics.empty:
            m = final_metrics.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision @ 0.35", f"{m['precision_at_threshold']:.4f}")
            c2.metric("Recall @ 0.35",    f"{m['recall_at_threshold']:.4f}")
            c3.metric("PSI",              f"{m['psi']:.4f}", delta="Stable")
        st.info("**Threshold = 0.35:** In credit risk, missing a default (false negative) "
                "costs more than a false alarm. Lower threshold increases recall "
                "at the cost of precision.")


# ════════════════════════════════════════════════════════════
# PAGE 4 — SHAP & SCORECARD
# ════════════════════════════════════════════════════════════
elif page == "SHAP & Scorecard":
    st.title("🧠 SHAP Explainability & Credit Scorecard")
    st.markdown("Understanding why the model makes each prediction.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["SHAP Global", "SHAP Individual", "Credit Scorecard"])

    with tab1:
        st.subheader("Global Feature Importance (SHAP)")
        col1, col2 = st.columns(2)
        with col1:
            show_chart("14_shap_global_bar.png", "Mean Absolute SHAP Values")
        with col2:
            show_chart("15_shap_beeswarm.png", "Beeswarm — Direction & Magnitude")
        st.markdown("---")
        show_chart("16_shap_fico_dependence.png", "SHAP Dependence: FICO vs Interest Rate")
        st.info("**Key finding:** FICO below 680 causes a sharp increase in predicted "
                "default probability. The effect is amplified by high interest rates "
                "(shown in red) — consistent with the IV analysis.")

    with tab2:
        st.subheader("Individual Loan Explanations")
        col1, col2 = st.columns(2)
        with col1:
            show_chart("17_shap_waterfall_default.png",
                       "Defaulted Loan — Predicted PD: 0.644")
            st.error("High-risk loan: grade, loan amount, and interest rate "
                     "all push toward default.")
        with col2:
            show_chart("18_shap_waterfall_good.png",
                       "Good Loan — Predicted PD: 0.655")
            st.success("Features that reduce default risk are shown in green. "
                       "Higher FICO and lower DTI push score downward (toward safe).")
        st.markdown("""
        **How to read waterfall plots:**
        - Each bar shows one feature's contribution to the prediction
        - Red bars push toward **higher** default probability
        - Green bars push toward **lower** default probability
        - The final prediction is the base rate plus all contributions summed
        """)

    with tab3:
        st.subheader("Points-Based Credit Scorecard")
        st.markdown("""
        Traditional scorecards convert the logistic regression model into a
        human-interpretable points system. Standard scaling: **PDO=20, Base Score=600, Base Odds=50:1**.
        """)

        col1, col2, col3 = st.columns(3)
        col1.metric("PDO",        "20 points")
        col2.metric("Base Score", "600")
        col3.metric("Base Odds",  "50:1 (good:bad)")

        st.markdown("---")
        st.subheader("Default Rate by Score Band")
        show_chart("20_scorecard_default_rate_by_band.png")

        scorecard_df = load_csv("scorecard_table.csv")
        if not scorecard_df.empty:
            st.dataframe(
                scorecard_df.style.format({
                    "default_rate":  "{:.2f}%",
                    "approval_rate": "{:.2f}%"
                }).background_gradient(subset=["default_rate"], cmap="RdYlGn_r"),
                use_container_width=True
            )

        st.markdown("---")
        show_chart("19_score_distribution.png", "Score Distribution by Outcome")
        st.info("**Interpretation:** Higher score = lower default risk. "
                "Borrowers scoring above 600 have substantially lower default rates. "
                "The score separation between good and bad loans validates the scorecard.")
