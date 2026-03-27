"""
Production Streamlit dashboard for credit card fraud detection and explainability.
"""

# %% Imports and Setup
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from loguru import logger

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ROOT_DIR

FRAUD_COLOR: str = "#e63946"
LEGIT_COLOR: str = "#2dc653"

DATA_PROCESSED_DIR: Path = ROOT_DIR / "data" / "processed"
OUTPUTS_MODELS_DIR: Path = ROOT_DIR / "outputs" / "models"
OUTPUTS_REPORTS_DIR: Path = ROOT_DIR / "outputs" / "reports"
OUTPUTS_FIGURES_DIR: Path = ROOT_DIR / "outputs" / "figures"
LOGS_DIR: Path = ROOT_DIR / "logs"
LOG_FILEPATH: Path = LOGS_DIR / "streamlit_app.log"

FEATURE_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)] + [
    "amount_log",
    "amount_squared",
    "time_sin",
    "time_cos",
    "v14_v10_interaction",
    "v14_v12_interaction",
    "v10_v12_interaction",
    "amount_v14",
    "high_amount_flag",
]

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_FILEPATH, level="INFO", rotation="10 MB", enqueue=True)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# %% Cached Loaders
@st.cache_resource
def load_model_and_scaler() -> tuple[Any | None, Any | None]:
    """Load and cache trained model and scaler artifacts.

    Args:
        None.

    Returns:
        Tuple of (model, scaler) where either can be None if missing.

    Raises:
        OSError: If artifact deserialization fails.
    """
    model_path: Path = OUTPUTS_MODELS_DIR / "best_model.pkl"
    scaler_path: Path = OUTPUTS_MODELS_DIR / "robust_scaler.pkl"
    model: Any | None = None
    scaler: Any | None = None

    if model_path.exists():
        model = joblib.load(model_path)
        logger.info("Loaded best model: {}", model_path)
    else:
        logger.warning("Missing best model artifact: {}", model_path)

    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        logger.info("Loaded scaler: {}", scaler_path)
    else:
        logger.warning("Missing scaler artifact: {}", scaler_path)

    return model, scaler


@st.cache_data
def load_dataframe(filepath: Path) -> pd.DataFrame:
    """Load and cache parquet/csv DataFrames with extension-based readers.

    Args:
        filepath: Path to tabular file.

    Returns:
        Loaded DataFrame or empty DataFrame if file is unavailable.

    Raises:
        OSError: If file read fails unexpectedly.
    """
    if not filepath.exists():
        logger.warning("Requested dataframe file not found: {}", filepath)
        return pd.DataFrame()

    if filepath.suffix == ".csv":
        dataframe: pd.DataFrame = pd.read_csv(filepath)
    elif filepath.suffix == ".parquet":
        dataframe = pd.read_parquet(filepath)
    else:
        logger.warning("Unsupported dataframe file extension for {}", filepath)
        return pd.DataFrame()

    logger.info("Loaded dataframe {} with shape {}", filepath.name, dataframe.shape)
    return dataframe


@st.cache_data
def load_image_bytes(filepath: Path) -> bytes:
    """Load and cache image bytes for Streamlit rendering.

    Args:
        filepath: Path to image file.

    Returns:
        Raw image bytes, or empty bytes if missing.

    Raises:
        OSError: If file cannot be opened.
    """
    if not filepath.exists():
        logger.warning("Requested image file not found: {}", filepath)
        return b""
    return filepath.read_bytes()


# %% Helpers
def show_missing_file_warning(filepath: Path) -> None:
    """Display a consistent warning for missing artifacts.

    Args:
        filepath: Missing file path.

    Returns:
        None.

    Raises:
        None.
    """
    st.warning(f"Missing required file: `{filepath}`")


def render_footer() -> None:
    """Render standard footer across all pages.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    st.markdown("---")
    st.caption(
        "Built with ❤️ using Python, LightGBM & SHAP | Data: ULB Machine Learning Group | "
        "Deployed on Streamlit Community Cloud"
    )


def display_image_or_warning(filepath: Path, caption: str) -> None:
    """Display cached image bytes or warn if file is missing.

    Args:
        filepath: Path to image artifact.
        caption: Caption text shown under the image.

    Returns:
        None.

    Raises:
        None.
    """
    image_bytes: bytes = load_image_bytes(filepath)
    if image_bytes:
        st.image(image_bytes, caption=caption, use_column_width=True)
    else:
        show_missing_file_warning(filepath)


def engineer_transaction_features(amount: float, time_input: int, pca_values: dict[str, float]) -> pd.DataFrame:
    """Engineer model features from user transaction inputs.

    Args:
        amount: Transaction amount in dollars.
        time_input: Seconds elapsed since first transaction.
        pca_values: Dictionary of V1-V28 input values.

    Returns:
        Single-row DataFrame with all model-required features.

    Raises:
        KeyError: If any expected PCA feature is missing.
    """
    time_hour: float = (time_input % 86400) / 3600
    engineered_values: dict[str, float] = {
        "amount_log": float(np.log1p(amount)),
        "amount_squared": float(amount**2),
        "time_sin": float(np.sin(2 * np.pi * time_hour / 24)),
        "time_cos": float(np.cos(2 * np.pi * time_hour / 24)),
        "v14_v10_interaction": float(pca_values["V14"] * pca_values["V10"]),
        "v14_v12_interaction": float(pca_values["V14"] * pca_values["V12"]),
        "v10_v12_interaction": float(pca_values["V10"] * pca_values["V12"]),
        "amount_v14": float(amount * pca_values["V14"]),
        "high_amount_flag": float(int(amount > 1000)),
    }

    row_dict: dict[str, float] = {}
    for feature_name in [f"V{i}" for i in range(1, 29)]:
        row_dict[feature_name] = float(pca_values[feature_name])
    row_dict.update(engineered_values)

    feature_df: pd.DataFrame = pd.DataFrame([row_dict], columns=FEATURE_COLUMNS)
    return feature_df


def style_model_comparison(dataframe: pd.DataFrame) -> Any:
    """Apply visual highlighting to the best model row in comparison table.

    Args:
        dataframe: Model comparison DataFrame.

    Returns:
        Styled DataFrame object for Streamlit rendering.

    Raises:
        ValueError: If expected columns are missing.
    """
    if dataframe.empty or "pr_auc" not in dataframe.columns:
        return dataframe
    best_index: int = int(dataframe["pr_auc"].astype(float).idxmax())
    return dataframe.style.apply(
        lambda row: ["background-color: rgba(45, 198, 83, 0.30)" if row.name == best_index else "" for _ in row],
        axis=1,
    )


def render_prediction_shap(model: Any, scaled_input_df: pd.DataFrame) -> tuple[plt.Figure, list[tuple[str, float]]]:
    """Build SHAP waterfall plot and top feature contributions for one prediction.

    Args:
        model: Trained model supporting tree SHAP.
        scaled_input_df: Single-row scaled feature DataFrame.

    Returns:
        Tuple of (matplotlib_figure, top_feature_contributions).

    Raises:
        ValueError: If SHAP computation fails.
    """
    explainer: shap.TreeExplainer = shap.TreeExplainer(model)
    raw_explanation: shap.Explanation = explainer(scaled_input_df)
    if raw_explanation.values.ndim == 3:
        explanation: shap.Explanation = shap.Explanation(
            values=raw_explanation.values[:, :, 1],
            base_values=raw_explanation.base_values[:, 1],
            data=raw_explanation.data,
            feature_names=list(scaled_input_df.columns),
        )
    else:
        explanation = raw_explanation

    values_1d: np.ndarray = explanation.values[0]
    abs_order: np.ndarray = np.argsort(np.abs(values_1d))[::-1][:5]
    top_contributions: list[tuple[str, float]] = [
        (str(scaled_input_df.columns[index]), float(values_1d[index])) for index in abs_order
    ]

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation[0], max_display=12, show=False)
    figure: plt.Figure = plt.gcf()
    return figure, top_contributions


# %% Page 1 — Overview
def render_overview_page() -> None:
    """Render the project overview page.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    st.title("💳 Credit Card Fraud Detection")
    st.subheader("Production-ready machine learning pipeline for real-time fraud risk assessment.")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Total Transactions", "284,807")
    metric_col2.metric("Fraud Cases", "492")
    metric_col3.metric("Fraud Rate", "0.1727%")
    metric_col4.metric("Best Model PR-AUC", "0.8795")

    st.markdown("### About This Project")
    st.write(
        "Credit card fraud is extremely rare but highly costly, which makes class imbalance the core modeling "
        "challenge. This project uses the ULB Machine Learning Group dataset with anonymized PCA features and "
        "engineered transaction behavior features. The training pipeline emphasizes robust preprocessing, calibrated "
        "evaluation on imbalanced holdout data, and SHAP explainability for transparent decision support."
    )

    st.markdown("### Model Performance Summary")
    comparison_path: Path = OUTPUTS_REPORTS_DIR / "model_comparison.csv"
    comparison_df: pd.DataFrame = load_dataframe(comparison_path)
    if comparison_df.empty:
        show_missing_file_warning(comparison_path)
    else:
        st.dataframe(style_model_comparison(comparison_df), use_container_width=True)

    st.markdown("### Key Findings")
    st.markdown("- `V4` is the strongest global SHAP indicator in the best model.")
    st.markdown("- Engineered feature `amount_v14` appears among top fraud-driving features.")
    st.markdown("- `V14`, `V12`, and `V2` remain consistently important across EDA and SHAP.")

    class_distribution_path: Path = OUTPUTS_FIGURES_DIR / "01_class_distribution.png"
    display_image_or_warning(
        class_distribution_path,
        "Class imbalance in the dataset: legitimate vs fraudulent transactions.",
    )
    render_footer()


# %% Page 2 — Fraud Predictor
def render_predictor_page() -> None:
    """Render real-time transaction prediction page.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    st.title("Real-Time Fraud Detection")
    st.caption("Enter transaction details below to get an instant fraud assessment")

    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        st.warning("Model or scaler artifact is missing. Please ensure training and preprocessing are completed.")
        render_footer()
        return

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown("### Transaction Input")
        amount: float = float(
            st.number_input(
                "Transaction Amount ($)",
                min_value=0.0,
                max_value=50000.0,
                value=100.0,
                step=0.01,
            )
        )
        time_input: int = int(
            st.slider(
                "Time (seconds since first transaction)",
                min_value=0,
                max_value=172792,
                value=50000,
            )
        )

        pca_values: dict[str, float] = {f"V{i}": 0.0 for i in range(1, 29)}
        with st.expander("Advanced: PCA Features (V1-V28)"):
            st.caption(
                "These are anonymized PCA-transformed features. Leave at 0.0 for a typical transaction profile."
            )
            pca_columns = st.columns(4)
            for index in range(28):
                feature_name: str = f"V{index + 1}"
                with pca_columns[index % 4]:
                    pca_values[feature_name] = float(
                        st.slider(
                            feature_name,
                            min_value=-10.0,
                            max_value=10.0,
                            value=0.0,
                            step=0.1,
                        )
                    )

        analyze_clicked: bool = st.button("🔍 Analyze Transaction", type="primary", use_container_width=True)

    with right_col:
        st.markdown("### Assessment Results")
        if analyze_clicked:
            feature_df: pd.DataFrame = engineer_transaction_features(amount=amount, time_input=time_input, pca_values=pca_values)
            scaled_array: np.ndarray = scaler.transform(feature_df[FEATURE_COLUMNS])
            scaled_df: pd.DataFrame = pd.DataFrame(scaled_array, columns=FEATURE_COLUMNS)
            fraud_probability: float = float(model.predict_proba(scaled_df)[0, 1])
            confidence_value: float = float(max(fraud_probability, 1.0 - fraud_probability))

            if fraud_probability > 0.5:
                st.error("🚨 FRAUD DETECTED")
                st.metric("Fraud Probability", f"{fraud_probability * 100:.2f}%")
                st.markdown(
                    "<div style='height: 12px; background-color: rgba(230,57,70,0.20); border-radius: 8px;'>"
                    f"<div style='height: 12px; width: {fraud_probability * 100:.2f}%; background-color:{FRAUD_COLOR}; "
                    "border-radius: 8px;'></div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.success("✅ LEGITIMATE TRANSACTION")
                st.metric("Fraud Probability", f"{fraud_probability * 100:.2f}%")
                st.markdown(
                    "<div style='height: 12px; background-color: rgba(45,198,83,0.20); border-radius: 8px;'>"
                    f"<div style='height: 12px; width: {(1.0 - fraud_probability) * 100:.2f}%; background-color:{LEGIT_COLOR}; "
                    "border-radius: 8px;'></div></div>",
                    unsafe_allow_html=True,
                )

            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Fraud Probability", f"{fraud_probability:.4f}")
            metric_col2.metric("Confidence", f"{confidence_value:.4f}")

            shap_figure, top_features = render_prediction_shap(model=model, scaled_input_df=scaled_df)
            st.pyplot(shap_figure, use_container_width=True)
            plt.close(shap_figure)

            st.markdown("#### Top 5 Features Driving This Prediction")
            for feature_name, contribution in top_features:
                direction: str = "↑ fraud risk" if contribution > 0 else "↓ fraud risk"
                st.markdown(f"- `{feature_name}`: `{contribution:.4f}` ({direction})")
        else:
            st.info("Fill inputs and click **Analyze Transaction** to view risk and SHAP explanation.")

    render_footer()


# %% Page 3 — Model Performance
def render_model_performance_page() -> None:
    """Render model performance analytics page with tabs.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    st.title("Model Evaluation & Comparison")
    tab1, tab2, tab3, tab4 = st.tabs(["Model Comparison", "ROC & PR Curves", "Confusion Matrix", "Feature Importance"])

    with tab1:
        comparison_figure_path: Path = OUTPUTS_FIGURES_DIR / "06_model_comparison.png"
        display_image_or_warning(comparison_figure_path, "Side-by-side metric comparison across all trained models.")

        comparison_path: Path = OUTPUTS_REPORTS_DIR / "model_comparison.csv"
        comparison_df: pd.DataFrame = load_dataframe(comparison_path)
        if comparison_df.empty:
            show_missing_file_warning(comparison_path)
        else:
            st.dataframe(style_model_comparison(comparison_df), use_container_width=True)

    with tab2:
        roc_pr_path: Path = OUTPUTS_FIGURES_DIR / "07_roc_pr_curves.png"
        display_image_or_warning(roc_pr_path, "ROC and Precision-Recall curves on imbalanced holdout test data.")
        st.markdown(
            "ROC-AUC measures ranking quality across thresholds, while PR-AUC captures precision-recall tradeoffs for "
            "the rare fraud class. In highly imbalanced fraud detection, PR-AUC is the more informative metric."
        )

    with tab3:
        st.markdown("#### LightGBM Confusion Matrix (Best Model)")
        confusion_values: np.ndarray = np.array([[56834, 30], [13, 85]])
        heatmap_figure = go.Figure(
            data=go.Heatmap(
                z=confusion_values,
                x=["Predicted Legit", "Predicted Fraud"],
                y=["Actual Legit", "Actual Fraud"],
                colorscale="Viridis",
                text=confusion_values,
                texttemplate="%{text}",
                hoverongaps=False,
            )
        )
        heatmap_figure.update_layout(height=500, margin=dict(l=40, r=40, t=40, b=40))
        st.plotly_chart(heatmap_figure, use_container_width=True)
        st.markdown(
            "- **TN (56,834):** Legitimate transactions correctly classified as legitimate.\n"
            "- **FP (30):** Legitimate transactions incorrectly flagged as fraud.\n"
            "- **FN (13):** Fraud transactions missed by the model.\n"
            "- **TP (85):** Fraud transactions correctly detected."
        )

    with tab4:
        shap_bar_path: Path = OUTPUTS_FIGURES_DIR / "08_shap_summary_bar.png"
        display_image_or_warning(shap_bar_path, "Top global SHAP feature importances for the best model.")

        shap_importance_path: Path = OUTPUTS_REPORTS_DIR / "shap_feature_importance.csv"
        shap_importance_df: pd.DataFrame = load_dataframe(shap_importance_path)
        if shap_importance_df.empty:
            show_missing_file_warning(shap_importance_path)
        else:
            st.dataframe(shap_importance_df, use_container_width=True)

    render_footer()


# %% Page 4 — SHAP Explainability
def render_shap_explainability_page() -> None:
    """Render SHAP explainability page with global and local explanations.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    st.title("Understanding Model Decisions with SHAP")
    st.write(
        "SHAP values quantify how each feature pushes a prediction toward fraud or legitimacy. Positive SHAP values "
        "increase fraud risk, while negative values reduce it. This helps convert model outputs into transparent, "
        "auditable reasoning for analysts and stakeholders."
    )

    st.markdown("### Global Explanations")
    col_left, col_right = st.columns(2)
    with col_left:
        display_image_or_warning(
            OUTPUTS_FIGURES_DIR / "08_shap_summary_bar.png",
            "Global ranking of features by mean absolute SHAP importance.",
        )
    with col_right:
        display_image_or_warning(
            OUTPUTS_FIGURES_DIR / "09_shap_beeswarm.png",
            "Distribution of per-feature SHAP impacts across sampled transactions.",
        )

    st.markdown("### Individual Predictions — Fraud Cases")
    display_image_or_warning(
        OUTPUTS_FIGURES_DIR / "10_shap_waterfall_fraud.png",
        "Why the model flagged these transactions as fraud",
    )

    st.markdown("### Individual Predictions — Legitimate Cases")
    display_image_or_warning(
        OUTPUTS_FIGURES_DIR / "11_shap_waterfall_legit.png",
        "Why the model classified these as legitimate",
    )

    st.markdown("### Feature Dependence")
    display_image_or_warning(
        OUTPUTS_FIGURES_DIR / "12_shap_dependence.png",
        "Dependence plots show how feature values and interactions shift fraud risk contributions.",
    )

    render_footer()


# %% Main Execution
def main() -> None:
    """Run Streamlit multi-page navigation and render selected page.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    st.sidebar.title("Navigation")
    selected_page: str = st.sidebar.radio(
        "Go to",
        ["🏠 Overview", "🔍 Fraud Predictor", "📊 Model Performance", "🔬 SHAP Explainability"],
    )

    if selected_page == "🏠 Overview":
        render_overview_page()
    elif selected_page == "🔍 Fraud Predictor":
        render_predictor_page()
    elif selected_page == "📊 Model Performance":
        render_model_performance_page()
    else:
        render_shap_explainability_page()


if __name__ == "__main__":
    main()
