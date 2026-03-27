"""
Production-grade SHAP explainability pipeline for the best fraud model.
"""

# %% Imports and Setup
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from loguru import logger

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED, OUTPUTS_FIGURES, OUTPUTS_MODELS, OUTPUTS_REPORTS, RANDOM_STATE, ROOT_DIR

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_theme(style="darkgrid")
shap.initjs()

FRAUD_COLOR: str = "#e63946"
LEGIT_COLOR: str = "#2dc653"
TOP_N_FEATURES: int = 20

LOGS_DIR: Path = ROOT_DIR / "logs"
LOG_FILEPATH: Path = LOGS_DIR / "shap_analysis.log"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUTS_REPORTS.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_FILEPATH, level="INFO", rotation="10 MB", enqueue=True)


# %% Load Model and Data
def load_artifacts(
    model_path: Path,
    X_test_path: Path,
    y_test_path: Path,
) -> tuple[object, pd.DataFrame, pd.Series]:
    """Load trained model and processed test artifacts.

    Args:
        model_path: Path to serialized best model.
        X_test_path: Path to processed test features parquet.
        y_test_path: Path to processed test labels parquet.

    Returns:
        Tuple of (model, X_test, y_test).

    Raises:
        FileNotFoundError: If any artifact path does not exist.
        OSError: If any artifact cannot be loaded.
    """
    for artifact_path in [model_path, X_test_path, y_test_path]:
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing artifact: {artifact_path}")

    model: object = joblib.load(model_path)
    X_test: pd.DataFrame = pd.read_parquet(X_test_path)
    y_test: pd.Series = pd.read_parquet(y_test_path)["Class"]

    logger.info("Loaded model type: {}", type(model).__name__)
    logger.info("Loaded X_test shape: {}", X_test.shape)
    logger.info("Fraud count in test set: {}", int((y_test == 1).sum()))
    return model, X_test, y_test


# %% Compute SHAP Values
def compute_shap_values(
    model: object,
    X_test: pd.DataFrame,
    sample_size: int = 2000,
) -> tuple[shap.Explanation, pd.DataFrame]:
    """Compute SHAP explanations on a sampled test subset.

    Args:
        model: Trained tree-based model.
        X_test: Holdout feature dataset.
        sample_size: Number of rows sampled for SHAP computations.

    Returns:
        Tuple of (shap_explanation, X_sample).

    Raises:
        ValueError: If X_test is empty or sample size is invalid.
    """
    if X_test.empty:
        raise ValueError("X_test is empty; cannot compute SHAP values")

    effective_sample_size: int = min(sample_size, len(X_test))
    # Stratification is not possible here because y labels are not part of this function signature.
    X_sample: pd.DataFrame = X_test.sample(n=effective_sample_size, random_state=RANDOM_STATE)

    explainer: shap.TreeExplainer = shap.TreeExplainer(model)
    raw_shap_explanation: shap.Explanation = explainer(X_sample)
    if raw_shap_explanation.values.ndim == 3:
        shap_explanation = shap.Explanation(
            values=raw_shap_explanation.values[:, :, 1],
            base_values=raw_shap_explanation.base_values[:, 1],
            data=raw_shap_explanation.data,
            feature_names=list(X_sample.columns),
        )
    else:
        shap_explanation = raw_shap_explanation

    logger.info("Explainer type: {}", type(explainer).__name__)
    logger.info("SHAP sample shape: {}", X_sample.shape)
    logger.info("SHAP values shape: {}", shap_explanation.values.shape)

    mean_abs_shap: pd.Series = pd.Series(
        np.abs(shap_explanation.values).mean(axis=0),
        index=X_sample.columns,
    ).sort_values(ascending=False)
    logger.info("Top 10 mean absolute SHAP features:\n{}", mean_abs_shap.head(10).to_string())
    return shap_explanation, X_sample


# %% SHAP Summary Bar Plot
def plot_shap_summary_bar(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    save_path: Path,
) -> None:
    """Create and save SHAP mean absolute importance bar plot.

    Args:
        shap_values: SHAP explanation object.
        X_sample: Feature sample used for SHAP values.
        save_path: Output file path for summary bar figure.

    Returns:
        None.

    Raises:
        OSError: If figure cannot be saved.
    """
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=TOP_N_FEATURES, show=False)
    plt.title("SHAP Feature Importance — Mean |SHAP Value|")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary bar plot: {}", save_path)


# %% SHAP Beeswarm Plot
def plot_shap_beeswarm(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    save_path: Path,
) -> None:
    """Create and save SHAP beeswarm feature-impact plot.

    Args:
        shap_values: SHAP explanation object.
        X_sample: Feature sample used for SHAP values.
        save_path: Output file path for beeswarm figure.

    Returns:
        None.

    Raises:
        OSError: If figure cannot be saved.
    """
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="dot", max_display=TOP_N_FEATURES, show=False)
    plt.title("SHAP Beeswarm — Feature Impact on Fraud Prediction")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP beeswarm plot: {}", save_path)


# %% SHAP Waterfall Plot — Top Fraud Cases
def plot_shap_waterfall_fraud(
    shap_explanation: shap.Explanation,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    save_path: Path,
    n_cases: int = 3,
) -> None:
    """Create side-by-side SHAP waterfall plots for fraud cases.

    Args:
        shap_explanation: SHAP explanation object.
        X_sample: Sampled feature matrix.
        y_sample: Sampled labels aligned to X_sample index.
        save_path: Output path for waterfall composite figure.
        n_cases: Number of fraud cases to visualize.

    Returns:
        None.

    Raises:
        ValueError: If no fraud cases are available in the sample.
    """
    fraud_positions: list[int] = list(np.where(y_sample.values == 1)[0])
    if not fraud_positions:
        raise ValueError("No fraud cases found in SHAP sample")

    selected_positions: list[int] = fraud_positions[:n_cases]
    figure, axes = plt.subplots(1, n_cases, figsize=(20, 8))
    axes_list: list[Any] = [axes] if n_cases == 1 else list(axes)

    for subplot_axis, sample_position in zip(axes_list, selected_positions):
        plt.sca(subplot_axis)
        shap.plots.waterfall(shap_explanation[sample_position], max_display=12, show=False)
        subplot_axis.set_title(f"Fraud sample idx={int(X_sample.index[sample_position])}", color=FRAUD_COLOR)

    figure.suptitle("SHAP Waterfall — Why These Were Flagged as Fraud", color=FRAUD_COLOR, fontsize=14)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved SHAP fraud waterfall plot: {}", save_path)
    logger.info("Fraud waterfall sample indices: {}", [int(X_sample.index[i]) for i in selected_positions])


# %% SHAP Waterfall Plot — Top Legit Cases
def plot_shap_waterfall_legit(
    shap_explanation: shap.Explanation,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    save_path: Path,
    n_cases: int = 3,
) -> None:
    """Create side-by-side SHAP waterfall plots for legitimate cases.

    Args:
        shap_explanation: SHAP explanation object.
        X_sample: Sampled feature matrix.
        y_sample: Sampled labels aligned to X_sample index.
        save_path: Output path for waterfall composite figure.
        n_cases: Number of legitimate cases to visualize.

    Returns:
        None.

    Raises:
        ValueError: If no legitimate cases are available in the sample.
    """
    legit_positions: list[int] = list(np.where(y_sample.values == 0)[0])
    if not legit_positions:
        raise ValueError("No legitimate cases found in SHAP sample")

    selected_positions: list[int] = legit_positions[:n_cases]
    figure, axes = plt.subplots(1, n_cases, figsize=(20, 8))
    axes_list: list[Any] = [axes] if n_cases == 1 else list(axes)

    for subplot_axis, sample_position in zip(axes_list, selected_positions):
        plt.sca(subplot_axis)
        shap.plots.waterfall(shap_explanation[sample_position], max_display=12, show=False)
        subplot_axis.set_title(f"Legit sample idx={int(X_sample.index[sample_position])}", color=LEGIT_COLOR)

    figure.suptitle("SHAP Waterfall — Why These Were Flagged as Legitimate", color=LEGIT_COLOR, fontsize=14)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved SHAP legit waterfall plot: {}", save_path)
    logger.info("Legit waterfall sample indices: {}", [int(X_sample.index[i]) for i in selected_positions])


# %% SHAP Dependence Plots
def plot_shap_dependence(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    save_path: Path,
) -> None:
    """Create dependence plots for top fraud-indicator features.

    Args:
        shap_values: SHAP explanation object.
        X_sample: Feature sample used for SHAP values.
        save_path: Output path for dependence-plot figure.

    Returns:
        None.

    Raises:
        OSError: If figure cannot be saved.
    """
    top_features: list[str] = ["V14", "V10", "V12", "V4", "V11", "V17"]
    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    flattened_axes: list[Any] = list(np.array(axes).flatten())

    for subplot_axis, feature_name in zip(flattened_axes, top_features):
        shap.dependence_plot(
            feature_name,
            shap_values.values,
            X_sample,
            interaction_index="auto",
            ax=subplot_axis,
            show=False,
        )
        subplot_axis.set_title(feature_name)

    figure.suptitle("SHAP Dependence Plots — Top 6 Fraud Indicators", fontsize=14)
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved SHAP dependence plots: {}", save_path)


# %% SHAP Force Plot — HTML Export
def export_shap_force_plots(
    model: object,
    shap_explanation: shap.Explanation,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    save_dir: Path,
) -> None:
    """Export SHAP force plots as HTML for fraud and legitimate cases.

    Args:
        model: Trained tree model (unused in direct export, retained for API parity).
        shap_explanation: SHAP explanation object.
        X_sample: Sampled features.
        y_sample: Sampled labels.
        save_dir: Directory where HTML artifacts are written.

    Returns:
        None.

    Raises:
        OSError: If export paths cannot be written.
    """
    _ = model
    save_dir.mkdir(parents=True, exist_ok=True)

    fraud_positions: list[int] = list(np.where(y_sample.values == 1)[0])[:5]
    legit_positions: list[int] = list(np.where(y_sample.values == 0)[0])[:5]

    for export_index, sample_position in enumerate(fraud_positions, start=1):
        force_plot = shap.force_plot(
            shap_explanation.base_values[sample_position],
            shap_explanation.values[sample_position],
            X_sample.iloc[sample_position],
            matplotlib=False,
        )
        output_path: Path = save_dir / f"force_plot_fraud_{export_index}.html"
        shap.save_html(str(output_path), force_plot)
        logger.info("Saved force plot HTML: {}", output_path)

    for export_index, sample_position in enumerate(legit_positions, start=1):
        force_plot = shap.force_plot(
            shap_explanation.base_values[sample_position],
            shap_explanation.values[sample_position],
            X_sample.iloc[sample_position],
            matplotlib=False,
        )
        output_path: Path = save_dir / f"force_plot_legit_{export_index}.html"
        shap.save_html(str(output_path), force_plot)
        logger.info("Saved force plot HTML: {}", output_path)


# %% SHAP Feature Importance DataFrame
def compute_shap_importance(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    save_path: Path,
) -> pd.DataFrame:
    """Compute and persist rank-ordered SHAP feature importance.

    Args:
        shap_values: SHAP explanation object.
        X_sample: Feature sample used for SHAP values.
        save_path: CSV output path for feature-importance table.

    Returns:
        DataFrame with feature, mean_abs_shap, and rank.

    Raises:
        OSError: If CSV cannot be saved.
    """
    importance_df: pd.DataFrame = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        }
    ).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)
    importance_df = importance_df[["feature", "mean_abs_shap", "rank"]]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(save_path, index=False)
    logger.info("Saved SHAP feature importance CSV: {}", save_path)
    logger.info("Top 15 SHAP features:\n{}", importance_df.head(15).to_string(index=False))
    return importance_df


# %% Main Execution
if __name__ == "__main__":
    run_start_time: float = time.perf_counter()

    model, X_test, y_test = load_artifacts(
        model_path=OUTPUTS_MODELS / "best_model.pkl",
        X_test_path=DATA_PROCESSED / "X_test.parquet",
        y_test_path=DATA_PROCESSED / "y_test.parquet",
    )

    shap_explanation, X_sample = compute_shap_values(model=model, X_test=X_test, sample_size=2000)
    y_sample: pd.Series = y_test.loc[X_sample.index]

    shap_summary_bar_path: Path = OUTPUTS_FIGURES / "08_shap_summary_bar.png"
    shap_beeswarm_path: Path = OUTPUTS_FIGURES / "09_shap_beeswarm.png"
    shap_waterfall_fraud_path: Path = OUTPUTS_FIGURES / "10_shap_waterfall_fraud.png"
    shap_waterfall_legit_path: Path = OUTPUTS_FIGURES / "11_shap_waterfall_legit.png"
    shap_dependence_path: Path = OUTPUTS_FIGURES / "12_shap_dependence.png"
    shap_importance_csv_path: Path = OUTPUTS_REPORTS / "shap_feature_importance.csv"

    plot_shap_summary_bar(shap_values=shap_explanation, X_sample=X_sample, save_path=shap_summary_bar_path)
    plot_shap_beeswarm(shap_values=shap_explanation, X_sample=X_sample, save_path=shap_beeswarm_path)
    plot_shap_waterfall_fraud(
        shap_explanation=shap_explanation,
        X_sample=X_sample,
        y_sample=y_sample,
        save_path=shap_waterfall_fraud_path,
        n_cases=3,
    )
    plot_shap_waterfall_legit(
        shap_explanation=shap_explanation,
        X_sample=X_sample,
        y_sample=y_sample,
        save_path=shap_waterfall_legit_path,
        n_cases=3,
    )
    plot_shap_dependence(shap_values=shap_explanation, X_sample=X_sample, save_path=shap_dependence_path)
    export_shap_force_plots(
        model=model,
        shap_explanation=shap_explanation,
        X_sample=X_sample,
        y_sample=y_sample,
        save_dir=OUTPUTS_REPORTS,
    )
    compute_shap_importance(
        shap_values=shap_explanation,
        X_sample=X_sample,
        save_path=shap_importance_csv_path,
    )

    logger.info("Saved figure: {}", shap_summary_bar_path)
    logger.info("Saved figure: {}", shap_beeswarm_path)
    logger.info("Saved figure: {}", shap_waterfall_fraud_path)
    logger.info("Saved figure: {}", shap_waterfall_legit_path)
    logger.info("Saved figure: {}", shap_dependence_path)
    logger.info("Saved report: {}", shap_importance_csv_path)

    total_runtime_seconds: float = time.perf_counter() - run_start_time
    logger.info("Total SHAP runtime: {:.2f} seconds", total_runtime_seconds)
    logger.info("SHAP Analysis Complete")
