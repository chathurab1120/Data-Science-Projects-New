"""
Production-grade exploratory data analysis for credit card fraud detection.
"""

# %% Imports and Setup
from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from loguru import logger
from scipy.stats import ks_2samp

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_RAW, OUTPUTS_FIGURES, RANDOM_STATE, ROOT_DIR

plt.style.use("seaborn-v0_8-darkgrid")
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

LOGS_DIR: Path = ROOT_DIR / "logs"
OUTPUTS_REPORTS: Path = ROOT_DIR / "outputs" / "reports"
LOG_FILEPATH: Path = LOGS_DIR / "eda.log"
FIGURE_SIZE: tuple[int, int] = (12, 6)

LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUTS_REPORTS.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_FILEPATH, level="INFO", rotation="10 MB", enqueue=True)


# %% Load Data
def load_data(filepath: Path) -> pd.DataFrame:
    """Load and validate the credit card dataset.

    Args:
        filepath: Absolute or relative path to the CSV dataset.

    Returns:
        A validated pandas DataFrame.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If required columns are missing.
        pd.errors.ParserError: If pandas cannot parse the CSV file.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    dataframe: pd.DataFrame = pd.read_csv(filepath)
    logger.info("Loaded dataset from {}", filepath)
    logger.info("Dataset shape: {}", dataframe.shape)
    logger.info("Dtypes summary:\n{}", dataframe.dtypes.value_counts().to_string())

    memory_mb: float = float(dataframe.memory_usage(deep=True).sum() / (1024**2))
    logger.info("Memory usage: {:.2f} MB", memory_mb)

    required_columns: list[str] = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
    missing_columns: list[str] = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    logger.info("Required columns validated successfully")
    return dataframe


# %% Basic Dataset Info
def basic_info(dataframe: pd.DataFrame) -> dict[str, float | int]:
    """Compute and log foundational dataset metadata.

    Args:
        dataframe: Input transaction dataset.

    Returns:
        A dictionary of dataset-wide summary metrics.

    Raises:
        KeyError: If the Class column is missing.
    """
    if "Class" not in dataframe.columns:
        raise KeyError("Class column is required for basic_info")

    n_rows: int = int(dataframe.shape[0])
    n_cols: int = int(dataframe.shape[1])
    n_duplicates: int = int(dataframe.duplicated().sum())
    n_missing: int = int(dataframe.isna().sum().sum())
    fraud_count: int = int((dataframe["Class"] == 1).sum())
    legit_count: int = int((dataframe["Class"] == 0).sum())
    fraud_pct: float = float((fraud_count / n_rows) * 100.0)
    legit_pct: float = float((legit_count / n_rows) * 100.0)

    info: dict[str, float | int] = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_duplicates": n_duplicates,
        "n_missing": n_missing,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_pct": round(fraud_pct, 4),
        "legit_pct": round(legit_pct, 4),
    }

    for metric_name, metric_value in info.items():
        logger.info("{}: {}", metric_name, metric_value)

    summary_table: pd.DataFrame = pd.DataFrame([info]).T.rename(columns={0: "value"})
    logger.info("Basic info summary table:\n{}", summary_table.to_string())
    return info


# %% Class Imbalance Visualization
def plot_class_distribution(dataframe: pd.DataFrame, save_path: Path) -> None:
    """Plot class count and percentage split.

    Args:
        dataframe: Input transaction dataset.
        save_path: Output image path for the generated plot.

    Returns:
        None.

    Raises:
        KeyError: If required columns are missing.
    """
    if "Class" not in dataframe.columns:
        raise KeyError("Class column is required for class distribution plot")

    plot_df: pd.DataFrame = dataframe.copy()
    plot_df["ClassLabel"] = plot_df["Class"].astype(str).map({"0": "Legitimate", "1": "Fraud"})
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    class_counts: pd.Series = dataframe["Class"].value_counts().sort_index()

    sns.countplot(
        data=plot_df,
        x="ClassLabel",
        order=["Legitimate", "Fraud"],
        ax=axes[0],
        palette={"Legitimate": "green", "Fraud": "red"},
    )
    axes[0].set_title("Class Counts")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")

    for patch, class_label in zip(axes[0].patches, [0, 1]):
        count_value: int = int(class_counts.get(class_label, 0))
        axes[0].annotate(
            str(count_value),
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 5),
            textcoords="offset points",
        )

    labels: list[str] = ["Legitimate", "Fraud"]
    sizes: list[int] = [int(class_counts.get(0, 0)), int(class_counts.get(1, 0))]
    axes[1].pie(sizes, labels=labels, colors=["green", "red"], autopct="%1.2f%%", startangle=90)
    axes[1].set_title("Class Percentage Split")

    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved class distribution figure: {}", save_path)


# %% Transaction Amount Analysis
def plot_amount_analysis(dataframe: pd.DataFrame, save_path: Path) -> None:
    """Plot amount distributions and group-wise spread.

    Args:
        dataframe: Input transaction dataset.
        save_path: Output image path for the generated plot.

    Returns:
        None.

    Raises:
        KeyError: If required columns are missing.
    """
    if "Class" not in dataframe.columns or "Amount" not in dataframe.columns:
        raise KeyError("Class and Amount columns are required for amount analysis")

    plot_df: pd.DataFrame = dataframe.copy()
    plot_df["ClassLabel"] = plot_df["Class"].astype(str).map({"0": "Legitimate", "1": "Fraud"})
    figure, axes = plt.subplots(2, 2, figsize=(16, 12))
    legit_data: pd.DataFrame = dataframe[dataframe["Class"] == 0]
    fraud_data: pd.DataFrame = dataframe[dataframe["Class"] == 1]

    sns.histplot(legit_data["Amount"] + 1.0, bins=50, ax=axes[0, 0], color="green", log_scale=True)
    axes[0, 0].set_title("Legitimate Transaction Amounts (Log Scale)")
    axes[0, 0].set_xlabel("Amount + 1")

    sns.histplot(fraud_data["Amount"] + 1.0, bins=50, ax=axes[0, 1], color="red", log_scale=True)
    axes[0, 1].set_title("Fraud Transaction Amounts (Log Scale)")
    axes[0, 1].set_xlabel("Amount + 1")

    sns.boxplot(
        data=plot_df,
        x="ClassLabel",
        y="Amount",
        order=["Legitimate", "Fraud"],
        ax=axes[1, 0],
        palette={"Legitimate": "green", "Fraud": "red"},
    )
    axes[1, 0].set_title("Amount by Class (Boxplot)")
    axes[1, 0].set_xlabel("Class")
    axes[1, 0].set_ylabel("Amount")

    sns.violinplot(
        data=plot_df,
        x="ClassLabel",
        y="Amount",
        order=["Legitimate", "Fraud"],
        ax=axes[1, 1],
        palette={"Legitimate": "green", "Fraud": "red"},
    )
    axes[1, 1].set_title("Amount by Class (Violin Plot)")
    axes[1, 1].set_xlabel("Class")
    axes[1, 1].set_ylabel("Amount")

    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved amount analysis figure: {}", save_path)


# %% Time Analysis
def plot_time_analysis(dataframe: pd.DataFrame, save_path: Path) -> None:
    """Plot transaction volume and fraud-rate progression across time.

    Args:
        dataframe: Input transaction dataset.
        save_path: Output image path for the generated plot.

    Returns:
        None.

    Raises:
        KeyError: If required columns are missing.
    """
    if "Class" not in dataframe.columns or "Time" not in dataframe.columns:
        raise KeyError("Class and Time columns are required for time analysis")

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    legit_time: pd.Series = dataframe.loc[dataframe["Class"] == 0, "Time"]
    fraud_time: pd.Series = dataframe.loc[dataframe["Class"] == 1, "Time"]

    sns.histplot(legit_time, bins=100, ax=axes[0], color="green", alpha=0.6, stat="count", label="Legit")
    sns.histplot(fraud_time, bins=100, ax=axes[0], color="red", alpha=0.6, stat="count", label="Fraud")
    axes[0].set_title("Transaction Volume Over Time")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Transaction Count")
    axes[0].legend()

    time_bins: pd.Series = pd.cut(dataframe["Time"], bins=48, include_lowest=True)
    fraud_rate: pd.Series = dataframe.groupby(time_bins)["Class"].mean() * 100.0
    fraud_rate.index = fraud_rate.index.astype(str)
    axes[1].plot(fraud_rate.index, fraud_rate.values, color="red", linewidth=2)
    axes[1].set_title("Fraud Rate Over Time (48 Bins)")
    axes[1].set_xlabel("Time Bin")
    axes[1].set_ylabel("Fraud Rate (%)")
    axes[1].tick_params(axis="x", rotation=90)

    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved time analysis figure: {}", save_path)


# %% Feature Distributions (V1-V28)
def plot_feature_distributions(dataframe: pd.DataFrame, save_path: Path) -> None:
    """Plot KDE overlays for V1 through V28 by class.

    Args:
        dataframe: Input transaction dataset.
        save_path: Output image path for the generated plot.

    Returns:
        None.

    Raises:
        KeyError: If any V-feature columns are missing.
    """
    feature_columns: list[str] = [f"V{i}" for i in range(1, 29)]
    missing_features: list[str] = [column for column in feature_columns if column not in dataframe.columns]
    if missing_features:
        raise KeyError(f"Missing V feature columns: {missing_features}")

    figure, axes = plt.subplots(7, 4, figsize=(28, 35))
    flattened_axes: np.ndarray[Any, Any] = axes.flatten()

    for index, feature_name in enumerate(feature_columns):
        axis = flattened_axes[index]
        sns.kdeplot(
            data=dataframe.loc[dataframe["Class"] == 0, feature_name],
            ax=axis,
            color="blue",
            fill=True,
            alpha=0.3,
            label="Legit" if index == 0 else None,
        )
        sns.kdeplot(
            data=dataframe.loc[dataframe["Class"] == 1, feature_name],
            ax=axis,
            color="red",
            fill=True,
            alpha=0.3,
            label="Fraud" if index == 0 else None,
        )
        axis.set_title(f"{feature_name} Distribution")
        if index == 0:
            axis.legend()

    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved feature distribution figure: {}", save_path)


# %% Correlation Analysis
def plot_correlation_matrix(dataframe: pd.DataFrame, save_path: Path) -> None:
    """Plot feature correlation heatmap excluding Time.

    Args:
        dataframe: Input transaction dataset.
        save_path: Output image path for the generated plot.

    Returns:
        None.

    Raises:
        KeyError: If expected numeric columns are missing.
    """
    columns_to_use: list[str] = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    missing_columns: list[str] = [column for column in columns_to_use if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Missing columns for correlation matrix: {missing_columns}")

    correlation_matrix: pd.DataFrame = dataframe[columns_to_use].corr()
    figure, axis = plt.subplots(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", ax=axis)
    axis.set_title("Feature Correlation Matrix")

    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved correlation matrix figure: {}", save_path)


# %% Statistical Summary
def compute_statistical_summary(dataframe: pd.DataFrame, save_path: Path) -> None:
    """Compute class-wise statistics and KS-test metrics by feature.

    Args:
        dataframe: Input transaction dataset.
        save_path: CSV output path for statistical summary.

    Returns:
        None.

    Raises:
        KeyError: If required columns are missing.
    """
    target_features: list[str] = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    missing_columns: list[str] = [column for column in target_features + ["Class"] if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Missing columns for statistical summary: {missing_columns}")

    rows: list[dict[str, float | str]] = []
    fraud_df: pd.DataFrame = dataframe[dataframe["Class"] == 1]
    legit_df: pd.DataFrame = dataframe[dataframe["Class"] == 0]

    for feature_name in target_features:
        fraud_values: pd.Series = fraud_df[feature_name].dropna()
        legit_values: pd.Series = legit_df[feature_name].dropna()
        ks_result = ks_2samp(fraud_values, legit_values)
        rows.append(
            {
                "feature": feature_name,
                "mean_fraud": float(fraud_values.mean()),
                "mean_legit": float(legit_values.mean()),
                "std_fraud": float(fraud_values.std()),
                "std_legit": float(legit_values.std()),
                "ks_statistic": float(ks_result.statistic),
                "ks_pvalue": float(ks_result.pvalue),
            }
        )

    summary_df: pd.DataFrame = pd.DataFrame(rows).sort_values(by="ks_statistic", ascending=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(save_path, index=False)
    logger.info("Saved statistical summary report: {}", save_path)
    logger.info("Top 10 most discriminative features:\n{}", summary_df.head(10).to_string(index=False))


# %% Main Execution
if __name__ == "__main__":
    start_time_seconds: float = time.perf_counter()
    dataset_path: Path = DATA_RAW / "creditcard.csv"

    logger.info("Starting EDA pipeline from root: {}", ROOT_DIR)
    logger.info("Random seed set to {}", RANDOM_STATE)

    df: pd.DataFrame = load_data(dataset_path)
    info_dict: dict[str, float | int] = basic_info(df)
    logger.info("Basic info keys: {}", list(info_dict.keys()))

    class_distribution_path: Path = OUTPUTS_FIGURES / "01_class_distribution.png"
    amount_analysis_path: Path = OUTPUTS_FIGURES / "02_amount_analysis.png"
    time_analysis_path: Path = OUTPUTS_FIGURES / "03_time_analysis.png"
    feature_distribution_path: Path = OUTPUTS_FIGURES / "04_feature_distributions.png"
    correlation_matrix_path: Path = OUTPUTS_FIGURES / "05_correlation_matrix.png"
    summary_report_path: Path = OUTPUTS_REPORTS / "feature_statistical_summary.csv"

    plot_class_distribution(df, class_distribution_path)
    plot_amount_analysis(df, amount_analysis_path)
    plot_time_analysis(df, time_analysis_path)
    plot_feature_distributions(df, feature_distribution_path)
    plot_correlation_matrix(df, correlation_matrix_path)
    compute_statistical_summary(df, summary_report_path)

    elapsed_seconds: float = time.perf_counter() - start_time_seconds
    logger.info("EDA Complete in {:.2f} seconds", elapsed_seconds)
    logger.info("Saved figure: {}", class_distribution_path)
    logger.info("Saved figure: {}", amount_analysis_path)
    logger.info("Saved figure: {}", time_analysis_path)
    logger.info("Saved figure: {}", feature_distribution_path)
    logger.info("Saved figure: {}", correlation_matrix_path)
    logger.info("Saved report: {}", summary_report_path)
