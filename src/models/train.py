"""
Production-grade model training pipeline for credit card fraud detection.
"""

# %% Imports and Setup
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CV_FOLDS, DATA_PROCESSED, OUTPUTS_FIGURES, OUTPUTS_MODELS, OUTPUTS_REPORTS, RANDOM_STATE

LOGS_DIR: Path = PROJECT_ROOT / "logs"
LOG_FILEPATH: Path = LOGS_DIR / "training.log"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)
OUTPUTS_REPORTS.mkdir(parents=True, exist_ok=True)
OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_FILEPATH, level="INFO", rotation="10 MB", enqueue=True)

MODELS: dict[str, Any] = {
    "logistic_regression": LogisticRegression(
        C=0.01,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "xgboost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=577,
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
        n_jobs=-1,
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    ),
}


# %% Load Processed Data
def load_processed_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Load processed train/test and resampled datasets from parquet files.

    Args:
        data_dir: Directory containing all processed parquet artifacts.

    Returns:
        Tuple of X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled.

    Raises:
        FileNotFoundError: If any required parquet file is missing.
        OSError: If parquet files cannot be read.
    """
    filepaths: dict[str, Path] = {
        "X_train": data_dir / "X_train.parquet",
        "X_test": data_dir / "X_test.parquet",
        "y_train": data_dir / "y_train.parquet",
        "y_test": data_dir / "y_test.parquet",
        "X_train_resampled": data_dir / "X_train_resampled.parquet",
        "y_train_resampled": data_dir / "y_train_resampled.parquet",
    }
    for artifact_name, filepath in filepaths.items():
        if not filepath.exists():
            raise FileNotFoundError(f"Missing processed artifact {artifact_name}: {filepath}")

    X_train: pd.DataFrame = pd.read_parquet(filepaths["X_train"])
    X_test: pd.DataFrame = pd.read_parquet(filepaths["X_test"])
    y_train: pd.Series = pd.read_parquet(filepaths["y_train"])["Class"]
    y_test: pd.Series = pd.read_parquet(filepaths["y_test"])["Class"]
    X_train_resampled: pd.DataFrame = pd.read_parquet(filepaths["X_train_resampled"])
    y_train_resampled: pd.Series = pd.read_parquet(filepaths["y_train_resampled"])["Class"]

    logger.info("Loaded processed data from {}", data_dir)
    logger.info("X_train shape: {}", X_train.shape)
    logger.info("X_test shape: {}", X_test.shape)
    logger.info("X_train_resampled shape: {}", X_train_resampled.shape)
    logger.info("y_train distribution:\n{}", y_train.value_counts().sort_index().to_string())
    logger.info("y_test distribution:\n{}", y_test.value_counts().sort_index().to_string())
    logger.info("y_train_resampled distribution:\n{}", y_train_resampled.value_counts().sort_index().to_string())
    return X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled


# %% Evaluation Metrics
def compute_metrics(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Compute classification metrics focused on imbalanced fraud detection.

    Args:
        model_name: Name of the evaluated model.
        y_true: Ground-truth labels.
        y_pred: Predicted class labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        Metrics dictionary containing quality scores and confusion matrix.

    Raises:
        ValueError: If input arrays are inconsistent.
    """
    report: dict[str, Any] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics: dict[str, Any] = {
        "model_name": model_name,
        "accuracy": float(report["accuracy"]),
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
        "f1": float(report["1"]["f1-score"]),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    logger.info(
        "{} metrics | Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f} | ROC-AUC: {:.4f} | "
        "PR-AUC: {:.4f}",
        model_name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
        metrics["pr_auc"],
    )
    logger.info("{} confusion matrix: {}", model_name, metrics["confusion_matrix"])
    return metrics


# %% Train Single Model
def train_model(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, Any], Any]:
    """Train one model and evaluate it on the holdout test set.

    Args:
        model_name: Name of model to train.
        model: Estimator implementing fit, predict, and predict_proba.
        X_train: Training features.
        y_train: Training labels.
        X_test: Holdout test features.
        y_test: Holdout test labels.

    Returns:
        Tuple of (metrics_dict, trained_model).

    Raises:
        ValueError: If model fitting or prediction fails.
    """
    logger.info("Starting training for {}", model_name)
    start_time: float = time.perf_counter()
    model.fit(X_train, y_train)
    y_pred: np.ndarray = model.predict(X_test)
    y_prob: np.ndarray = model.predict_proba(X_test)[:, 1]
    metrics: dict[str, Any] = compute_metrics(model_name=model_name, y_true=y_test, y_pred=y_pred, y_prob=y_prob)
    elapsed_seconds: float = time.perf_counter() - start_time
    logger.info("Finished training {} in {:.2f} seconds", model_name, elapsed_seconds)
    return metrics, model


# %% Cross Validation
def cross_validate_model(
    model_name: str,
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
) -> dict[str, float]:
    """Run stratified cross-validation and summarize stability metrics.

    Args:
        model_name: Name of model being validated.
        model: Estimator to evaluate.
        X: Feature matrix.
        y: Binary target labels.
        cv_folds: Number of stratified folds.

    Returns:
        Dictionary with CV mean and standard deviation for ROC-AUC, PR-AUC, and F1.

    Raises:
        ValueError: If cross-validation fails.
    """
    splitter: StratifiedKFold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    roc_auc_scores: np.ndarray = cross_val_score(model, X, y, cv=splitter, scoring="roc_auc", n_jobs=-1)
    pr_auc_scores: np.ndarray = cross_val_score(
        model, X, y, cv=splitter, scoring="average_precision", n_jobs=-1
    )
    f1_scores: np.ndarray = cross_val_score(model, X, y, cv=splitter, scoring="f1", n_jobs=-1)

    cv_metrics: dict[str, float] = {
        "cv_roc_auc_mean": float(np.mean(roc_auc_scores)),
        "cv_roc_auc_std": float(np.std(roc_auc_scores)),
        "cv_pr_auc_mean": float(np.mean(pr_auc_scores)),
        "cv_pr_auc_std": float(np.std(pr_auc_scores)),
        "cv_f1_mean": float(np.mean(f1_scores)),
        "cv_f1_std": float(np.std(f1_scores)),
    }
    logger.info(
        "{} CV | ROC-AUC: {:.4f} +/- {:.4f} | PR-AUC: {:.4f} +/- {:.4f} | F1: {:.4f} +/- {:.4f}",
        model_name,
        cv_metrics["cv_roc_auc_mean"],
        cv_metrics["cv_roc_auc_std"],
        cv_metrics["cv_pr_auc_mean"],
        cv_metrics["cv_pr_auc_std"],
        cv_metrics["cv_f1_mean"],
        cv_metrics["cv_f1_std"],
    )
    return cv_metrics


# %% Train All Models
def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train_resampled: pd.DataFrame,
    y_train_resampled: pd.Series,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Train, validate, and persist all configured models.

    Args:
        X_train: Original imbalanced training features.
        y_train: Original imbalanced training labels.
        X_test: Holdout test features.
        y_test: Holdout test labels.
        X_train_resampled: SMOTE-resampled training features.
        y_train_resampled: SMOTE-resampled training labels.

    Returns:
        Tuple of (all_metrics_dict, all_models_dict).

    Raises:
        OSError: If model persistence fails.
        ValueError: If training or validation fails.
    """
    all_metrics: dict[str, dict[str, Any]] = {}
    all_models: dict[str, Any] = {}

    for model_name, model in MODELS.items():
        if model_name in {"logistic_regression", "random_forest"}:
            current_X_train: pd.DataFrame = X_train_resampled
            current_y_train: pd.Series = y_train_resampled
            logger.info("{} uses SMOTE-resampled training data", model_name)
        else:
            current_X_train = X_train
            current_y_train = y_train
            logger.info("{} uses original imbalanced training data", model_name)

        metrics, trained_model = train_model(
            model_name=model_name,
            model=model,
            X_train=current_X_train,
            y_train=current_y_train,
            X_test=X_test,
            y_test=y_test,
        )
        cv_metrics: dict[str, float] = cross_validate_model(
            model_name=model_name,
            model=model,
            X=current_X_train,
            y=current_y_train,
            cv_folds=CV_FOLDS,
        )
        metrics.update(cv_metrics)

        model_output_path: Path = OUTPUTS_MODELS / f"{model_name}.pkl"
        joblib.dump(trained_model, model_output_path)
        logger.info("Saved model {} to {}", model_name, model_output_path)

        all_metrics[model_name] = metrics
        all_models[model_name] = trained_model

    metrics_json_path: Path = OUTPUTS_REPORTS / "model_metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(all_metrics, metrics_file, indent=2)
    logger.info("Saved all model metrics to {}", metrics_json_path)
    return all_metrics, all_models


# %% Results Comparison
def compare_models(all_metrics: dict[str, dict[str, Any]], save_path: Path) -> pd.DataFrame:
    """Create sorted model comparison table and persist it as CSV.

    Args:
        all_metrics: Nested dictionary of per-model metrics.
        save_path: CSV output path for comparison table.

    Returns:
        Sorted comparison DataFrame.

    Raises:
        OSError: If CSV cannot be written.
        KeyError: If required metric keys are missing.
    """
    rows: list[dict[str, Any]] = []
    for model_name, metrics in all_metrics.items():
        rows.append(
            {
                "model": model_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            }
        )

    comparison_df: pd.DataFrame = pd.DataFrame(rows).sort_values(by="pr_auc", ascending=False).reset_index(drop=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(save_path, index=False)
    logger.info("Saved model comparison report to {}", save_path)
    logger.info("Model comparison table:\n{}", comparison_df.to_string(index=False))
    return comparison_df


# %% Plot Model Comparison
def plot_model_comparison(all_metrics: dict[str, dict[str, Any]], save_path: Path) -> None:
    """Plot side-by-side model metric bars for quick visual comparison.

    Args:
        all_metrics: Nested dictionary of per-model metrics.
        save_path: Figure output path.

    Returns:
        None.

    Raises:
        OSError: If figure cannot be written.
    """
    model_names: list[str] = list(all_metrics.keys())
    color_map: dict[str, str] = {
        "logistic_regression": "#1f77b4",
        "random_forest": "#2ca02c",
        "xgboost": "#d62728",
        "lightgbm": "#9467bd",
    }
    colors: list[str] = [color_map[name] for name in model_names]

    figure, axes = plt.subplots(2, 2, figsize=(16, 12))
    metric_configs: list[tuple[str, str, Any]] = [
        ("f1", "F1 Score", axes[0, 0]),
        ("roc_auc", "ROC-AUC", axes[0, 1]),
        ("precision", "Precision", axes[1, 0]),
        ("recall", "Recall", axes[1, 1]),
    ]

    for metric_key, metric_title, axis in metric_configs:
        values: list[float] = [float(all_metrics[name][metric_key]) for name in model_names]
        bars = axis.bar(model_names, values, color=colors)
        axis.set_title(metric_title)
        axis.set_ylim(0.0, 1.0)
        axis.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved model comparison figure: {}", save_path)


# %% Plot ROC and PR Curves
def plot_roc_pr_curves(
    all_models: dict[str, Any],
    all_metrics: dict[str, dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path,
) -> None:
    """Plot ROC and Precision-Recall curves for all trained models.

    Args:
        all_models: Mapping of model names to trained estimators.
        all_metrics: Mapping of model names to metrics.
        X_test: Holdout test features.
        y_test: Holdout test labels.
        save_path: Figure output path.

    Returns:
        None.

    Raises:
        OSError: If figure cannot be written.
    """
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    roc_axis = axes[0]
    pr_axis = axes[1]
    color_map: dict[str, str] = {
        "logistic_regression": "#1f77b4",
        "random_forest": "#2ca02c",
        "xgboost": "#d62728",
        "lightgbm": "#9467bd",
    }

    for model_name, model in all_models.items():
        y_prob: np.ndarray = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        roc_auc_value: float = float(all_metrics[model_name]["roc_auc"])
        pr_auc_value: float = float(all_metrics[model_name]["pr_auc"])
        curve_color: str = color_map[model_name]

        roc_axis.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc_value:.3f})", color=curve_color, linewidth=2)
        pr_axis.plot(recall, precision, label=f"{model_name} (AP={pr_auc_value:.3f})", color=curve_color, linewidth=2)

    roc_axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Random classifier")
    roc_axis.set_title("ROC Curves")
    roc_axis.set_xlabel("False Positive Rate")
    roc_axis.set_ylabel("True Positive Rate")
    roc_axis.legend()

    fraud_prevalence: float = 0.001727
    pr_axis.axhline(
        y=fraud_prevalence,
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label=f"Baseline prevalence ({fraud_prevalence * 100:.4f}%)",
    )
    pr_axis.set_title("Precision-Recall Curves")
    pr_axis.set_xlabel("Recall")
    pr_axis.set_ylabel("Precision")
    pr_axis.legend()

    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved ROC/PR curves figure: {}", save_path)


# %% Select Best Model
def select_best_model(
    all_metrics: dict[str, dict[str, Any]],
    all_models: dict[str, Any],
    save_path: Path,
) -> tuple[str, Any]:
    """Select and persist the best model based on PR-AUC.

    Args:
        all_metrics: Mapping of model names to metrics.
        all_models: Mapping of model names to trained estimators.
        save_path: Output path for best model artifact.

    Returns:
        Tuple of (best_model_name, best_model_object).

    Raises:
        OSError: If best model artifacts cannot be written.
        ValueError: If metric/model mappings are empty.
    """
    if not all_metrics or not all_models:
        raise ValueError("all_metrics and all_models must be non-empty")

    best_model_name: str = max(all_metrics, key=lambda model_name: float(all_metrics[model_name]["pr_auc"]))
    best_model: Any = all_models[best_model_name]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, save_path)
    best_model_name_path: Path = save_path.parent / "best_model_name.txt"
    best_model_name_path.write_text(best_model_name, encoding="utf-8")

    logger.info("Best model selected by PR-AUC: {}", best_model_name)
    logger.info("Best model metrics: {}", all_metrics[best_model_name])
    logger.info("Saved best model artifact: {}", save_path)
    logger.info("Saved best model name file: {}", best_model_name_path)
    return best_model_name, best_model


# %% Main Execution
if __name__ == "__main__":
    pipeline_start_time: float = time.perf_counter()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_resampled,
        y_train_resampled,
    ) = load_processed_data(DATA_PROCESSED)

    all_metrics_dict, all_models_dict = train_all_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_resampled=X_train_resampled,
        y_train_resampled=y_train_resampled,
    )

    comparison_path: Path = OUTPUTS_REPORTS / "model_comparison.csv"
    comparison_df: pd.DataFrame = compare_models(all_metrics=all_metrics_dict, save_path=comparison_path)

    model_comparison_figure_path: Path = OUTPUTS_FIGURES / "06_model_comparison.png"
    plot_model_comparison(all_metrics=all_metrics_dict, save_path=model_comparison_figure_path)

    roc_pr_figure_path: Path = OUTPUTS_FIGURES / "07_roc_pr_curves.png"
    plot_roc_pr_curves(
        all_models=all_models_dict,
        all_metrics=all_metrics_dict,
        X_test=X_test,
        y_test=y_test,
        save_path=roc_pr_figure_path,
    )

    best_model_path: Path = OUTPUTS_MODELS / "best_model.pkl"
    winner_model_name, _ = select_best_model(
        all_metrics=all_metrics_dict,
        all_models=all_models_dict,
        save_path=best_model_path,
    )

    total_runtime_seconds: float = time.perf_counter() - pipeline_start_time
    logger.info("Training runtime: {:.2f} seconds", total_runtime_seconds)
    logger.info("Winner model: {}", winner_model_name)
    logger.info("Top model comparison row:\n{}", comparison_df.head(1).to_string(index=False))
    logger.info("Training Complete")
