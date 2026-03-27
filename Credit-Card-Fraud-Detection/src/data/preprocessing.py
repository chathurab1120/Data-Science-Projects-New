"""
Production-grade preprocessing and feature engineering for credit card fraud detection.
"""

# %% Imports and Setup
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED, DATA_RAW, OUTPUTS_MODELS, RANDOM_STATE, TEST_SIZE

LOGS_DIR: Path = PROJECT_ROOT / "logs"
LOG_FILEPATH: Path = LOGS_DIR / "preprocessing.log"
SCALER_PATH: Path = OUTPUTS_MODELS / "robust_scaler.pkl"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_FILEPATH, level="INFO", rotation="10 MB", enqueue=True)

# Keep requested tooling imports explicitly visible and initialized.
VALIDATION_SPLITTER: StratifiedKFold = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=RANDOM_STATE
)
REFERENCE_SCALER: StandardScaler = StandardScaler()
REFERENCE_IMB_PIPELINE: ImbPipeline = ImbPipeline(
    steps=[
        ("under_sampler", RandomUnderSampler(random_state=RANDOM_STATE)),
        ("smote", SMOTE(k_neighbors=5, random_state=RANDOM_STATE)),
    ]
)
logger.info("Initialized preprocessing utilities and logger")


# %% Load Raw Data
def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Load raw credit card transaction data.

    Args:
        filepath: Path to the raw CSV dataset.

    Returns:
        Loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the input CSV does not exist.
        pd.errors.ParserError: If the CSV cannot be parsed.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Raw dataset not found: {filepath}")

    dataframe: pd.DataFrame = pd.read_csv(filepath)
    class_distribution: pd.Series = dataframe["Class"].value_counts().sort_index()
    logger.info("Loaded raw data from {}", filepath)
    logger.info("Raw shape: {}", dataframe.shape)
    logger.info("Class distribution:\n{}", class_distribution.to_string())
    return dataframe


# %% Feature Engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready engineered features for fraud learning.

    Args:
        df: Raw transaction dataset.

    Returns:
        DataFrame enriched with engineered features.

    Raises:
        KeyError: If required source columns are missing.
    """
    required_columns: list[str] = ["Amount", "Time", "V14", "V10", "V12"]
    missing_columns: list[str] = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns for feature engineering: {missing_columns}")

    enriched_df: pd.DataFrame = df.copy()
    enriched_df["amount_log"] = np.log1p(enriched_df["Amount"])
    enriched_df["amount_squared"] = enriched_df["Amount"] ** 2
    enriched_df["time_hour"] = (enriched_df["Time"] % (3600 * 24)) / 3600
    enriched_df["time_sin"] = np.sin(2 * np.pi * enriched_df["time_hour"] / 24)
    enriched_df["time_cos"] = np.cos(2 * np.pi * enriched_df["time_hour"] / 24)
    enriched_df["v14_v10_interaction"] = enriched_df["V14"] * enriched_df["V10"]
    enriched_df["v14_v12_interaction"] = enriched_df["V14"] * enriched_df["V12"]
    enriched_df["v10_v12_interaction"] = enriched_df["V10"] * enriched_df["V12"]
    enriched_df["amount_v14"] = enriched_df["Amount"] * enriched_df["V14"]
    enriched_df["high_amount_flag"] = (enriched_df["Amount"] > 1000).astype(int)

    created_features: list[str] = [
        "amount_log",
        "amount_squared",
        "time_hour",
        "time_sin",
        "time_cos",
        "v14_v10_interaction",
        "v14_v12_interaction",
        "v10_v12_interaction",
        "amount_v14",
        "high_amount_flag",
    ]
    logger.info("Created {} engineered features", len(created_features))
    logger.info("Engineered features: {}", created_features)
    return enriched_df


# %% Define Feature Sets
def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], str]:
    """Define final feature list and target column.

    Args:
        df: Feature-engineered dataset.

    Returns:
        Tuple of (feature_columns, target_column).

    Raises:
        KeyError: If required model features are missing.
    """
    pca_features: list[str] = [f"V{i}" for i in range(1, 29)]
    engineered_features: list[str] = [
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
    feature_columns: list[str] = pca_features + engineered_features
    target_column: str = "Class"

    missing_columns: list[str] = [column for column in feature_columns + [target_column] if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required model columns: {missing_columns}")

    logger.info("Final feature count: {}", len(feature_columns))
    logger.info("Target column: {}", target_column)
    return feature_columns, target_column


# %% Train/Test Split
def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create stratified train/test split with class balance checks.

    Args:
        df: Input modeling dataset.
        feature_cols: Feature column names.
        target_col: Target column name.
        test_size: Fraction assigned to test split.
        random_state: Seed for deterministic splitting.

    Returns:
        Tuple of X_train, X_test, y_train, y_test.

    Raises:
        KeyError: If feature or target columns are missing.
        ValueError: If split arguments are invalid.
    """
    missing_columns: list[str] = [column for column in feature_cols + [target_col] if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns for split: {missing_columns}")

    X: pd.DataFrame = df[feature_cols].copy()
    y: pd.Series = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info("X_train shape: {}", X_train.shape)
    logger.info("X_test shape: {}", X_test.shape)
    logger.info("Fraud count train: {}", int((y_train == 1).sum()))
    logger.info("Fraud count test: {}", int((y_test == 1).sum()))

    full_fraud_pct: float = float(y.mean() * 100.0)
    train_fraud_pct: float = float(y_train.mean() * 100.0)
    test_fraud_pct: float = float(y_test.mean() * 100.0)
    logger.info(
        "Fraud percentage full/train/test: {:.4f}% / {:.4f}% / {:.4f}%",
        full_fraud_pct,
        train_fraud_pct,
        test_fraud_pct,
    )

    train_diff: float = abs(train_fraud_pct - full_fraud_pct)
    test_diff: float = abs(test_fraud_pct - full_fraud_pct)
    if train_diff > 0.1 or test_diff > 0.1:
        logger.warning(
            "Fraud ratio drift exceeds 0.1 percentage points: train_diff={:.4f}, test_diff={:.4f}",
            train_diff,
            test_diff,
        )

    return X_train, X_test, y_train, y_test


# %% Scaling
def fit_scaler(X_train: pd.DataFrame, feature_cols: list[str]) -> RobustScaler:
    """Fit and persist a RobustScaler on training data only.

    Args:
        X_train: Training features.
        feature_cols: Ordered list of feature columns to scale.

    Returns:
        Fitted RobustScaler instance.

    Raises:
        KeyError: If any requested feature is absent from X_train.
        ValueError: If training data is empty.
    """
    missing_columns: list[str] = [column for column in feature_cols if column not in X_train.columns]
    if missing_columns:
        raise KeyError(f"Missing columns for scaler fitting: {missing_columns}")

    scaler: RobustScaler = RobustScaler()
    scaler.fit(X_train[feature_cols])
    joblib.dump(scaler, SCALER_PATH)
    logger.info("RobustScaler fitted on training data")
    logger.info("Saved scaler to {}", SCALER_PATH)
    return scaler


def apply_scaler(
    scaler: RobustScaler,
    X: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Apply a fitted scaler and preserve DataFrame structure.

    Args:
        scaler: Pre-fitted RobustScaler object.
        X: Feature matrix to transform.
        feature_cols: Ordered feature columns for transformation.

    Returns:
        Scaled feature DataFrame with original index/columns.

    Raises:
        KeyError: If feature columns are missing from X.
        ValueError: If scaler cannot transform provided data.
    """
    missing_columns: list[str] = [column for column in feature_cols if column not in X.columns]
    if missing_columns:
        raise KeyError(f"Missing columns for scaler transform: {missing_columns}")

    scaled_array: np.ndarray = scaler.transform(X[feature_cols])
    scaled_df: pd.DataFrame = pd.DataFrame(scaled_array, columns=feature_cols, index=X.index)
    logger.info("Applied scaler. Input shape: {}, Output shape: {}", X.shape, scaled_df.shape)
    return scaled_df


# %% Handle Class Imbalance
def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to rebalance the minority fraud class.

    Args:
        X_train: Scaled training feature set.
        y_train: Training targets.
        random_state: Seed for deterministic SMOTE behavior.

    Returns:
        Tuple of resampled features and targets.

    Raises:
        ValueError: If SMOTE fails due to class constraints.
    """
    before_counts: pd.Series = y_train.value_counts().sort_index()
    smote: SMOTE = SMOTE(k_neighbors=5, random_state=random_state)
    X_resampled_array, y_resampled_array = smote.fit_resample(X_train, y_train)

    X_resampled: pd.DataFrame = pd.DataFrame(X_resampled_array, columns=X_train.columns)
    y_resampled: pd.Series = pd.Series(y_resampled_array, name=y_train.name)
    after_counts: pd.Series = y_resampled.value_counts().sort_index()

    synthetic_fraud_samples: int = int(after_counts.get(1, 0) - before_counts.get(1, 0))
    logger.info("Class distribution before SMOTE:\n{}", before_counts.to_string())
    logger.info("Class distribution after SMOTE:\n{}", after_counts.to_string())
    logger.info("Synthetic fraud samples created: {}", synthetic_fraud_samples)
    return X_resampled, y_resampled


# %% Save Processed Data
def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train_resampled: pd.DataFrame,
    y_train_resampled: pd.Series,
    save_dir: Path,
) -> None:
    """Persist processed train/test and resampled artifacts as parquet files.

    Args:
        X_train: Scaled training features.
        X_test: Scaled testing features.
        y_train: Training targets.
        y_test: Testing targets.
        X_train_resampled: SMOTE-resampled training features.
        y_train_resampled: SMOTE-resampled training targets.
        save_dir: Directory for parquet outputs.

    Returns:
        None.

    Raises:
        OSError: If files cannot be written to disk.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    file_map: dict[str, Path] = {
        "X_train": save_dir / "X_train.parquet",
        "X_test": save_dir / "X_test.parquet",
        "y_train": save_dir / "y_train.parquet",
        "y_test": save_dir / "y_test.parquet",
        "X_train_resampled": save_dir / "X_train_resampled.parquet",
        "y_train_resampled": save_dir / "y_train_resampled.parquet",
    }

    X_train.to_parquet(file_map["X_train"], index=False)
    X_test.to_parquet(file_map["X_test"], index=False)
    y_train.to_frame(name="Class").to_parquet(file_map["y_train"], index=False)
    y_test.to_frame(name="Class").to_parquet(file_map["y_test"], index=False)
    X_train_resampled.to_parquet(file_map["X_train_resampled"], index=False)
    y_train_resampled.to_frame(name="Class").to_parquet(file_map["y_train_resampled"], index=False)

    for artifact_name, artifact_path in file_map.items():
        file_size_mb: float = float(artifact_path.stat().st_size / (1024**2))
        logger.info("Saved {} -> {} ({:.2f} MB)", artifact_name, artifact_path, file_size_mb)


# %% Full Pipeline Runner
def run_preprocessing_pipeline(data_path: Path) -> dict[str, Any]:
    """Run complete preprocessing pipeline from raw data to persisted artifacts.

    Args:
        data_path: Path to raw credit card CSV file.

    Returns:
        Dictionary summary containing feature metadata and output shapes.

    Raises:
        Exception: Re-raises any pipeline exception after logging full context.
    """
    start_time: float = time.perf_counter()
    try:
        raw_df: pd.DataFrame = load_raw_data(data_path)
        enriched_df: pd.DataFrame = engineer_features(raw_df)
        feature_cols, target_col = get_feature_columns(enriched_df)

        X_train, X_test, y_train, y_test = split_data(
            df=enriched_df,
            feature_cols=feature_cols,
            target_col=target_col,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        scaler: RobustScaler = fit_scaler(X_train=X_train, feature_cols=feature_cols)
        X_train_scaled: pd.DataFrame = apply_scaler(scaler=scaler, X=X_train, feature_cols=feature_cols)
        X_test_scaled: pd.DataFrame = apply_scaler(scaler=scaler, X=X_test, feature_cols=feature_cols)

        X_train_resampled, y_train_resampled = apply_smote(
            X_train=X_train_scaled,
            y_train=y_train,
            random_state=RANDOM_STATE,
        )

        save_processed_data(
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            X_train_resampled=X_train_resampled,
            y_train_resampled=y_train_resampled,
            save_dir=DATA_PROCESSED,
        )

        elapsed_seconds: float = time.perf_counter() - start_time
        summary: dict[str, Any] = {
            "feature_cols": feature_cols,
            "X_train_shape": X_train_scaled.shape,
            "X_test_shape": X_test_scaled.shape,
            "X_resampled_shape": X_train_resampled.shape,
            "scaler_path": str(SCALER_PATH),
        }
        logger.info("Preprocessing pipeline runtime: {:.2f} seconds", elapsed_seconds)
        return summary
    except (FileNotFoundError, KeyError, OSError, ValueError, RuntimeError) as exception:
        logger.exception("Preprocessing pipeline failed: {}", exception)
        raise


# %% Main Execution
if __name__ == "__main__":
    pipeline_summary: dict[str, Any] = run_preprocessing_pipeline(DATA_RAW / "creditcard.csv")
    logger.info("Pipeline summary: {}", pipeline_summary)
    logger.info("Preprocessing Complete")
