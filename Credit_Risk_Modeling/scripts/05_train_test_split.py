"""
Script: 05_train_test_split.py
Purpose: Split the model-ready dataset chronologically into train and test sets.
         Train: loans issued 2007-2015. Test: loans issued 2016-2018.
         This mimics out-of-time (OOT) validation — the industry standard
         for credit risk models. Random split would leak future patterns
         into training and produce overly optimistic AUC.
Inputs:  data/model_data.parquet
Outputs: data/X_train.parquet, data/X_test.parquet
         data/y_train.parquet, data/y_test.parquet
         outputs/results/split_summary.csv
"""

# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# %% [0] Imports and configuration
import pandas as pd
import numpy as np
import yaml
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

RESULTS_PATH = Path(config["paths"]["outputs_results"])
TRAIN_CUTOFF  = config["data"]["train_cutoff_year"]
TEST_START    = config["data"]["test_start_year"]
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STAGE 05 — Chronological Train / Test Split")
print("=" * 60)
print(f"  Train period : 2007 - {TRAIN_CUTOFF}")
print(f"  Test period  : {TEST_START} - 2018")
print("  Split type   : Out-of-Time (OOT) — industry standard")


# %% [1] Load model data
print("\n[1] Loading model data")
df = pd.read_parquet("data/model_data.parquet")
print(f"    Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")


# %% [2] Define feature columns and target
# issue_year is metadata used for splitting — not a model feature.
# We drop it from X to prevent data leakage via time information.

print("\n[2] Defining features and target")

meta_cols    = ["default_flag", "issue_year"]
feature_cols = [c for c in df.columns if c not in meta_cols]

print(f"    Feature columns : {len(feature_cols)}")
print("    Target column   : default_flag")
print(f"    issue_year range: {df['issue_year'].min()} - {df['issue_year'].max()}")


# %% [3] Chronological split
# Train on 2007-2015, test on 2016-2018.
# This is out-of-time validation — the model is tested on loans it has
# never seen, originated after the training period ends.
# Interview point: random split would allow the model to learn from
# future data patterns, making performance look better than it really is.

print("\n[3] Performing chronological split")

train_mask = df["issue_year"] <= TRAIN_CUTOFF
test_mask  = df["issue_year"] >= TEST_START

df_train = df[train_mask].copy()
df_test  = df[test_mask].copy()

X_train = df_train[feature_cols]
y_train = df_train["default_flag"]
X_test  = df_test[feature_cols]
y_test  = df_test["default_flag"]

print(f"    Train set : {X_train.shape[0]:,} rows ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"    Test set  : {X_test.shape[0]:,} rows ({X_test.shape[0]/len(df)*100:.1f}%)")


# %% [4] Validate split quality
# Check that default rates are similar between train and test.
# A large difference would indicate distribution shift (PSI issue).
# Also check year coverage to confirm no overlap.

print("\n[4] Validating split quality")

train_default_rate = y_train.mean()
test_default_rate  = y_test.mean()
rate_diff          = abs(train_default_rate - test_default_rate)

print(f"    Train default rate : {train_default_rate:.4f} ({train_default_rate*100:.2f}%)")
print(f"    Test default rate  : {test_default_rate:.4f} ({test_default_rate*100:.2f}%)")
print(f"    Absolute difference: {rate_diff:.4f}")

if rate_diff > 0.05:
    print("    WARNING: Default rate difference > 5% — check for distribution shift")
else:
    print("    OK: Default rates are consistent between train and test")

print(f"\n    Train year range: {df_train['issue_year'].min()} - {df_train['issue_year'].max()}")
print(f"    Test year range : {df_test['issue_year'].min()} - {df_test['issue_year'].max()}")

# Confirm no overlap
train_years = set(df_train["issue_year"].unique())
test_years  = set(df_test["issue_year"].unique())
overlap     = train_years.intersection(test_years)
print(f"    Year overlap between train and test: {overlap if overlap else 'None — clean split'}")

# Class balance in train set
train_class_counts = y_train.value_counts()
print("\n    Train class distribution:")
print(f"      Good (0): {train_class_counts[0]:,} ({train_class_counts[0]/len(y_train)*100:.1f}%)")
print(f"      Bad  (1): {train_class_counts[1]:,} ({train_class_counts[1]/len(y_train)*100:.1f}%)")

# Class balance in test set
test_class_counts = y_test.value_counts()
print("\n    Test class distribution:")
print(f"      Good (0): {test_class_counts[0]:,} ({test_class_counts[0]/len(y_test)*100:.1f}%)")
print(f"      Bad  (1): {test_class_counts[1]:,} ({test_class_counts[1]/len(y_test)*100:.1f}%)")


# %% [5] Check for non-numeric columns
# Tree models handle most types but logistic regression needs all-numeric.
# Flag any object columns here so the modelling stage can handle them.

print("\n[5] Checking feature dtypes")
non_numeric = X_train.select_dtypes(include=["object"]).columns.tolist()
if non_numeric:
    print(f"    Non-numeric columns found: {non_numeric}")
    print("    These will need label encoding before logistic regression")
    # Label encode for compatibility
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in non_numeric:
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col]  = le.transform(X_test[col].astype(str))
    print(f"    Label encoded {len(non_numeric)} columns")
else:
    print("    All feature columns are numeric — ready for modelling")


# %% [6] Save splits
print("\n[6] Saving train/test splits")

data_dir = Path("data")

X_train.to_parquet(data_dir / "X_train.parquet", index=False)
X_test.to_parquet(data_dir  / "X_test.parquet",  index=False)
y_train.to_frame().to_parquet(data_dir / "y_train.parquet", index=False)
y_test.to_frame().to_parquet(data_dir  / "y_test.parquet",  index=False)

print(f"    Saved: data/X_train.parquet  {X_train.shape}")
print(f"    Saved: data/X_test.parquet   {X_test.shape}")
print(f"    Saved: data/y_train.parquet  {y_train.shape}")
print(f"    Saved: data/y_test.parquet   {y_test.shape}")

split_summary = pd.DataFrame([
    {"set": "train", "rows": len(X_train), "features": len(feature_cols),
     "default_rate": round(train_default_rate, 4),
     "year_min": int(df_train["issue_year"].min()),
     "year_max": int(df_train["issue_year"].max())},
    {"set": "test",  "rows": len(X_test),  "features": len(feature_cols),
     "default_rate": round(test_default_rate, 4),
     "year_min": int(df_test["issue_year"].min()),
     "year_max": int(df_test["issue_year"].max())},
])
split_summary.to_csv(RESULTS_PATH / "split_summary.csv", index=False)
print("    Saved: outputs/results/split_summary.csv")


# %% [7] Final summary
print("\n" + "=" * 60)
print("STAGE 05 COMPLETE")
print("=" * 60)
print(f"  X_train : {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
print(f"  X_test  : {X_test.shape[0]:,} rows x {X_test.shape[1]} features")
print(f"  y_train : {y_train.shape[0]:,} labels  (default rate: {train_default_rate:.4f})")
print(f"  y_test  : {y_test.shape[0]:,} labels  (default rate: {test_default_rate:.4f})")
print("\n  Next step: Run scripts/06_baseline_model.py")
