# -*- coding: utf-8 -*-
"""
Script: 01_data_loading.py
Purpose: Load the LendingClub loan dataset, perform initial inspection,
         engineer the binary target variable (default_flag), and save
         a summary report.
Inputs:  data/accepted_2007_to_2018Q4.csv  (~2.26M rows)
Outputs: outputs/results/data_summary.csv
         outputs/results/class_distribution.csv
         outputs/results/missing_values.csv
"""

# %% [0] Imports and configuration
import pandas as pd
import numpy as np
import yaml
import warnings
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

RANDOM_STATE = config["project"]["random_state"]
DATA_PATH = Path(config["paths"]["data_raw"])
RESULTS_PATH = Path(config["paths"]["outputs_results"])
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STAGE 01 — Data Loading & Initial Inspection")
print("=" * 60)


# %% [1] Load raw data
# LendingClub CSV is large (~1.3 GB compressed). We load a representative
# sample first to inspect structure, then load full data for modelling.
# The last two rows of the CSV are metadata — skip them with skipfooter.

print(f"\n[1] Loading data from: {DATA_PATH}")

try:
    df = pd.read_csv(DATA_PATH)
    # Drop last 2 rows — LendingClub appends metadata footer rows
    df = df[:-2].copy()
    print(f"    Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
except FileNotFoundError:
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Download from https://www.kaggle.com/datasets/wordsforthewise/lending-club "
        "and place accepted_2007_to_2018Q4.csv in the data/ folder."
    )


# %% [2] Basic inspection
# Always start with shape, dtypes, and a quick look at the data.
# This builds intuition before any transformation.

print("\n[2] Basic inspection")
print(f"    Columns: {df.shape[1]}")
print(f"    Rows:    {df.shape[0]:,}")
print(f"\n    Data types breakdown:")
print(df.dtypes.value_counts())
print(f"\n    First 3 rows (selected columns):")
peek_cols = ["loan_amnt", "int_rate", "grade", "loan_status", "annual_inc", "dti"]
print(df[peek_cols].head(3).to_string())


# %% [3] Understand the target column
# loan_status has many values — we only keep the ones we can definitively
# label as good (Fully Paid) or bad (Charged Off / Default).
# Current loans, late loans etc. are ambiguous — exclude them.

print("\n[3] Target column — loan_status value counts")
status_counts = df["loan_status"].value_counts()
print(status_counts.to_string())

# Interview point: We use Charged Off + Default as the bad class.
# "Late" loans are excluded because their final outcome is unknown.
# Including them would introduce label noise.
positive_classes = config["data"]["positive_classes"]   # bad loans
negative_classes = config["data"]["negative_classes"]   # good loans

keep_mask = df["loan_status"].isin(positive_classes + negative_classes)
df = df[keep_mask].copy()
print(f"\n    After filtering ambiguous statuses: {df.shape[0]:,} rows")


# %% [4] Engineer binary target variable
# 1 = default (bad), 0 = fully paid (good)
# This is the standard convention in credit risk: 1 = bad event

print("\n[4] Engineering binary target: default_flag")
df["default_flag"] = df["loan_status"].isin(positive_classes).astype(int)

default_rate = df["default_flag"].mean()
good_count = (df["default_flag"] == 0).sum()
bad_count = (df["default_flag"] == 1).sum()

print(f"    Good loans (0): {good_count:,}  ({100 - default_rate * 100:.1f}%)")
print(f"    Bad loans  (1): {bad_count:,}  ({default_rate * 100:.1f}%)")
print(f"    Overall default rate: {default_rate:.4f}")
print(f"\n    Interview note: ~20-25% bad rate is typical for LendingClub.")
print(f"    Class imbalance ratio approx 1:{int(good_count / bad_count)}")


# %% [5] Identify columns with high missingness
# We will drop columns with >40% missing in the cleaning stage.
# Flag them here so the analyst can review before dropping.

print("\n[5] Missing value analysis")
MISSING_THRESHOLD = config["data"]["missing_threshold"]

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct
}).sort_values("missing_pct", ascending=False)

high_missing = missing_df[missing_df["missing_pct"] > MISSING_THRESHOLD * 100]
print(f"    Columns with >{MISSING_THRESHOLD * 100:.0f}% missing: {len(high_missing)}")
print(high_missing.head(20).to_string())


# %% [6] Parse date columns
# issue_d and earliest_cr_line are strings — parse to datetime now
# so downstream scripts can compute credit_history_months correctly.

print("\n[6] Parsing date columns")
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
df["issue_year"] = df["issue_d"].dt.year

print(f"    issue_d range:          {df['issue_d'].min().date()} → {df['issue_d'].max().date()}")
print(f"    earliest_cr_line range: {df['earliest_cr_line'].min().date()} → {df['earliest_cr_line'].max().date()}")
print(f"    Loan vintage distribution (by year):")
print(df["issue_year"].value_counts().sort_index().to_string())


# %% [7] Define columns to keep for modelling
# We explicitly select the columns relevant to origination-time prediction.
# This is the leakage prevention step — anything known only AFTER the loan
# event is excluded here so it cannot accidentally slip into the model.

print("\n[7] Selecting origination-time features")

# Post-event leakage columns — known only after loan outcome
LEAKAGE_COLS = [
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_amnt", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d",
    "out_prncp", "out_prncp_inv", "funded_amnt_inv"
]

# Core feature columns available at origination
FEATURE_COLS = [
    "loan_amnt", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "mths_since_last_delinq", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "initial_list_status",
    "application_type", "mort_acc", "pub_rec_bankruptcies",
    "issue_d", "earliest_cr_line", "issue_year"
]

# Target and ID
TARGET_COLS = ["loan_status", "default_flag"]

keep_cols = [c for c in FEATURE_COLS + TARGET_COLS if c in df.columns]
df_clean = df[keep_cols].copy()

print(f"    Starting columns: {df.shape[1]}")
print(f"    Leakage columns excluded: {len(LEAKAGE_COLS)}")
print(f"    Working feature set: {df_clean.shape[1]} columns")
print(f"    Working dataframe shape: {df_clean.shape}")


# %% [8] Save outputs
# Save summary statistics, class distribution, and missing value report.
# These feed into the EDA notebook and README results table.

print("\n[8] Saving outputs")

# Data summary statistics
summary = df_clean.describe(include="all").T
summary_path = RESULTS_PATH / "data_summary.csv"
summary.to_csv(summary_path)
print(f"    Saved: {summary_path}")

# Class distribution
class_dist = pd.DataFrame({
    "label": ["Good (Fully Paid)", "Bad (Default/Charged Off)"],
    "count": [good_count, bad_count],
    "percentage": [f"{100-default_rate*100:.2f}%", f"{default_rate*100:.2f}%"]
})
class_dist_path = RESULTS_PATH / "class_distribution.csv"
class_dist.to_csv(class_dist_path, index=False)
print(f"    Saved: {class_dist_path}")

# Missing value report
missing_path = RESULTS_PATH / "missing_values.csv"
missing_df.to_csv(missing_path)
print(f"    Saved: {missing_path}")

# Save the filtered working dataframe for use in Stage 02
# We use parquet for speed and type preservation on large datasets
working_data_path = Path("data/working_data.parquet")
df_clean.to_parquet(working_data_path, index=False)
print(f"    Saved working dataframe: {working_data_path}")


# %% [9] Final summary
print("\n" + "=" * 60)
print("STAGE 01 COMPLETE")
print("=" * 60)
print(f"  Final dataset shape : {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
print(f"  Default rate        : {default_rate:.4f} ({default_rate*100:.2f}%)")
print(f"  Date range          : {df_clean['issue_d'].min().date()} → {df_clean['issue_d'].max().date()}")
print(f"  Outputs saved to    : {RESULTS_PATH}")
print(f"\n  Next step: Run scripts/02_eda.py")
