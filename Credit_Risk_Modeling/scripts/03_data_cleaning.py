"""
Script: 03_data_cleaning.py
Purpose: Clean the working dataset — drop leakage columns, handle missing
         values, parse and clean string columns, cap outliers at 99th
         percentile, and save a clean analysis-ready dataframe.
Inputs:  data/working_data.parquet
Outputs: data/clean_data.parquet
         outputs/results/cleaning_report.csv
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
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STAGE 03 — Data Cleaning")
print("=" * 60)


# %% [1] Load working data
print("\n[1] Loading working data")
df = pd.read_parquet("data/working_data.parquet")
print(f"    Shape on load: {df.shape[0]:,} rows x {df.shape[1]} columns")
shape_before = df.shape


# %% [2] Drop any remaining leakage columns
# These columns are known only after the loan outcome — using them
# would give the model perfect AUC but make it useless in production.
# At origination time, none of these values exist yet.

print("\n[2] Dropping leakage and post-event columns")

leakage_cols = [
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_amnt", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d",
    "out_prncp", "out_prncp_inv", "funded_amnt_inv", "loan_status"
]

cols_to_drop = [c for c in leakage_cols if c in df.columns]
df = df.drop(columns=cols_to_drop)
print(f"    Dropped {len(cols_to_drop)} leakage/post-event columns")
print(f"    Remaining columns: {df.shape[1]}")


# %% [3] Clean string columns that should be numeric
# revol_util is stored as a string with % sign in some versions.
# int_rate may also have % signs. Clean both defensively.
# emp_length needs special parsing to extract the numeric value.

print("\n[3] Cleaning string columns")

# Clean revol_util — remove % and convert to float
if df["revol_util"].dtype == object:
    df["revol_util"] = (
        df["revol_util"].astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace("nan", np.nan)
        .astype(float)
    )
    print("    revol_util: cleaned % string to float")
else:
    print("    revol_util: already numeric, no cleaning needed")

# Clean int_rate — remove % if present
if df["int_rate"].dtype == object:
    df["int_rate"] = (
        df["int_rate"].astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace("nan", np.nan)
        .astype(float)
    )
    print("    int_rate: cleaned % string to float")
else:
    print("    int_rate: already numeric, no cleaning needed")

# Clean emp_length — extract numeric years
# Values like "10+ years", "< 1 year", "5 years" -> numeric
emp_length_map = {
    "< 1 year": 0,
    "1 year":   1,
    "2 years":  2,
    "3 years":  3,
    "4 years":  4,
    "5 years":  5,
    "6 years":  6,
    "7 years":  7,
    "8 years":  8,
    "9 years":  9,
    "10+ years": 10
}
df["emp_length_num"] = df["emp_length"].map(emp_length_map)
print("    emp_length: mapped to numeric (emp_length_num)")
print(f"    emp_length null rate after mapping: {df['emp_length_num'].isnull().mean():.3f}")


# %% [4] Handle missing values
# Strategy:
#   Numeric columns  -> median imputation (robust to outliers)
#   Categorical cols -> mode or 'Unknown' category
# We do NOT drop rows — with 1.3M rows we can afford to impute.
# Interview point: median is preferred over mean for skewed financial data.

print("\n[4] Handling missing values")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Exclude target and engineered columns from imputation list
exclude_from_impute = ["default_flag", "issue_year"]
numeric_to_impute = [c for c in numeric_cols if c not in exclude_from_impute]

impute_report = []

# Numeric: median imputation
for col in numeric_to_impute:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        impute_report.append({
            "column": col,
            "null_count": null_count,
            "null_pct": round(null_count / len(df) * 100, 2),
            "strategy": "median",
            "fill_value": round(median_val, 4)
        })

# Categorical: fill with 'Unknown'
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        df[col] = df[col].fillna("Unknown")
        impute_report.append({
            "column": col,
            "null_count": null_count,
            "null_pct": round(null_count / len(df) * 100, 2),
            "strategy": "fill_Unknown",
            "fill_value": "Unknown"
        })

impute_df = pd.DataFrame(impute_report).sort_values("null_pct", ascending=False)
print(f"    Columns imputed: {len(impute_report)}")
print(impute_df.to_string(index=False))

# Verify no nulls remain in numeric columns
remaining_nulls = df[numeric_to_impute].isnull().sum().sum()
print(f"\n    Remaining nulls in numeric columns: {remaining_nulls}")


# %% [5] Cap outliers at 99th percentile
# Extreme outliers in financial data (e.g., annual_inc = $9M) distort
# model training. We cap at the 99th percentile — this preserves the
# direction of the relationship while reducing leverage of extreme points.
# Interview point: we cap rather than drop to preserve sample size.

print("\n[5] Capping outliers at 99th percentile")

cap_cols = ["annual_inc", "loan_amnt", "dti", "revol_bal", "open_acc",
            "total_acc", "inq_last_6mths", "delinq_2yrs"]

cap_report = []
for col in cap_cols:
    if col in df.columns:
        p99 = df[col].quantile(0.99)
        n_capped = (df[col] > p99).sum()
        df[col] = df[col].clip(upper=p99)
        cap_report.append({
            "column": col,
            "p99_value": round(p99, 2),
            "rows_capped": n_capped,
            "pct_capped": round(n_capped / len(df) * 100, 3)
        })

cap_df = pd.DataFrame(cap_report)
print(cap_df.to_string(index=False))


# %% [6] Final column check and type alignment
print("\n[6] Final column check")
print(f"    Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\n    Column dtypes:")
print(df.dtypes.value_counts())
print(f"\n    Remaining nulls (all columns): {df.isnull().sum().sum()}")
print("\n    Target distribution:")
print(df["default_flag"].value_counts(normalize=True).round(4))


# %% [7] Save outputs
print("\n[7] Saving outputs")

clean_path = Path("data/clean_data.parquet")
df.to_parquet(clean_path, index=False)
print(f"    Saved clean data: {clean_path}")

cleaning_report = pd.concat([
    impute_df.assign(step="imputation"),
    cap_df.assign(step="outlier_capping")
], ignore_index=True)
report_path = RESULTS_PATH / "cleaning_report.csv"
cleaning_report.to_csv(report_path, index=False)
print(f"    Saved cleaning report: {report_path}")


# %% [8] Final summary
print("\n" + "=" * 60)
print("STAGE 03 COMPLETE")
print("=" * 60)
print(f"  Rows before : {shape_before[0]:,}")
print(f"  Rows after  : {df.shape[0]:,}  (no rows dropped — imputation only)")
print(f"  Cols before : {shape_before[1]}")
print(f"  Cols after  : {df.shape[1]}")
print("  Clean data  : data/clean_data.parquet")
print("\n  Next step: Run scripts/04_feature_engineering.py")
