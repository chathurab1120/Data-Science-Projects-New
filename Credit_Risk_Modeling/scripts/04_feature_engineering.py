"""
Script: 04_feature_engineering.py
Purpose: Engineer all domain-specific features for credit risk modelling.
         Create 6 new features, encode categorical variables, and produce
         the final modelling-ready feature matrix.
Inputs:  data/clean_data.parquet
Outputs: data/model_data.parquet
         outputs/results/feature_list.csv
         outputs/results/engineered_features_summary.csv
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
print("STAGE 04 — Feature Engineering")
print("=" * 60)


# %% [1] Load clean data
print("\n[1] Loading clean data")
df = pd.read_parquet("data/clean_data.parquet")
print(f"    Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")


# %% [2] Engineer Feature 1 — credit_history_months
# Longer credit history generally signals lower risk.
# A borrower with 20 years of credit history is more predictable
# than someone who opened their first card 6 months ago.
# Blueprint formula: (issue_d - earliest_cr_line) in months

print("\n[2] Engineering credit_history_months")
df["credit_history_months"] = (
    (df["issue_d"] - df["earliest_cr_line"]).dt.days / 30
).round(1)

# Cap negative values (data entry errors) at 0
df["credit_history_months"] = df["credit_history_months"].clip(lower=0)

print(f"    Mean  : {df['credit_history_months'].mean():.1f} months")
print(f"    Median: {df['credit_history_months'].median():.1f} months")
print(f"    Min   : {df['credit_history_months'].min():.1f} months")
print(f"    Max   : {df['credit_history_months'].max():.1f} months")
print(f"    Nulls : {df['credit_history_months'].isnull().sum()}")


# %% [3] Engineer Feature 2 — loan_to_income
# Affordability ratio — how large is the loan relative to annual income?
# High ratio = borrower is stretched. +1 in denominator avoids division by zero.
# Interview point: this mirrors the Loan-to-Value concept in mortgage risk.

print("\n[3] Engineering loan_to_income")
df["loan_to_income"] = (df["loan_amnt"] / (df["annual_inc"] + 1)).round(6)

# Cap extreme values — ratio > 1 means loan exceeds annual income
p99_lti = df["loan_to_income"].quantile(0.99)
df["loan_to_income"] = df["loan_to_income"].clip(upper=p99_lti)

print(f"    Mean  : {df['loan_to_income'].mean():.4f}")
print(f"    Median: {df['loan_to_income'].median():.4f}")
print(f"    P99   : {p99_lti:.4f}")
print(f"    Nulls : {df['loan_to_income'].isnull().sum()}")


# %% [4] Engineer Feature 3 — fico_midpoint
# Combines fico_range_low and fico_range_high into a single numeric.
# This is the standard approach — the midpoint is representative and
# avoids multicollinearity from including both endpoints as features.

print("\n[4] Engineering fico_midpoint")
df["fico_midpoint"] = ((df["fico_range_low"] + df["fico_range_high"]) / 2).round(1)

print(f"    Mean  : {df['fico_midpoint'].mean():.1f}")
print(f"    Median: {df['fico_midpoint'].median():.1f}")
print(f"    Min   : {df['fico_midpoint'].min():.1f}")
print(f"    Max   : {df['fico_midpoint'].max():.1f}")


# %% [5] Engineer Feature 4 — grade_ordinal
# Converts letter grade (A-G) to ordinal integer (1-7).
# This preserves the natural ordering of risk grades.
# Using ordinal encoding here is correct — grade is truly ordered.
# Interview point: one-hot encoding grade would destroy the ordering.

print("\n[5] Engineering grade_ordinal")
grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
df["grade_ord"] = df["grade"].map(grade_map)

unmapped = df["grade_ord"].isnull().sum()
print(f"    Grade mapping: {grade_map}")
print(f"    Unmapped values: {unmapped}")
print(df.groupby("grade")["grade_ord"].first().to_string())


# %% [6] Engineer Feature 5 — high_dti_flag
# Binary flag for borrowers in the DTI stress zone (>30).
# Works as an interaction feature — high DTI combined with low FICO
# is a stronger signal than either alone.
# Interview point: domain-derived binary flags often outperform
# the raw continuous variable in tree models.

print("\n[6] Engineering high_dti_flag")
df["high_dti_flag"] = (df["dti"] > 30).astype(int)

flag_rate = df["high_dti_flag"].mean()
default_by_dti = df.groupby("high_dti_flag")["default_flag"].mean()
print(f"    Borrowers with DTI > 30: {flag_rate*100:.1f}%")
print(f"    Default rate — DTI <= 30: {default_by_dti[0]:.4f}")
print(f"    Default rate — DTI >  30: {default_by_dti[1]:.4f}")


# %% [7] Engineer Feature 6 — revol_util_clean
# revol_util should already be numeric after cleaning stage.
# We create a clean copy with nulls filled and extreme values capped.
# High revolving utilisation (>80%) is a classic stress signal.

print("\n[7] Engineering revol_util_clean")
df["revol_util_clean"] = df["revol_util"].clip(upper=100).fillna(df["revol_util"].median())

high_util_default = df[df["revol_util_clean"] > 80]["default_flag"].mean()
low_util_default  = df[df["revol_util_clean"] <= 80]["default_flag"].mean()
print(f"    Default rate revol_util > 80%: {high_util_default:.4f}")
print(f"    Default rate revol_util <= 80%: {low_util_default:.4f}")
print(f"    Nulls: {df['revol_util_clean'].isnull().sum()}")


# %% [8] Encode categorical variables
# home_ownership and purpose: one-hot encoding (no natural order)
# verification_status: one-hot encoding
# grade and sub_grade: already handled via ordinal + will drop raw strings
# Interview point: we use drop_first=True to avoid multicollinearity
# (dummy variable trap) — relevant for logistic regression.

print("\n[8] Encoding categorical variables")

# One-hot encode
ohe_cols = ["home_ownership", "purpose", "verification_status"]
df_encoded = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=int)

new_cols = [c for c in df_encoded.columns if c not in df.columns]
print(f"    One-hot encoded columns: {ohe_cols}")
print(f"    New dummy columns created: {len(new_cols)}")
print(f"    Shape after encoding: {df_encoded.shape}")


# %% [9] Define final feature matrix
# Select only the columns that will be used for modelling.
# Drop raw string columns that have been encoded or superseded.
# Drop date columns — their information is captured in engineered features.
# Keep issue_year for chronological train/test split downstream.

print("\n[9] Defining final feature matrix")

# Columns to drop from final feature set
drop_from_features = [
    "grade",              # replaced by grade_ord
    "sub_grade",          # too granular, grade_ord captures the signal
    "emp_length",         # replaced by emp_length_num
    "fico_range_low",     # replaced by fico_midpoint
    "fico_range_high",    # replaced by fico_midpoint
    "revol_util",         # replaced by revol_util_clean
    "issue_d",            # date — not a model feature
    "earliest_cr_line",   # date — captured in credit_history_months
]

df_model = df_encoded.drop(
    columns=[c for c in drop_from_features if c in df_encoded.columns]
)

# Separate feature columns from target and metadata
meta_cols   = ["default_flag", "issue_year"]
feature_cols = [c for c in df_model.columns if c not in meta_cols]

print(f"    Total columns in model data : {df_model.shape[1]}")
print(f"    Feature columns             : {len(feature_cols)}")
print(f"    Meta columns                : {meta_cols}")
print("\n    Final feature list:")
for i, col in enumerate(feature_cols, 1):
    print(f"      {i:2d}. {col}")


# %% [10] Save outputs
print("\n[10] Saving outputs")

model_path = Path("data/model_data.parquet")
df_model.to_parquet(model_path, index=False)
print(f"    Saved model data: {model_path}")

feature_df = pd.DataFrame({
    "feature": feature_cols,
    "dtype": [str(df_model[c].dtype) for c in feature_cols]
})
feature_path = RESULTS_PATH / "feature_list.csv"
feature_df.to_csv(feature_path, index=False)
print(f"    Saved feature list: {feature_path}")

eng_summary = pd.DataFrame([
    {"feature": "credit_history_months", "type": "numeric", "rationale": "Credit maturity signal"},
    {"feature": "loan_to_income",        "type": "numeric", "rationale": "Affordability ratio"},
    {"feature": "fico_midpoint",         "type": "numeric", "rationale": "Combined FICO score"},
    {"feature": "grade_ord",             "type": "ordinal", "rationale": "Ordinal risk grade"},
    {"feature": "high_dti_flag",         "type": "binary",  "rationale": "DTI stress zone flag"},
    {"feature": "revol_util_clean",      "type": "numeric", "rationale": "Cleaned revolving utilisation"},
])
eng_path = RESULTS_PATH / "engineered_features_summary.csv"
eng_summary.to_csv(eng_path, index=False)
print(f"    Saved engineered features summary: {eng_path}")


# %% [11] Final summary
print("\n" + "=" * 60)
print("STAGE 04 COMPLETE")
print("=" * 60)
print(f"  Final model data shape : {df_model.shape[0]:,} rows x {df_model.shape[1]} columns")
print(f"  Feature columns        : {len(feature_cols)}")
print("  Target                 : default_flag (0=Good, 1=Bad)")
print(f"  Model data saved to    : {model_path}")
print("\n  Next step: Run scripts/05_train_test_split.py")
