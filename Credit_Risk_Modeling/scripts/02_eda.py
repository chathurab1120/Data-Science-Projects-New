"""
Script: 02_eda.py
Purpose: Exploratory Data Analysis — produce all 5 EDA charts and the
         Information Value (IV) table used for feature selection.
Inputs:  data/working_data.parquet
Outputs: outputs/charts/01_default_rate_by_grade.png
         outputs/charts/02_fico_distribution_by_outcome.png
         outputs/charts/03_default_rate_by_purpose.png
         outputs/charts/04_missing_value_heatmap.png
         outputs/charts/05_dti_vs_income_scatter.png
         outputs/results/information_value_table.csv
"""

# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# %% [0] Imports and configuration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import missingno as msno
import yaml
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHARTS_PATH = Path(config["paths"]["outputs_charts"])
RESULTS_PATH = Path(config["paths"]["outputs_results"])
CHARTS_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Plot styling — clean, professional look for portfolio
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11
})

print("=" * 60)
print("STAGE 02 — Exploratory Data Analysis")
print("=" * 60)


# %% [1] Load working data
print("\n[1] Loading working data")
df = pd.read_parquet("data/working_data.parquet")
print(f"    Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"    Default rate: {df['default_flag'].mean():.4f}")


# %% [2] Chart 1 — Default Rate by Loan Grade
# Grade is LendingClub's internal risk ranking (A = best, G = worst).
# This chart validates that the grading system has real predictive power.
# Interview point: if grade has IV > 0.3 it is a strong feature to keep.

print("\n[2] Chart 1 — Default Rate by Loan Grade")

grade_default = (
    df.groupby("grade")["default_flag"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "default_rate", "count": "loan_count"})
    .sort_values("grade")
)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(
    grade_default["grade"],
    grade_default["default_rate"] * 100,
    color=sns.color_palette("RdYlGn_r", len(grade_default)),
    edgecolor="white",
    linewidth=0.8
)

# Label each bar with the default rate value
for bar, rate in zip(bars, grade_default["default_rate"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{rate*100:.1f}%",
        ha="center", va="bottom", fontsize=10, fontweight="bold"
    )

ax.set_title("Default Rate by Loan Grade", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Loan Grade (A = Lowest Risk, G = Highest Risk)", fontsize=11)
ax.set_ylabel("Default Rate (%)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.set_ylim(0, grade_default["default_rate"].max() * 100 + 8)

plt.tight_layout()
save_path = CHARTS_PATH / "01_default_rate_by_grade.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")
print(grade_default[["grade", "default_rate", "loan_count"]].to_string(index=False))


# %% [3] Chart 2 — FICO Score Distribution by Outcome
# Overlapping KDE plot shows how FICO separates defaulters from non-defaulters.
# Expected: defaulters cluster at 650-700, non-defaulters peak at 720+.
# Interview point: FICO is one of the top IV features in credit risk models.

print("\n[3] Chart 2 — FICO Distribution by Outcome")

# Compute FICO midpoint if not already present
df["fico_midpoint"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

fig, ax = plt.subplots(figsize=(10, 6))

good = df[df["default_flag"] == 0]["fico_midpoint"].dropna()
bad  = df[df["default_flag"] == 1]["fico_midpoint"].dropna()

good.plot.kde(ax=ax, label="Fully Paid (Good)", color="#2ecc71", linewidth=2.5)
bad.plot.kde(ax=ax,  label="Defaulted (Bad)",   color="#e74c3c", linewidth=2.5)

ax.fill_between(
    ax.lines[0].get_xdata(), ax.lines[0].get_ydata(), alpha=0.15, color="#2ecc71"
)
ax.fill_between(
    ax.lines[1].get_xdata(), ax.lines[1].get_ydata(), alpha=0.15, color="#e74c3c"
)

ax.set_title("FICO Score Distribution by Loan Outcome", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("FICO Score (Midpoint)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.legend(fontsize=11)
ax.set_xlim(600, 850)

plt.tight_layout()
save_path = CHARTS_PATH / "02_fico_distribution_by_outcome.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")
print(f"    Good loans avg FICO: {good.mean():.1f}")
print(f"    Bad loans avg FICO:  {bad.mean():.1f}")


# %% [4] Chart 3 — Default Rate by Loan Purpose
# Some loan purposes have structurally higher default rates.
# This feature has strong business intuition and good IV.
# Interview point: small business loans default at higher rates —
# likely because business income is more volatile than salary income.

print("\n[4] Chart 3 — Default Rate by Loan Purpose")

purpose_default = (
    df.groupby("purpose")["default_flag"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "default_rate", "count": "loan_count"})
    .sort_values("default_rate", ascending=True)
)

fig, ax = plt.subplots(figsize=(10, 8))
colors = sns.color_palette("RdYlGn_r", len(purpose_default))
bars = ax.barh(
    purpose_default["purpose"],
    purpose_default["default_rate"] * 100,
    color=colors,
    edgecolor="white"
)

for bar, rate in zip(bars, purpose_default["default_rate"]):
    ax.text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{rate*100:.1f}%",
        va="center", fontsize=9, fontweight="bold"
    )

ax.set_title("Default Rate by Loan Purpose", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Default Rate (%)", fontsize=11)
ax.set_ylabel("")
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.set_xlim(0, purpose_default["default_rate"].max() * 100 + 5)

plt.tight_layout()
save_path = CHARTS_PATH / "03_default_rate_by_purpose.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")


# %% [5] Chart 4 — Missing Value Heatmap
# missingno.matrix shows which columns have systematic missingness.
# Systematic patterns (columns missing together) reveal data structure.
# We sample 5000 rows for speed — the pattern is visible on a sample.

print("\n[5] Chart 4 — Missing Value Heatmap")

sample = df.sample(5000, random_state=42)

# Keep only columns with at least some missingness for a readable chart
cols_with_missing = [c for c in sample.columns if sample[c].isnull().any()]
sample_missing = sample[cols_with_missing]

fig, ax = plt.subplots(figsize=(14, 6))
msno.matrix(sample_missing, ax=ax, sparkline=False, fontsize=8, color=(0.2, 0.5, 0.8))
ax.set_title("Missing Value Pattern (5,000 row sample)", fontsize=14, fontweight="bold", pad=15)

plt.tight_layout()
save_path = CHARTS_PATH / "04_missing_value_heatmap.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")
print(f"    Columns with missingness: {len(cols_with_missing)}")


# %% [6] Chart 5 — DTI vs Annual Income Scatter
# High DTI + low income is a strong default signal.
# Log scale on income axis because income is heavily right-skewed.
# We sample 20,000 points for readability.

print("\n[6] Chart 5 — DTI vs Annual Income Scatter")

scatter_sample = df[
    (df["dti"].notna()) &
    (df["annual_inc"].notna()) &
    (df["annual_inc"] > 0) &
    (df["annual_inc"] < 500000) &
    (df["dti"] < 60)
].sample(20000, random_state=42)

fig, ax = plt.subplots(figsize=(10, 6))

good_s = scatter_sample[scatter_sample["default_flag"] == 0]
bad_s  = scatter_sample[scatter_sample["default_flag"] == 1]

ax.scatter(good_s["annual_inc"], good_s["dti"], alpha=0.15, s=8,
           color="#2ecc71", label="Fully Paid")
ax.scatter(bad_s["annual_inc"],  bad_s["dti"],  alpha=0.25, s=8,
           color="#e74c3c", label="Defaulted")

ax.set_xscale("log")
ax.set_title("DTI vs Annual Income by Loan Outcome", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Annual Income (log scale, USD)", fontsize=11)
ax.set_ylabel("Debt-to-Income Ratio (DTI)", fontsize=11)
ax.legend(fontsize=11, markerscale=3)

plt.tight_layout()
save_path = CHARTS_PATH / "05_dti_vs_income_scatter.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")


# %% [7] Information Value (IV) table
# IV measures each feature's predictive power for the target.
# Decision rule:
#   IV < 0.02  -> Useless, drop
#   IV 0.02-0.1 -> Weak
#   IV 0.1-0.3  -> Medium, keep
#   IV > 0.3    -> Strong, prioritise
# We use a simple binning approach: 10 quantile bins for numeric,
# raw categories for categorical features.

print("\n[7] Computing Information Value table")

def compute_iv(series, target, bins=10):
    """Compute Information Value for a single feature vs binary target."""
    df_iv = pd.DataFrame({"feature": series, "target": target}).dropna()

    if df_iv["feature"].dtype == "object":
        df_iv["bucket"] = df_iv["feature"]
    else:
        try:
            df_iv["bucket"] = pd.qcut(df_iv["feature"], q=bins, duplicates="drop")
        except Exception:
            df_iv["bucket"] = pd.cut(df_iv["feature"], bins=bins)

    grouped = df_iv.groupby("bucket")["target"].agg(["sum", "count"])
    grouped.columns = ["bad", "total"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    total_bad  = grouped["bad"].sum()
    total_good = grouped["good"].sum()

    grouped["pct_bad"]  = grouped["bad"]  / total_bad
    grouped["pct_good"] = grouped["good"] / total_good

    # Avoid log(0) by replacing zeros
    grouped["pct_bad"]  = grouped["pct_bad"].replace(0, 1e-6)
    grouped["pct_good"] = grouped["pct_good"].replace(0, 1e-6)

    grouped["woe"] = np.log(grouped["pct_bad"] / grouped["pct_good"])
    grouped["iv"]  = (grouped["pct_bad"] - grouped["pct_good"]) * grouped["woe"]

    return grouped["iv"].sum()

# Candidate features to evaluate
iv_features = [
    "fico_midpoint", "int_rate", "dti", "grade", "sub_grade",
    "annual_inc", "loan_amnt", "revol_util", "open_acc",
    "delinq_2yrs", "inq_last_6mths", "pub_rec", "mort_acc",
    "home_ownership", "purpose", "emp_length", "verification_status"
]

iv_results = []
for feat in iv_features:
    if feat in df.columns:
        iv_val = compute_iv(df[feat], df["default_flag"])
        strength = (
            "Strong"  if iv_val > 0.3  else
            "Medium"  if iv_val > 0.1  else
            "Weak"    if iv_val > 0.02 else
            "Useless"
        )
        iv_results.append({"feature": feat, "iv": round(iv_val, 4), "strength": strength})

iv_df = pd.DataFrame(iv_results).sort_values("iv", ascending=False)

iv_path = RESULTS_PATH / "information_value_table.csv"
iv_df.to_csv(iv_path, index=False)

print(f"\n    Information Value Table:")
print(iv_df.to_string(index=False))
print(f"\n    Saved: {iv_path}")


# %% [8] Final summary
print("\n" + "=" * 60)
print("STAGE 02 COMPLETE")
print("=" * 60)
print(f"  Charts saved to  : {CHARTS_PATH}")
print(f"  Results saved to : {RESULTS_PATH}")
print(f"\n  Charts produced:")
print(f"    01_default_rate_by_grade.png")
print(f"    02_fico_distribution_by_outcome.png")
print(f"    03_default_rate_by_purpose.png")
print(f"    04_missing_value_heatmap.png")
print(f"    05_dti_vs_income_scatter.png")
print(f"\n  Next step: Run scripts/03_data_cleaning.py")
