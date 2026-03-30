"""
Script: 10_scorecard.py
Purpose: Convert the logistic regression model into a points-based credit
         scorecard using standard scaling (score 600 = 50:1 odds, PDO = 20).
         This is the bonus stage — demonstrates knowledge of traditional
         credit risk methodology still used widely in banking.
Inputs:  outputs/models/logistic_regression.pkl
         data/X_train.parquet, data/X_test.parquet
         data/y_test.parquet
Outputs: outputs/results/scorecard_table.csv
         outputs/charts/19_score_distribution.png
         outputs/charts/20_scorecard_default_rate_by_band.png
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
import yaml
import joblib
import warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHARTS_PATH  = Path(config["paths"]["outputs_charts"])
RESULTS_PATH = Path(config["paths"]["outputs_results"])
RANDOM_STATE = config["project"]["random_state"]
for p in [CHARTS_PATH, RESULTS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "sans-serif",
    "font.size":        11
})

print("=" * 60)
print("STAGE 10 — Credit Scorecard (Bonus)")
print("=" * 60)


# %% [1] Load logistic regression model and data
# The scorecard is built from logistic regression — not XGBoost.
# LR produces log-odds scores that map directly to points.
# XGBoost cannot be converted to a scorecard in the traditional sense.
# Interview point: Basel II/III requires interpretable scorecards
# for regulatory capital calculations in retail lending.

print("\n[1] Loading logistic regression model and data")
lr_pipeline = joblib.load("outputs/models/logistic_regression.pkl")
X_train     = pd.read_parquet("data/X_train.parquet")
X_test      = pd.read_parquet("data/X_test.parquet")
y_test      = pd.read_parquet("data/y_test.parquet").squeeze()

lr_model  = lr_pipeline.named_steps["model"]
scaler    = lr_pipeline.named_steps["scaler"]

print(f"    LR model loaded")
print(f"    Features: {X_train.shape[1]}")
print(f"    Intercept: {lr_model.intercept_[0]:.4f}")


# %% [2] Scorecard scaling parameters
# Standard scorecard scaling formula:
#   Score = Offset + Factor * log(odds)
# Where:
#   Offset = score at base odds (600 at 50:1 odds)
#   Factor = PDO / log(2)  — PDO = Points to Double the Odds (20)
#   log(odds) = log(P_good / P_bad)
# Interview point: PDO=20 means a borrower needs 20 more points
# to be half as likely to default. This is industry standard.

print("\n[2] Scorecard scaling parameters")

PDO    = 20       # Points to double the odds
ODDS   = 50       # Base odds (good:bad) at base score
SCORE0 = 600      # Base score

FACTOR = PDO / np.log(2)
OFFSET = SCORE0 - FACTOR * np.log(ODDS)

print(f"    PDO (Points to Double Odds) : {PDO}")
print(f"    Base odds (good:bad)        : {ODDS}:1")
print(f"    Base score                  : {SCORE0}")
print(f"    Factor                      : {FACTOR:.4f}")
print(f"    Offset                      : {OFFSET:.4f}")


# %% [3] Compute log-odds scores on test set
# The logistic regression outputs log-odds directly via decision_function.
# We then apply the scaling formula to convert to credit score points.
# Higher score = lower risk (opposite of raw probability).

print("\n[3] Computing credit scores")

# Get scaled features
X_test_scaled  = scaler.transform(X_test)
X_train_scaled = scaler.transform(X_train)

# Log-odds from logistic regression
log_odds_test  = lr_model.decision_function(X_test_scaled)
log_odds_train = lr_model.decision_function(X_train_scaled)

# Convert log-odds to credit score
# Note: decision_function gives log(P_bad/P_good) for binary classification
# We negate it so higher score = lower risk
scores_test  = OFFSET + FACTOR * (-log_odds_test)
scores_train = OFFSET + FACTOR * (-log_odds_train)

print(f"    Test score distribution:")
print(f"      Mean   : {scores_test.mean():.1f}")
print(f"      Std    : {scores_test.std():.1f}")
print(f"      Min    : {scores_test.min():.1f}")
print(f"      Max    : {scores_test.max():.1f}")
print(f"      Median : {np.median(scores_test):.1f}")

# Validate — higher score should correlate with lower default rate
score_series = pd.Series(scores_test)
default_series = y_test.reset_index(drop=True)
correlation = score_series.corr(default_series)
print(f"\n    Score-default correlation: {correlation:.4f}")
print(f"    (Should be negative — higher score = lower default risk)")


# %% [4] Score band analysis
# Divide scores into bands and compute default rate per band.
# This is how scorecards are operationalised in lending decisions.
# Each band maps to a credit policy decision (approve/review/decline).

print("\n[4] Score band analysis")

score_df = pd.DataFrame({
    "score":        scores_test,
    "default_flag": y_test.values
})

# Define score bands (standard 20-point bands)
bins   = [300, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 850]
labels = ["<520", "520-540", "540-560", "560-580", "580-600",
          "600-620", "620-640", "640-660", "660-680", "680-700", ">700"]

score_df["score_band"] = pd.cut(score_df["score"], bins=bins, labels=labels)

band_summary = (
    score_df.groupby("score_band", observed=True)["default_flag"]
    .agg(["sum", "count", "mean"])
    .reset_index()
    .rename(columns={"sum": "defaults", "count": "total", "mean": "default_rate"})
)
band_summary["approval_rate"] = (band_summary["total"] / len(score_df) * 100).round(2)
band_summary["default_rate"]  = (band_summary["default_rate"] * 100).round(2)

print(f"\n    Score Band Table:")
print(band_summary.to_string(index=False))

band_summary.to_csv(RESULTS_PATH / "scorecard_table.csv", index=False)
print(f"\n    Saved: outputs/results/scorecard_table.csv")


# %% [5] Feature contribution table
# Shows how much each feature contributes to the scorecard in points.
# This is the actual scorecard — each feature adds or subtracts points.
# Interview point: a loan officer can hand-score an applicant using this table.

print("\n[5] Feature contribution to scorecard")

feature_names = X_train.columns.tolist()
coefficients  = lr_model.coef_[0]
scale_means   = scaler.mean_
scale_stds    = scaler.scale_

contrib_df = pd.DataFrame({
    "feature":        feature_names,
    "coefficient":    coefficients,
    "mean":           scale_means,
    "std":            scale_stds,
}).copy()

# Points contribution = -Factor * coef * (value - mean) / std
# At mean value, contribution = 0 (captured in offset)
# Range of contribution = -Factor * coef * 2*std / std = -Factor * coef * 2
contrib_df["points_per_unit"] = -FACTOR * contrib_df["coefficient"] / contrib_df["std"]
contrib_df["range_points"]    = (contrib_df["points_per_unit"] * contrib_df["std"] * 2).abs()
contrib_df = contrib_df.sort_values("range_points", ascending=False)

print(f"\n    Top 15 features by scorecard point range:")
print(contrib_df[["feature", "points_per_unit", "range_points"]].head(15).to_string(index=False))


# %% [6] Chart — Score Distribution by Outcome
print("\n[6] Score distribution chart")

fig, ax = plt.subplots(figsize=(11, 6))

good_scores = scores_test[y_test.values == 0]
bad_scores  = scores_test[y_test.values == 1]

ax.hist(good_scores, bins=60, alpha=0.55, color="#2ecc71",
        label=f"Fully Paid (n={len(good_scores):,})", density=True)
ax.hist(bad_scores,  bins=60, alpha=0.55, color="#e74c3c",
        label=f"Defaulted (n={len(bad_scores):,})",  density=True)

ax.axvline(x=600, color="#2c3e50", linestyle="--", linewidth=1.5,
           label="Score = 600 (base threshold)")
ax.set_title("Credit Score Distribution by Loan Outcome\n(Higher Score = Lower Risk)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Credit Score", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.legend(fontsize=11)

plt.tight_layout()
save_path = CHARTS_PATH / "19_score_distribution.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")


# %% [7] Chart — Default Rate by Score Band
print("\n[7] Default rate by score band chart")

fig, ax = plt.subplots(figsize=(12, 6))
colors = sns.color_palette("RdYlGn", len(band_summary))

bars = ax.bar(band_summary["score_band"].astype(str),
              band_summary["default_rate"],
              color=colors, edgecolor="white")

for bar, rate in zip(bars, band_summary["default_rate"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f"{rate:.1f}%", ha="center", fontsize=9, fontweight="bold")

ax.set_title("Default Rate by Credit Score Band",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Credit Score Band", fontsize=11)
ax.set_ylabel("Default Rate (%)", fontsize=11)
ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
save_path = CHARTS_PATH / "20_scorecard_default_rate_by_band.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")


# %% [8] Final summary
print("\n" + "=" * 60)
print("STAGE 10 COMPLETE — Credit Scorecard")
print("=" * 60)
print(f"  Scorecard scaling: PDO={PDO}, Base Score={SCORE0}, Base Odds={ODDS}:1")
print(f"  Score range in test set: {scores_test.min():.0f} - {scores_test.max():.0f}")
print(f"  Score-default correlation: {correlation:.4f}")
print(f"\n  Outputs:")
print(f"    outputs/results/scorecard_table.csv")
print(f"    outputs/charts/19_score_distribution.png")
print(f"    outputs/charts/20_scorecard_default_rate_by_band.png")
print(f"\n  ALL MODELLING STAGES COMPLETE")
print(f"  Next step: Build the Streamlit dashboard (scripts -> app/)")
