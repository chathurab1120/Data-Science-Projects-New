"""
Script: 09_shap_explainability.py
Purpose: Generate all 4 SHAP explainability outputs for the champion model.
         Global bar plot, beeswarm plot, FICO dependence plot, and
         waterfall plots for individual loan predictions.
Inputs:  outputs/models/champion_model.pkl
         data/X_test.parquet, data/y_test.parquet
Outputs: outputs/charts/14_shap_global_bar.png
         outputs/charts/15_shap_beeswarm.png
         outputs/charts/16_shap_fico_dependence.png
         outputs/charts/17_shap_waterfall_default.png
         outputs/charts/18_shap_waterfall_good.png
"""

# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# %% [0] Imports and configuration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import yaml
import joblib
import warnings
from pathlib import Path
import shap

warnings.filterwarnings("ignore")

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHARTS_PATH  = Path(config["paths"]["outputs_charts"])
RANDOM_STATE = config["project"]["random_state"]
CHARTS_PATH.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "font.family":      "sans-serif",
    "font.size":        11
})

print("=" * 60)
print("STAGE 09 — SHAP Explainability")
print("=" * 60)


# %% [1] Load champion model and test data
print("\n[1] Loading champion model and test data")
champion = joblib.load("outputs/models/champion_model.pkl")
X_test   = pd.read_parquet("data/X_test.parquet")
y_test   = pd.read_parquet("data/y_test.parquet").squeeze()

print(f"    Model: {type(champion).__name__}")
print(f"    X_test shape: {X_test.shape}")

# Sample for SHAP — computing exact SHAP on 518K rows takes too long.
# 5000 rows is standard for SHAP summaries — representative and fast.
# Interview point: SHAP is O(n*features) — sampling is necessary at scale.
np.random.seed(RANDOM_STATE)
sample_idx  = np.random.choice(len(X_test), size=5000, replace=False)
X_shap      = X_test.iloc[sample_idx].reset_index(drop=True)
y_shap      = y_test.iloc[sample_idx].reset_index(drop=True)

print(f"    SHAP sample size: {len(X_shap):,}")
print(f"    Default rate in sample: {y_shap.mean():.4f}")


# %% [2] Compute SHAP values
# TreeExplainer is the fast exact explainer for tree-based models.
# It computes exact Shapley values using the tree structure directly.
# Interview point: SHAP values satisfy three key properties —
# local accuracy, missingness, and consistency — making them the
# gold standard for model explanation.

print("\n[2] Computing SHAP values (TreeExplainer)")
print("    This may take 1-2 minutes...")

explainer   = shap.TreeExplainer(champion)
shap_values = explainer.shap_values(X_shap)
expected_value = explainer.expected_value

print(f"    SHAP values shape: {np.array(shap_values).shape}")
print(f"    Expected value (base rate): {expected_value:.4f}")
print("    SHAP computation complete")


# %% [3] Chart 1 — Global SHAP Bar Plot
# Top 15 features ranked by mean absolute SHAP value.
# This is the correct way to compute feature importance for tree models —
# unlike gain-based importance, SHAP importance accounts for interactions.
# Interview point: int_rate and grade_ord should dominate — consistent with IV table.

print("\n[3] Global SHAP bar plot (top 15 features)")

shap_df = pd.DataFrame(shap_values, columns=X_shap.columns)
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
top15 = mean_abs_shap.head(15)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 15))
bars = ax.barh(top15.index[::-1], top15.values[::-1],
               color=colors[::-1], edgecolor="white")

for bar, val in zip(bars, top15.values[::-1]):
    ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)

ax.set_title("SHAP Feature Importance — Top 15 Features\n(Mean Absolute SHAP Value)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
ax.set_ylabel("")

plt.tight_layout()
save_path = CHARTS_PATH / "14_shap_global_bar.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")
print(f"\n    Top 10 features by SHAP importance:")
for i, (feat, val) in enumerate(top15.head(10).items(), 1):
    print(f"      {i:2d}. {feat:<35} {val:.4f}")


# %% [4] Chart 2 — SHAP Beeswarm Plot
# One dot per observation per feature — shows direction AND magnitude.
# Red = high feature value, Blue = low feature value.
# Dots on the right = push prediction toward default.
# This is the most information-dense SHAP chart — use it in interviews.

print("\n[4] SHAP beeswarm plot")

fig, ax = plt.subplots(figsize=(11, 9))
shap.summary_plot(
    shap_values,
    X_shap,
    max_display=15,
    show=False,
    plot_size=None
)
plt.title("SHAP Beeswarm Plot — Top 15 Features\n(Red = High Value, Blue = Low Value)",
          fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
save_path = CHARTS_PATH / "15_shap_beeswarm.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")


# %% [5] Chart 3 — SHAP Dependence Plot (FICO)
# Shows SHAP value vs fico_midpoint, coloured by int_rate.
# Expected finding: FICO below 680 causes sharp increase in default SHAP.
# The colour interaction reveals that low FICO + high int_rate = worst case.
# Interview point: this non-linear relationship is why tree models
# outperform logistic regression on this feature.

print("\n[5] SHAP dependence plot — fico_midpoint")

fig, ax = plt.subplots(figsize=(10, 7))

if "int_rate" in X_shap.columns:
    color_feature = X_shap["int_rate"].values
    color_label   = "int_rate"
else:
    color_feature = X_shap["grade_ord"].values
    color_label   = "grade_ord"

fico_idx   = list(X_shap.columns).index("fico_midpoint")
fico_shap  = shap_values[:, fico_idx]
fico_vals  = X_shap["fico_midpoint"].values

sc = ax.scatter(fico_vals, fico_shap, c=color_feature,
                cmap="RdYlGn_r", alpha=0.4, s=12, rasterized=True)
plt.colorbar(sc, ax=ax, label=color_label)
ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.axvline(x=680, color="#e74c3c", linewidth=1.5, linestyle="--",
           alpha=0.7, label="FICO = 680 (stress zone boundary)")
ax.set_title("SHAP Dependence Plot: FICO Midpoint\n(coloured by Interest Rate)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("FICO Score (Midpoint)", fontsize=11)
ax.set_ylabel("SHAP Value (impact on default probability)", fontsize=11)
ax.legend(fontsize=10)

plt.tight_layout()
save_path = CHARTS_PATH / "16_shap_fico_dependence.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")


# %% [6] Chart 4a — Waterfall Plot: Single Defaulted Loan
# Pick one actual defaulted loan and explain its prediction feature by feature.
# The waterfall shows which features pushed the score up (toward default)
# and which pushed it down, starting from the base rate.
# Interview point: this is how you explain a rejection decision to a regulator.

print("\n[6] Waterfall plot — single defaulted loan")

default_indices = np.where(y_shap.values == 1)[0]
default_idx     = default_indices[0]

default_shap_vals = shap_values[default_idx]
default_features  = X_shap.iloc[default_idx]
default_pred      = champion.predict_proba(X_shap.iloc[[default_idx]])[0, 1]

print(f"    Selected loan index: {default_idx}")
print(f"    Predicted default probability: {default_pred:.4f}")
print(f"    Actual outcome: Default (1)")

# Build waterfall manually for clean rendering
top_n = 12
abs_shap    = np.abs(default_shap_vals)
top_indices = np.argsort(abs_shap)[::-1][:top_n]
top_features = [X_shap.columns[i] for i in top_indices]
top_shap     = [default_shap_vals[i] for i in top_indices]
top_vals     = [default_features.iloc[i] for i in top_indices]

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#e74c3c" if s > 0 else "#2ecc71" for s in top_shap]
y_pos  = range(len(top_features))
ax.barh(y_pos, top_shap, color=colors, edgecolor="white", height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{f}\n= {v:.2f}" for f, v in zip(top_features, top_vals)], fontsize=9)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_title(f"SHAP Waterfall — Defaulted Loan\nPredicted PD = {default_pred:.3f} (Actual: Default)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("SHAP Value (Red = increases default risk)", fontsize=11)
ax.invert_yaxis()

plt.tight_layout()
save_path = CHARTS_PATH / "17_shap_waterfall_default.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")


# %% [7] Chart 4b — Waterfall Plot: Single Good Loan
# Same explanation for a loan that was fully paid.
# Contrasting the two waterfall plots is extremely powerful —
# it shows exactly what separates a good borrower from a bad one
# according to the model. Use both side by side in your presentation.

print("\n[7] Waterfall plot — single good loan")

good_indices = np.where(y_shap.values == 0)[0]
good_idx     = good_indices[0]

good_shap_vals = shap_values[good_idx]
good_features  = X_shap.iloc[good_idx]
good_pred      = champion.predict_proba(X_shap.iloc[[good_idx]])[0, 1]

print(f"    Selected loan index: {good_idx}")
print(f"    Predicted default probability: {good_pred:.4f}")
print(f"    Actual outcome: Fully Paid (0)")

abs_shap_good    = np.abs(good_shap_vals)
top_indices_good = np.argsort(abs_shap_good)[::-1][:top_n]
top_features_good = [X_shap.columns[i] for i in top_indices_good]
top_shap_good     = [good_shap_vals[i] for i in top_indices_good]
top_vals_good     = [good_features.iloc[i] for i in top_indices_good]

fig, ax = plt.subplots(figsize=(10, 8))
colors_good = ["#e74c3c" if s > 0 else "#2ecc71" for s in top_shap_good]
ax.barh(y_pos, top_shap_good, color=colors_good, edgecolor="white", height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{f}\n= {v:.2f}" for f, v in zip(top_features_good, top_vals_good)], fontsize=9)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_title(f"SHAP Waterfall — Good Loan (Fully Paid)\nPredicted PD = {good_pred:.3f} (Actual: Good)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("SHAP Value (Green = decreases default risk)", fontsize=11)
ax.invert_yaxis()

plt.tight_layout()
save_path = CHARTS_PATH / "18_shap_waterfall_good.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")


# %% [8] Final summary
print("\n" + "=" * 60)
print("STAGE 09 COMPLETE — SHAP Explainability")
print("=" * 60)
print(f"  Charts saved to: {CHARTS_PATH}")
print(f"\n  SHAP outputs produced:")
print(f"    14_shap_global_bar.png       — Top 15 features by importance")
print(f"    15_shap_beeswarm.png         — Direction + magnitude for all samples")
print(f"    16_shap_fico_dependence.png  — Non-linear FICO effect")
print(f"    17_shap_waterfall_default.png — Single defaulted loan explained")
print(f"    18_shap_waterfall_good.png   — Single good loan explained")
print(f"\n  Next step: Run scripts/10_scorecard.py")
