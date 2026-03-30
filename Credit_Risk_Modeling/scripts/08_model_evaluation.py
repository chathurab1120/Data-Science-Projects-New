"""
Script: 08_model_evaluation.py
Purpose: Full validation of the champion model — confusion matrix, lift chart,
         calibration curve, precision-recall curve, and Population Stability
         Index (PSI). This is the model validation report.
Inputs:  outputs/models/champion_model.pkl
         data/X_train.parquet, data/X_test.parquet
         data/y_train.parquet, data/y_test.parquet
Outputs: outputs/charts/10_confusion_matrix.png
         outputs/charts/11_lift_chart.png
         outputs/charts/12_calibration_curve.png
         outputs/charts/13_precision_recall_curve.png
         outputs/results/psi_report.csv
         outputs/results/lift_table.csv
         outputs/results/final_metrics.csv
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
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHARTS_PATH = Path(config["paths"]["outputs_charts"])
RESULTS_PATH = Path(config["paths"]["outputs_results"])
RANDOM_STATE = config["project"]["random_state"]
THRESHOLD = config["evaluation"]["threshold"]
TOP_DECILE = config["evaluation"]["top_decile"]
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
print("STAGE 08 — Model Evaluation & Validation")
print("=" * 60)


# %% [1] Load champion model and data
print("\n[1] Loading champion model and data")
champion = joblib.load("outputs/models/champion_model.pkl")
X_train = pd.read_parquet("data/X_train.parquet")
X_test = pd.read_parquet("data/X_test.parquet")
y_train = pd.read_parquet("data/y_train.parquet").squeeze()
y_test = pd.read_parquet("data/y_test.parquet").squeeze()

y_proba_test = champion.predict_proba(X_test)[:, 1]
y_proba_train = champion.predict_proba(X_train)[:, 1]
y_pred_class = (y_proba_test >= THRESHOLD).astype(int)

print(f"    Champion model loaded: {type(champion).__name__}")
print(f"    Test set size: {len(y_test):,}")
print(f"    Threshold: {THRESHOLD}")


# %% [2] Core metrics recap
print("\n[2] Core metrics")

test_auc = roc_auc_score(y_test, y_proba_test)
train_auc = roc_auc_score(y_train, y_proba_train)
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba_test)
ks_stat = (tpr - fpr).max()
gini = 2 * test_auc - 1

print(f"    Test AUC  : {test_auc:.4f}")
print(f"    Train AUC : {train_auc:.4f}")
print(f"    KS Stat   : {ks_stat:.4f}")
print(f"    Gini      : {gini:.4f}")


# %% [3] Confusion Matrix
# At threshold 0.35 we catch more defaults at the cost of more false alarms.
# In credit risk, missing a default (false negative) costs more than
# a false alarm (false positive) — so a lower threshold is justified.
# Interview point: always justify your threshold with business logic.

print("\n[3] Confusion matrix at threshold", THRESHOLD)
cm = confusion_matrix(y_test, y_pred_class)
tn, fp, fn, tp = cm.ravel()

print(f"    True Negatives  (correctly approved): {tn:,}")
print(f"    False Positives (incorrectly declined): {fp:,}")
print(f"    False Negatives (missed defaults):     {fn:,}")
print(f"    True Positives  (correctly flagged):   {tp:,}")
print(f"    Precision: {tp/(tp+fp):.4f}")
print(f"    Recall   : {tp/(tp+fn):.4f}")

fig, ax = plt.subplots(figsize=(7, 6))
cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(
    cm_pct, annot=True, fmt=".2%", cmap="Blues",
    xticklabels=["Predicted Good", "Predicted Bad"],
    yticklabels=["Actual Good", "Actual Bad"],
    ax=ax, linewidths=0.5, cbar_kws={"format": mticker.PercentFormatter(xmax=1)}
)
ax.set_title(f"Confusion Matrix (threshold = {THRESHOLD})\nChampion: {type(champion).__name__}",
             fontsize=12, fontweight="bold", pad=12)
ax.set_ylabel("Actual Label", fontsize=11)
ax.set_xlabel("Predicted Label", fontsize=11)
plt.tight_layout()
save_path = CHARTS_PATH / "10_confusion_matrix.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")


# %% [4] Lift Chart and Top Decile Lift
# Lift measures how much better the model is than random selection.
# Top decile lift > 2.5x means: if you target the top 10% of predicted
# defaulters, you capture 2.5x more actual defaults than random.
# This directly translates to collections efficiency.

print("\n[4] Lift chart and decile analysis")

lift_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_proba": y_proba_test
}).sort_values("y_proba", ascending=False).reset_index(drop=True)

lift_df["decile"] = pd.qcut(lift_df.index, q=10, labels=range(1, 11))
avg_default_rate = y_test.mean()

decile_summary = (
    lift_df.groupby("decile")["y_true"]
    .agg(["sum", "count", "mean"])
    .reset_index()
    .rename(columns={"sum": "defaults", "count": "total", "mean": "default_rate"})
)
decile_summary["lift"] = decile_summary["default_rate"] / avg_default_rate
decile_summary["cumulative_defaults"] = decile_summary["defaults"].cumsum()
decile_summary["cumulative_lift"] = (
    decile_summary["cumulative_defaults"].cumsum() /
    (decile_summary["total"].cumsum() * avg_default_rate)
)

top_decile_lift = decile_summary.iloc[0]["lift"]
print(f"    Average default rate: {avg_default_rate:.4f}")
print(f"    Top decile lift     : {top_decile_lift:.2f}x")
print("\n    Decile table:")
print(decile_summary[["decile", "total", "defaults", "default_rate", "lift"]].to_string(index=False))

if top_decile_lift >= 2.5:
    print(f"\n    PASS: Top decile lift {top_decile_lift:.2f}x >= 2.5x target")
else:
    print(f"\n    NOTE: Top decile lift {top_decile_lift:.2f}x (target >= 2.5x)")

decile_summary.to_csv(RESULTS_PATH / "lift_table.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(decile_summary["decile"].astype(str), decile_summary["lift"],
       color=sns.color_palette("RdYlGn_r", 10), edgecolor="white")
ax.axhline(y=1.0, color="#e74c3c", linestyle="--", linewidth=1.5,
           label="Random baseline (lift = 1.0)")
ax.axhline(y=2.5, color="#27ae60", linestyle="--", linewidth=1.5,
           label="Target lift = 2.5x")
for i, (_, row) in enumerate(decile_summary.iterrows()):
    ax.text(i, row["lift"] + 0.03, f"{row['lift']:.2f}x",
            ha="center", fontsize=9, fontweight="bold")
ax.set_title(f"Lift Chart by Decile — {type(champion).__name__}",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Decile (1 = Highest Predicted Risk)", fontsize=11)
ax.set_ylabel("Lift over Random", fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
save_path = CHARTS_PATH / "11_lift_chart.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")


# %% [5] Calibration Curve
# A well-calibrated model means: when it predicts 30% default probability,
# approximately 30% of those loans actually default.
# Poor calibration = model scores are misleading for business decisions.
# Interview point: tree models are often poorly calibrated — worth noting.

print("\n[5] Calibration curve")

fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_proba_test, n_bins=10, strategy="uniform"
)

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
ax.plot(mean_predicted_value, fraction_of_positives,
        "o-", color="#e74c3c", linewidth=2, markersize=8, label=f"{type(champion).__name__}")
ax.fill_between(mean_predicted_value, fraction_of_positives,
                mean_predicted_value, alpha=0.1, color="#e74c3c")
ax.set_title("Calibration Curve — Champion Model", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Mean Predicted Probability", fontsize=11)
ax.set_ylabel("Fraction of Positives (Actual Default Rate)", fontsize=11)
ax.legend(fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
save_path = CHARTS_PATH / "12_calibration_curve.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")


# %% [6] Precision-Recall Curve
# In imbalanced datasets, PR curve is more informative than ROC.
# The area under PR curve (Average Precision) reflects performance
# specifically on the minority class (defaulters).
# Interview point: a model can have high AUC but low AP — always check both.

print("\n[6] Precision-recall curve")

precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba_test)
avg_precision = average_precision_score(y_test, y_proba_test)
random_baseline = y_test.mean()

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(recall, precision, color="#3498db", linewidth=2,
        label=f"Champion (AP = {avg_precision:.4f})")
ax.axhline(y=random_baseline, color="#e74c3c", linestyle="--", linewidth=1.5,
           label=f"Random baseline (AP = {random_baseline:.4f})")
ax.set_title("Precision-Recall Curve — Champion Model",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.legend(fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
save_path = CHARTS_PATH / "13_precision_recall_curve.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")
print(f"    Average Precision: {avg_precision:.4f}")


# %% [7] Population Stability Index (PSI)
# PSI measures how much the score distribution has shifted between
# train (expected) and test (actual) populations.
# PSI < 0.10 = stable, no action needed
# PSI 0.10-0.20 = slight shift, monitor
# PSI > 0.20 = significant shift, investigate or retrain
# Interview point: PSI is the primary monitoring metric in production.

print("\n[7] Computing Population Stability Index (PSI)")


def compute_psi(expected, actual, bins=10):
    """Compute PSI between expected (train) and actual (test) score distributions."""
    breakpoints = np.linspace(0, 1, bins + 1)
    breakpoints[0] = -0.001
    breakpoints[-1] = 1.001

    expected_counts = pd.cut(expected, bins=breakpoints).value_counts().sort_index()
    actual_counts = pd.cut(actual, bins=breakpoints).value_counts().sort_index()

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid log(0)
    expected_pct = expected_pct.replace(0, 1e-6)
    actual_pct = actual_pct.replace(0, 1e-6)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi_total = psi_values.sum()

    psi_df = pd.DataFrame({
        "bin": expected_counts.index.astype(str),
        "expected_pct": expected_pct.values.round(4),
        "actual_pct": actual_pct.values.round(4),
        "psi_contrib": psi_values.values.round(6)
    })
    return psi_total, psi_df


psi_total, psi_df = compute_psi(y_proba_train, y_proba_test)

print(f"    PSI (train vs test): {psi_total:.4f}")
if psi_total < 0.10:
    print("    STABLE: PSI < 0.10 — score distribution is consistent")
elif psi_total < 0.20:
    print("    MONITOR: PSI 0.10-0.20 — slight distribution shift")
else:
    print("    WARNING: PSI > 0.20 — significant distribution shift detected")

psi_df.to_csv(RESULTS_PATH / "psi_report.csv", index=False)
print("    Saved: outputs/results/psi_report.csv")


# %% [8] Save final consolidated metrics
print("\n[8] Saving final metrics")

final_metrics = pd.DataFrame([{
    "model": type(champion).__name__,
    "test_auc": round(test_auc, 4),
    "train_auc": round(train_auc, 4),
    "ks_stat": round(ks_stat, 4),
    "gini": round(gini, 4),
    "top_decile_lift": round(float(top_decile_lift), 4),
    "avg_precision": round(avg_precision, 4),
    "psi": round(float(psi_total), 4),
    "threshold": THRESHOLD,
    "precision_at_threshold": round(tp / (tp + fp), 4),
    "recall_at_threshold": round(tp / (tp + fn), 4),
}])

final_metrics.to_csv(RESULTS_PATH / "final_metrics.csv", index=False)
print(final_metrics.T.to_string(header=False))
print("\n    Saved: outputs/results/final_metrics.csv")


# %% [9] Final summary
print("\n" + "=" * 60)
print("STAGE 08 COMPLETE — Model Evaluation")
print("=" * 60)
print(f"  AUC            : {test_auc:.4f}")
print(f"  KS Statistic   : {ks_stat:.4f}")
print(f"  Gini           : {gini:.4f}")
print(f"  Top Decile Lift: {top_decile_lift:.2f}x")
print(f"  PSI            : {psi_total:.4f}")
print(f"\n  Charts saved to : {CHARTS_PATH}")
print(f"  Results saved to: {RESULTS_PATH}")
print("\n  Next step: Run scripts/09_shap_explainability.py")
