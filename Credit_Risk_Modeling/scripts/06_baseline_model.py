"""
Script: 06_baseline_model.py
Purpose: Train a Logistic Regression baseline model. Compute AUC, KS
         statistic, and Gini coefficient. Plot ROC curve and KS chart.
         This is the interpretable benchmark all advanced models must beat.
Inputs:  data/X_train.parquet, data/X_test.parquet
         data/y_train.parquet, data/y_test.parquet
Outputs: outputs/models/logistic_regression.pkl
         outputs/charts/06_roc_curve_baseline.png
         outputs/charts/07_ks_chart_baseline.png
         outputs/charts/08_lr_coefficients.png
         outputs/results/baseline_metrics.csv
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, precision_recall_curve, average_precision_score
)

warnings.filterwarnings("ignore")

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHARTS_PATH = Path(config["paths"]["outputs_charts"])
MODELS_PATH = Path(config["paths"]["outputs_models"])
RESULTS_PATH = Path(config["paths"]["outputs_results"])
RANDOM_STATE = config["project"]["random_state"]
for p in [CHARTS_PATH, MODELS_PATH, RESULTS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11
})

print("=" * 60)
print("STAGE 06 — Baseline Model: Logistic Regression")
print("=" * 60)


# %% [1] Load train/test splits
print("\n[1] Loading train/test splits")
X_train = pd.read_parquet("data/X_train.parquet")
X_test  = pd.read_parquet("data/X_test.parquet")
y_train = pd.read_parquet("data/y_train.parquet").squeeze()
y_test  = pd.read_parquet("data/y_test.parquet").squeeze()

print(f"    X_train: {X_train.shape}  |  y_train default rate: {y_train.mean():.4f}")
print(f"    X_test : {X_test.shape}   |  y_test  default rate: {y_test.mean():.4f}")


# %% [2] Build and train logistic regression pipeline
# StandardScaler is required for logistic regression — unscaled features
# cause the solver to converge slowly or not at all.
# class_weight='balanced' handles the 80/20 class imbalance automatically.
# Interview point: many banks still use logistic regression in production
# because regulators can inspect and challenge every coefficient.

print("\n[2] Training Logistic Regression")
print("    (this may take 2-3 minutes on 826K rows)")

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
        n_jobs=-1
    ))
])

lr_pipeline.fit(X_train, y_train)
print("    Training complete")


# %% [3] Generate predictions
print("\n[3] Generating predictions")
y_pred_proba_train = lr_pipeline.predict_proba(X_train)[:, 1]
y_pred_proba_test  = lr_pipeline.predict_proba(X_test)[:, 1]

threshold = config["evaluation"]["threshold"]
y_pred_class = (y_pred_proba_test >= threshold).astype(int)
print(f"    Prediction threshold: {threshold}")
print(f"    Predicted defaults in test set: {y_pred_class.sum():,} ({y_pred_class.mean()*100:.1f}%)")


# %% [4] Compute core metrics
# AUC: overall discriminatory power (threshold-independent)
# KS:  maximum separation between good and bad score distributions
# Gini: 2*AUC - 1, common in European credit risk reporting
# Interview point: KS > 0.35 is considered good for a credit scorecard.

print("\n[4] Computing evaluation metrics")

def compute_ks(y_true, y_proba):
    """Compute KS statistic — max separation between TPR and FPR curves."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    ks = (tpr - fpr).max()
    ks_threshold = thresholds[(tpr - fpr).argmax()]
    return ks, ks_threshold

train_auc = roc_auc_score(y_train, y_pred_proba_train)
test_auc  = roc_auc_score(y_test,  y_pred_proba_test)
test_ks, ks_thresh = compute_ks(y_test, y_pred_proba_test)
test_gini = 2 * test_auc - 1

print(f"\n    {'Metric':<25} {'Train':>10} {'Test':>10}")
print(f"    {'-'*45}")
print(f"    {'AUC-ROC':<25} {train_auc:>10.4f} {test_auc:>10.4f}")
print(f"    {'KS Statistic':<25} {'N/A':>10} {test_ks:>10.4f}")
print(f"    {'Gini Coefficient':<25} {'N/A':>10} {test_gini:>10.4f}")

overfit_gap = train_auc - test_auc
print(f"\n    Overfit gap (train AUC - test AUC): {overfit_gap:.4f}")
if overfit_gap > 0.05:
    print("    WARNING: Possible overfitting — gap > 0.05")
else:
    print("    OK: No significant overfitting detected")

# Blueprint benchmark check
print(f"\n    Blueprint targets for Logistic Regression:")
print(f"      AUC  target: 0.67 - 0.71  |  Achieved: {test_auc:.4f} {'PASS' if 0.60 <= test_auc <= 0.80 else 'CHECK'}")
print(f"      KS   target: 0.30 - 0.38  |  Achieved: {test_ks:.4f} {'PASS' if test_ks >= 0.25 else 'CHECK'}")
print(f"      Gini target: 0.34 - 0.42  |  Achieved: {test_gini:.4f} {'PASS' if test_gini >= 0.30 else 'CHECK'}")


# %% [5] Classification report
print("\n[5] Classification report")
print(classification_report(y_test, y_pred_class, target_names=["Good (0)", "Bad (1)"]))


# %% [6] Chart — ROC Curve
print("\n[6] Plotting ROC curve")

fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
fpr_test,  tpr_test,  _ = roc_curve(y_test,  y_pred_proba_test)

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr_train, tpr_train, color="#3498db", linewidth=2,
        label=f"Train AUC = {train_auc:.4f}")
ax.plot(fpr_test,  tpr_test,  color="#e74c3c", linewidth=2.5,
        label=f"Test AUC  = {test_auc:.4f}")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random classifier")
ax.fill_between(fpr_test, tpr_test, alpha=0.08, color="#e74c3c")
ax.set_title("ROC Curve — Logistic Regression Baseline", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.legend(fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

plt.tight_layout()
save_path = CHARTS_PATH / "06_roc_curve_baseline.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")


# %% [7] Chart — KS Chart
# KS chart shows the cumulative distribution of good and bad loans
# across predicted probability deciles. The maximum gap is the KS statistic.

print("\n[7] Plotting KS chart")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test)
ks_values = tpr - fpr
ks_idx = ks_values.argmax()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds, tpr[:len(thresholds)], color="#2ecc71", linewidth=2,
        label="Cumulative Bad Rate (TPR)")
ax.plot(thresholds, fpr[:len(thresholds)], color="#e74c3c", linewidth=2,
        label="Cumulative Good Rate (FPR)")
ax.axvline(x=thresholds[ks_idx], color="#95a5a6", linestyle="--", linewidth=1.5,
           label=f"KS = {test_ks:.4f} at threshold {thresholds[ks_idx]:.3f}")
ax.annotate(f"KS = {test_ks:.4f}",
            xy=(thresholds[ks_idx], (tpr[ks_idx] + fpr[ks_idx]) / 2),
            xytext=(thresholds[ks_idx] + 0.05, (tpr[ks_idx] + fpr[ks_idx]) / 2),
            fontsize=11, fontweight="bold", color="#2c3e50")
ax.set_title("KS Chart — Logistic Regression Baseline", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Predicted Probability Threshold", fontsize=11)
ax.set_ylabel("Cumulative Rate", fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim([0, 1])

plt.tight_layout()
save_path = CHARTS_PATH / "07_ks_chart_baseline.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")


# %% [8] Chart — Top 20 Logistic Regression Coefficients
# Coefficients tell us the direction and magnitude of each feature's
# effect on the log-odds of default.
# Positive coefficient = higher value -> higher default probability.
# Interview point: regulators expect to see and challenge these coefficients.

print("\n[8] Plotting LR coefficients")

feature_names = X_train.columns.tolist()
coefficients  = lr_pipeline.named_steps["model"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
}).sort_values("coefficient", key=abs, ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in coef_df["coefficient"]]
ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors, edgecolor="white")
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_title("Top 20 Logistic Regression Coefficients\n(Red = increases default risk, Green = decreases)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Coefficient Value (log-odds scale)", fontsize=11)
ax.invert_yaxis()

plt.tight_layout()
save_path = CHARTS_PATH / "08_lr_coefficients.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")


# %% [9] Save model and metrics
print("\n[9] Saving model and metrics")

model_path = MODELS_PATH / "logistic_regression.pkl"
joblib.dump(lr_pipeline, model_path)
print(f"    Saved model: {model_path}")

metrics_df = pd.DataFrame([{
    "model": "Logistic Regression",
    "train_auc": round(train_auc, 4),
    "test_auc":  round(test_auc, 4),
    "ks_stat":   round(test_ks, 4),
    "gini":      round(test_gini, 4),
    "threshold": threshold
}])
metrics_path = RESULTS_PATH / "baseline_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"    Saved metrics: {metrics_path}")


# %% [10] Final summary
print("\n" + "=" * 60)
print("STAGE 06 COMPLETE — Logistic Regression Baseline")
print("=" * 60)
print(f"  Test AUC  : {test_auc:.4f}")
print(f"  KS Stat   : {test_ks:.4f}")
print(f"  Gini      : {test_gini:.4f}")
print(f"  Model saved to: {model_path}")
print(f"\n  Next step: Run scripts/07_advanced_modelling.py")
