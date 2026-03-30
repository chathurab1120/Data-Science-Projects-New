"""
Script: 07_advanced_modelling.py
Purpose: Train XGBoost and LightGBM models with Optuna hyperparameter tuning.
         Compare all three models (LR, XGBoost, LightGBM) on AUC and KS.
         Select and save the champion model.
Inputs:  data/X_train.parquet, data/X_test.parquet
         data/y_train.parquet, data/y_test.parquet
Outputs: outputs/models/xgboost_model.pkl
         outputs/models/lightgbm_model.pkl
         outputs/models/champion_model.pkl
         outputs/charts/09_model_comparison.png
         outputs/results/model_comparison.csv
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
import yaml
import joblib
import warnings
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHARTS_PATH  = Path(config["paths"]["outputs_charts"])
MODELS_PATH  = Path(config["paths"]["outputs_models"])
RESULTS_PATH = Path(config["paths"]["outputs_results"])
RANDOM_STATE = config["project"]["random_state"]
N_TRIALS     = config["model"]["optuna"]["n_trials"]
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
print("STAGE 07 — Advanced Modelling: XGBoost + LightGBM")
print("=" * 60)


# %% [1] Load train/test splits
print("\n[1] Loading train/test splits")
X_train = pd.read_parquet("data/X_train.parquet")
X_test  = pd.read_parquet("data/X_test.parquet")
y_train = pd.read_parquet("data/y_train.parquet").squeeze()
y_test  = pd.read_parquet("data/y_test.parquet").squeeze()

print(f"    X_train: {X_train.shape}")
print(f"    X_test : {X_test.shape}")

# Load baseline metrics for comparison
baseline_metrics = pd.read_csv(RESULTS_PATH / "baseline_metrics.csv")
baseline_auc = baseline_metrics["test_auc"].values[0]
baseline_ks  = baseline_metrics["ks_stat"].values[0]
print(f"\n    Baseline (LR) — AUC: {baseline_auc:.4f} | KS: {baseline_ks:.4f}")


# %% [2] Sample for Optuna tuning
# Running Optuna on the full 826K row training set would take hours.
# We tune on a stratified 20% sample — fast and representative.
# Final model is then trained on full data with best params.
# Interview point: this is standard practice for large-dataset tuning.

print("\n[2] Preparing Optuna tuning sample (20% stratified)")
from sklearn.model_selection import train_test_split

X_tune, _, y_tune, _ = train_test_split(
    X_train, y_train,
    test_size=0.80,
    stratify=y_train,
    random_state=RANDOM_STATE
)
print(f"    Tuning sample: {X_tune.shape[0]:,} rows")
print(f"    Default rate in tuning sample: {y_tune.mean():.4f}")


# %% [3] XGBoost — default run first
print("\n[3] XGBoost — default parameters")
t0 = time.time()

xgb_default = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=int((y_train == 0).sum() / (y_train == 1).sum()),
    random_state=RANDOM_STATE,
    eval_metric="auc",
    early_stopping_rounds=30,
    verbosity=0
)

xgb_default.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

xgb_default_auc = roc_auc_score(y_test, xgb_default.predict_proba(X_test)[:, 1])
t_xgb_default = time.time() - t0
print(f"    Default XGBoost AUC: {xgb_default_auc:.4f}  (took {t_xgb_default:.1f}s)")


# %% [4] XGBoost — Optuna hyperparameter tuning
# We optimise 6 key hyperparameters using 5-fold cross-validation AUC.
# Optuna uses Bayesian optimisation (TPE sampler) — smarter than GridSearch.
# Interview point: Optuna is the industry standard for hyperparameter tuning.

print(f"\n[4] XGBoost — Optuna tuning ({N_TRIALS} trials, 5-fold CV)")
print("    This will take 5-10 minutes...")

def xgb_objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 6),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample":        trial.suggest_float("subsample", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.95),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
        "scale_pos_weight": int((y_tune == 0).sum() / (y_tune == 1).sum()),
        "random_state":     RANDOM_STATE,
        "verbosity":        0,
        "eval_metric":      "auc"
    }
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_tune, y_tune, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    return scores.mean()

xgb_study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)

best_xgb_params = xgb_study.best_params
best_xgb_params.update({
    "scale_pos_weight": int((y_train == 0).sum() / (y_train == 1).sum()),
    "random_state": RANDOM_STATE,
    "verbosity": 0,
    "eval_metric": "auc",
    "early_stopping_rounds": 30
})

print(f"    Best CV AUC: {xgb_study.best_value:.4f}")
print(f"    Best params: {xgb_study.best_params}")


# %% [5] XGBoost — train final model with best params on full training data
print("\n[5] Training final XGBoost on full training data")
t0 = time.time()

xgb_final = xgb.XGBClassifier(**best_xgb_params)
xgb_final.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

xgb_proba      = xgb_final.predict_proba(X_test)[:, 1]
xgb_train_proba = xgb_final.predict_proba(X_train)[:, 1]
xgb_test_auc   = roc_auc_score(y_test, xgb_proba)
xgb_train_auc  = roc_auc_score(y_train, xgb_train_proba)

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)
xgb_ks  = (tpr_xgb - fpr_xgb).max()
xgb_gini = 2 * xgb_test_auc - 1
t_xgb = time.time() - t0

print(f"    XGBoost Train AUC : {xgb_train_auc:.4f}")
print(f"    XGBoost Test AUC  : {xgb_test_auc:.4f}")
print(f"    XGBoost KS        : {xgb_ks:.4f}")
print(f"    XGBoost Gini      : {xgb_gini:.4f}")
print(f"    Training time     : {t_xgb:.1f}s")
print(f"    Improvement over LR baseline: +{xgb_test_auc - baseline_auc:.4f} AUC")

joblib.dump(xgb_final, MODELS_PATH / "xgboost_model.pkl")
print("    Saved: outputs/models/xgboost_model.pkl")


# %% [6] LightGBM — Optuna hyperparameter tuning
print(f"\n[6] LightGBM — Optuna tuning ({N_TRIALS} trials, 5-fold CV)")
print("    This will take 3-5 minutes...")

def lgb_objective(trial):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
        "num_leaves":      trial.suggest_int("num_leaves", 31, 127),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample":       trial.suggest_float("subsample", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.95),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "class_weight":    "balanced",
        "random_state":    RANDOM_STATE,
        "verbosity":       -1,
        "force_col_wise":  True
    }
    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_tune, y_tune, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    return scores.mean()

lgb_study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
lgb_study.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=False)

best_lgb_params = lgb_study.best_params
best_lgb_params.update({
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "verbosity": -1,
    "force_col_wise": True
})

print(f"    Best CV AUC: {lgb_study.best_value:.4f}")
print(f"    Best params: {lgb_study.best_params}")


# %% [7] LightGBM — train final model on full training data
print("\n[7] Training final LightGBM on full training data")
t0 = time.time()

lgb_final = lgb.LGBMClassifier(**best_lgb_params)
lgb_final.fit(X_train, y_train)

lgb_proba       = lgb_final.predict_proba(X_test)[:, 1]
lgb_train_proba = lgb_final.predict_proba(X_train)[:, 1]
lgb_test_auc    = roc_auc_score(y_test, lgb_proba)
lgb_train_auc   = roc_auc_score(y_train, lgb_train_proba)

fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgb_proba)
lgb_ks   = (tpr_lgb - fpr_lgb).max()
lgb_gini = 2 * lgb_test_auc - 1
t_lgb = time.time() - t0

print(f"    LightGBM Train AUC : {lgb_train_auc:.4f}")
print(f"    LightGBM Test AUC  : {lgb_test_auc:.4f}")
print(f"    LightGBM KS        : {lgb_ks:.4f}")
print(f"    LightGBM Gini      : {lgb_gini:.4f}")
print(f"    Training time      : {t_lgb:.1f}s")
print(f"    Improvement over LR baseline: +{lgb_test_auc - baseline_auc:.4f} AUC")

joblib.dump(lgb_final, MODELS_PATH / "lightgbm_model.pkl")
print("    Saved: outputs/models/lightgbm_model.pkl")


# %% [8] Model comparison and champion selection
# Champion = highest test AUC on out-of-time test set.
# We document training time so the deployment team can assess latency.
# Interview point: LightGBM is often faster than XGBoost on large datasets
# with comparable AUC — worth documenting this trade-off explicitly.

print("\n[8] Model comparison — selecting champion")

comparison = pd.DataFrame([
    {"model": "Logistic Regression", "train_auc": None,
     "test_auc": baseline_auc, "ks": baseline_ks,
     "gini": round(2*baseline_auc-1, 4), "train_time_s": None},
    {"model": "XGBoost",   "train_auc": round(xgb_train_auc, 4),
     "test_auc": round(xgb_test_auc, 4), "ks": round(xgb_ks, 4),
     "gini": round(xgb_gini, 4), "train_time_s": round(t_xgb, 1)},
    {"model": "LightGBM",  "train_auc": round(lgb_train_auc, 4),
     "test_auc": round(lgb_test_auc, 4), "ks": round(lgb_ks, 4),
     "gini": round(lgb_gini, 4), "train_time_s": round(t_lgb, 1)},
])

print(f"\n    {'Model':<25} {'Test AUC':>10} {'KS':>8} {'Gini':>8} {'Time(s)':>10}")
print(f"    {'-'*65}")
for _, row in comparison.iterrows():
    t = f"{row['train_time_s']:.1f}" if row['train_time_s'] else "N/A"
    print(f"    {row['model']:<25} {row['test_auc']:>10.4f} {row['ks']:>8.4f} {row['gini']:>8.4f} {t:>10}")

# Select champion by highest test AUC
gb_models = {"XGBoost": (xgb_final, xgb_test_auc), "LightGBM": (lgb_final, lgb_test_auc)}
champion_name  = max(gb_models, key=lambda k: gb_models[k][1])
champion_model = gb_models[champion_name][0]
champion_auc   = gb_models[champion_name][1]

print(f"\n    Champion model: {champion_name} (Test AUC: {champion_auc:.4f})")
joblib.dump(champion_model, MODELS_PATH / "champion_model.pkl")
print("    Saved: outputs/models/champion_model.pkl")

comparison.to_csv(RESULTS_PATH / "model_comparison.csv", index=False)
print("    Saved: outputs/results/model_comparison.csv")


# %% [9] Chart — Model Comparison Bar Chart
print("\n[9] Plotting model comparison chart")

fig, axes = plt.subplots(1, 3, figsize=(14, 6))
metrics_plot = ["test_auc", "ks", "gini"]
titles       = ["Test AUC", "KS Statistic", "Gini Coefficient"]
targets      = [0.72, 0.35, 0.44]
colors       = ["#3498db", "#e74c3c", "#2ecc71"]

for ax, metric, title, target in zip(axes, metrics_plot, titles, targets):
    vals   = comparison[metric].fillna(0)
    models = comparison["model"]
    bars   = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(y=target, color="#e67e22", linestyle="--", linewidth=1.5,
               label=f"Target: {target}")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=9)

plt.suptitle("Model Comparison: LR vs XGBoost vs LightGBM",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
save_path = CHARTS_PATH / "09_model_comparison.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {save_path}")


# %% [10] Final summary
print("\n" + "=" * 60)
print("STAGE 07 COMPLETE — Advanced Modelling")
print("=" * 60)
print(f"  Baseline LR AUC  : {baseline_auc:.4f}")
print(f"  XGBoost  Test AUC: {xgb_test_auc:.4f}")
print(f"  LightGBM Test AUC: {lgb_test_auc:.4f}")
print(f"  Champion model   : {champion_name}")
print(f"  Champion AUC     : {champion_auc:.4f}")
print(f"\n  Next step: Run scripts/08_model_evaluation.py")
