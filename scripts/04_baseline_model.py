# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
scripts/04_baseline_model.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Purpose : Train a TF-IDF + Logistic Regression baseline.
          Establishes the benchmark that BERT must beat.
          Industry practice: always build a fast, interpretable
          baseline before investing in deep learning.
Inputs  : outputs/results/train.csv
          outputs/results/val.csv
          outputs/results/test.csv
Outputs : outputs/models/baseline_tfidf_lr.pkl
          outputs/results/baseline_results.csv
          outputs/charts/07_baseline_confusion_matrix.png
          outputs/charts/08_baseline_roc_curve.png
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# %% [0] Imports and configuration
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import sys
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve,
)

_SCRIPT_DIR  = Path(__file__).parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_CONFIG_PATH = _PROJECT_DIR / "configs" / "config.yaml"

with open(_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RESULTS_PATH = _PROJECT_DIR / config["paths"]["outputs_results"]
MODELS_PATH  = _PROJECT_DIR / config["paths"]["outputs_models"]
CHARTS_PATH  = _PROJECT_DIR / config["paths"]["outputs_charts"]
MODELS_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_STATE     = config["project"]["random_state"]
TFIDF_MAX_FEAT   = config["baseline"]["tfidf_max_features"]
TFIDF_NGRAM_MIN  = config["baseline"]["tfidf_ngram_min"]
TFIDF_NGRAM_MAX  = config["baseline"]["tfidf_ngram_max"]
LR_MAX_ITER      = config["baseline"]["lr_max_iter"]
THRESHOLD        = config["evaluation"]["threshold"]

sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = {0: "#2ecc71", 1: "#e74c3c"}

sys.path.insert(0, str(_PROJECT_DIR))
from src.utils import print_section

print("=" * 60)
print("  Stage 4 -- TF-IDF + Logistic Regression Baseline")
print("=" * 60)


# %% [1] Load preprocessed splits
print_section("Step 1: Load Train / Val / Test Splits")
train_df = pd.read_csv(RESULTS_PATH / "train.csv")
val_df   = pd.read_csv(RESULTS_PATH / "val.csv")
test_df  = pd.read_csv(RESULTS_PATH / "test.csv")

X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
X_val,   y_val   = val_df["text"].tolist(),   val_df["label"].tolist()
X_test,  y_test  = test_df["text"].tolist(),  test_df["label"].tolist()

print(f"Train : {len(X_train):,} samples")
print(f"Val   : {len(X_val):,} samples")
print(f"Test  : {len(X_test):,} samples")


# %% [2] Build TF-IDF + Logistic Regression pipeline
# TF-IDF unigrams + bigrams capture single keywords AND common two-word phrases
# class_weight='balanced' handles any residual class imbalance automatically
# This pipeline is the industry-standard NLP baseline — fast, interpretable,
# and surprisingly competitive (~93-94% F1 on clean datasets)
print_section("Step 2: Build and Train Pipeline")
print(f"TF-IDF max_features : {TFIDF_MAX_FEAT:,}")
print(f"TF-IDF ngram range  : ({TFIDF_NGRAM_MIN}, {TFIDF_NGRAM_MAX})")
print(f"LR max_iter         : {LR_MAX_ITER}")
print(f"LR class_weight     : balanced")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features = TFIDF_MAX_FEAT,
        ngram_range  = (TFIDF_NGRAM_MIN, TFIDF_NGRAM_MAX),
        sublinear_tf = True,    # apply log(1+tf) — reduces impact of very frequent terms
        strip_accents = "unicode",
        analyzer     = "word",
        min_df       = 2,       # ignore terms appearing in fewer than 2 docs
    )),
    ("clf", LogisticRegression(
        C            = 1.0,
        max_iter     = LR_MAX_ITER,
        class_weight = "balanced",
        solver       = "saga",   # saga handles large sparse matrices well
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )),
])

print("\nFitting pipeline on training data ...")
t0 = time.time()
pipeline.fit(X_train, y_train)
train_time = time.time() - t0
print(f"Training complete in {train_time:.1f}s")


# %% [3] Evaluate on validation set
print_section("Step 3: Validation Set Evaluation")

val_preds       = pipeline.predict(X_val)
val_proba       = pipeline.predict_proba(X_val)[:, 1]
val_accuracy    = accuracy_score(y_val, val_preds)
val_f1_macro    = f1_score(y_val, val_preds, average="macro")
val_f1_fake     = f1_score(y_val, val_preds, pos_label=1, average="binary")
val_f1_real     = f1_score(y_val, val_preds, pos_label=0, average="binary")
val_precision   = precision_score(y_val, val_preds, average="macro")
val_recall      = recall_score(y_val, val_preds, average="macro")
val_roc_auc     = roc_auc_score(y_val, val_proba)

print(f"Accuracy        : {val_accuracy:.4f}")
print(f"F1 (macro)      : {val_f1_macro:.4f}")
print(f"F1 (Fake/pos)   : {val_f1_fake:.4f}")
print(f"F1 (Real/neg)   : {val_f1_real:.4f}")
print(f"Precision macro : {val_precision:.4f}")
print(f"Recall macro    : {val_recall:.4f}")
print(f"ROC-AUC         : {val_roc_auc:.4f}")


# %% [4] Evaluate on test set — final holdout numbers
print_section("Step 4: Test Set Evaluation (Final Holdout)")

test_preds       = pipeline.predict(X_test)
test_proba       = pipeline.predict_proba(X_test)[:, 1]
test_accuracy    = accuracy_score(y_test, test_preds)
test_f1_macro    = f1_score(y_test, test_preds, average="macro")
test_f1_fake     = f1_score(y_test, test_preds, pos_label=1, average="binary")
test_f1_real     = f1_score(y_test, test_preds, pos_label=0, average="binary")
test_precision   = precision_score(y_test, test_preds, average="macro")
test_recall      = recall_score(y_test, test_preds, average="macro")
test_roc_auc     = roc_auc_score(y_test, test_proba)

print(f"Accuracy        : {test_accuracy:.4f}")
print(f"F1 (macro)      : {test_f1_macro:.4f}")
print(f"F1 (Fake/pos)   : {test_f1_fake:.4f}")
print(f"F1 (Real/neg)   : {test_f1_real:.4f}")
print(f"Precision macro : {test_precision:.4f}")
print(f"Recall macro    : {test_recall:.4f}")
print(f"ROC-AUC         : {test_roc_auc:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, test_preds,
      target_names=["Real (0)", "Fake (1)"]))


# %% [5] Chart 7 — Confusion matrix
print_section("Step 5: Generate Charts")
print("Generating chart 7: confusion matrix ...")

cm = confusion_matrix(y_test, test_preds)
fig, ax = plt.subplots(figsize=(7, 6))

sns.heatmap(
    cm, annot=True, fmt=",d", cmap="RdYlGn",
    xticklabels=["Predicted Real", "Predicted Fake"],
    yticklabels=["Actual Real",    "Actual Fake"],
    linewidths=0.5, linecolor="white",
    annot_kws={"size": 14, "weight": "bold"},
    ax=ax,
)
ax.set_title("Baseline: TF-IDF + Logistic Regression\nConfusion Matrix (Test Set)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Actual Label",    fontsize=11)
ax.set_xlabel("Predicted Label", fontsize=11)

# Annotate TN/FP/FN/TP for interview clarity
tn, fp, fn, tp = cm.ravel()
fig.text(0.5, -0.02,
         f"TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}",
         ha="center", fontsize=10, color="#555")

plt.tight_layout()
out_path = CHARTS_PATH / "07_baseline_confusion_matrix.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out_path.name}")


# %% [6] Chart 8 — ROC curve
print("Generating chart 8: ROC curve ...")

fpr, tpr, _ = roc_curve(y_test, test_proba)
fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(fpr, tpr, color="#e74c3c", lw=2.5,
        label=f"TF-IDF + LR  (AUC = {test_roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="#aaa", linestyle="--", lw=1.5,
        label="Random Classifier (AUC = 0.50)")
ax.fill_between(fpr, tpr, alpha=0.08, color="#e74c3c")

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate",  fontsize=11)
ax.set_title("Baseline ROC Curve (Test Set)", fontsize=13,
             fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = CHARTS_PATH / "08_baseline_roc_curve.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out_path.name}")


# %% [7] Save model pipeline
print_section("Step 6: Save Model")
model_path = MODELS_PATH / "baseline_tfidf_lr.pkl"
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)
print(f"Saved : {model_path}")


# %% [8] Save results to CSV for Streamlit dashboard
print_section("Step 7: Save Results CSV")
results = [
    {"model": "TF-IDF + LR", "split": "val",
     "accuracy": round(val_accuracy, 4),
     "f1_macro": round(val_f1_macro, 4),
     "f1_fake":  round(val_f1_fake,  4),
     "f1_real":  round(val_f1_real,  4),
     "precision":round(val_precision,4),
     "recall":   round(val_recall,   4),
     "roc_auc":  round(val_roc_auc,  4)},
    {"model": "TF-IDF + LR", "split": "test",
     "accuracy": round(test_accuracy, 4),
     "f1_macro": round(test_f1_macro, 4),
     "f1_fake":  round(test_f1_fake,  4),
     "f1_real":  round(test_f1_real,  4),
     "precision":round(test_precision,4),
     "recall":   round(test_recall,   4),
     "roc_auc":  round(test_roc_auc,  4)},
]
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_PATH / "baseline_results.csv", index=False)
print(f"Saved : outputs/results/baseline_results.csv")
print(f"\nBaseline results:")
print(results_df.to_string(index=False))


# %% [9] Final status
print("\n" + "=" * 60)
print("  Stage 4 COMPLETE")
print("=" * 60)
print(f"Baseline Test Accuracy : {test_accuracy:.4f}")
print(f"Baseline Test F1 macro : {test_f1_macro:.4f}")
print(f"Baseline Test ROC-AUC  : {test_roc_auc:.4f}")
print(f"\nBERT target            : F1 macro > {test_f1_macro:.4f}")
print("\nNext step : python scripts/05_bert_training.py")
print("=" * 60)

