"""
scripts/06_model_evaluation.py
Purpose : Evaluate the best BERT checkpoint on the held-out
          test set. Produces final benchmark numbers and all
          comparison charts for the Streamlit dashboard.
Inputs  : outputs/models/bert_fake_news/
          outputs/results/test.csv
          outputs/results/baseline_results.csv
Outputs : outputs/results/bert_results.csv
          outputs/results/model_comparison.csv
          outputs/charts/09_bert_confusion_matrix.png
          outputs/charts/10_bert_roc_curve.png
          outputs/charts/11_model_comparison.png
          outputs/charts/12_training_curves.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

_SCRIPT_DIR  = Path(__file__).parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_CONFIG_PATH = _PROJECT_DIR / "configs" / "config.yaml"

with open(_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RESULTS_PATH = _PROJECT_DIR / config["paths"]["outputs_results"]
MODELS_PATH  = _PROJECT_DIR / config["paths"]["outputs_models"]
CHARTS_PATH  = _PROJECT_DIR / config["paths"]["outputs_charts"]
BERT_DIR     = MODELS_PATH / "bert_fake_news"

RANDOM_STATE = config["project"]["random_state"]
MAX_SEQ_LEN  = config["model"]["max_seq_len"]
THRESHOLD    = config["evaluation"]["threshold"]

sys.path.insert(0, str(_PROJECT_DIR))
from src.dataset import FakeNewsDataset
from src.trainer import get_device

sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = {"Real": "#2ecc71", "Fake": "#e74c3c"}

print("=" * 60)
print("  Stage 6 -- BERT Model Evaluation")
print("=" * 60)

# %% [1] Device and model loading
device = get_device()
print(f"\nLoading best BERT checkpoint from: {BERT_DIR}")
tokenizer = BertTokenizerFast.from_pretrained(str(BERT_DIR))
model     = BertForSequenceClassification.from_pretrained(str(BERT_DIR))
model     = model.to(device)
model.eval()
print("Model loaded and set to eval mode.")

# %% [2] Load test set
print("\nLoading test set...")
test_df = pd.read_csv(RESULTS_PATH / "test.csv")
X_test  = test_df["text"].tolist()
y_test  = test_df["label"].tolist()
print(f"Test samples : {len(X_test):,}")

test_dataset = FakeNewsDataset(X_test, y_test, tokenizer, MAX_SEQ_LEN)
test_loader  = DataLoader(test_dataset, batch_size=64,
                          shuffle=False, num_workers=0)

# %% [3] Run inference on test set
print("\nRunning inference on test set...")
all_preds, all_proba, all_labels = [], [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"]
        outputs        = model(input_ids=input_ids,
                               attention_mask=attention_mask)
        proba = torch.softmax(outputs.logits, dim=1)[:, 1]
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_proba.extend(proba.cpu().numpy())
        all_labels.extend(labels.numpy())

y_pred  = np.array(all_preds)
y_proba = np.array(all_proba)
y_true  = np.array(all_labels)

# %% [4] Compute metrics
accuracy  = accuracy_score(y_true, y_pred)
f1_macro  = f1_score(y_true, y_pred, average="macro")
f1_fake   = f1_score(y_true, y_pred, pos_label=1, average="binary")
f1_real   = f1_score(y_true, y_pred, pos_label=0, average="binary")
precision = precision_score(y_true, y_pred, average="macro")
recall    = recall_score(y_true, y_pred, average="macro")
roc_auc   = roc_auc_score(y_true, y_proba)

print("\n--- BERT Test Set Results ---")
print(f"Accuracy        : {accuracy:.4f}")
print(f"F1 (macro)      : {f1_macro:.4f}")
print(f"F1 (Fake)       : {f1_fake:.4f}")
print(f"F1 (Real)       : {f1_real:.4f}")
print(f"Precision macro : {precision:.4f}")
print(f"Recall macro    : {recall:.4f}")
print(f"ROC-AUC         : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred,
      target_names=["Real (0)", "Fake (1)"]))

# %% [5] Save BERT results
bert_results = pd.DataFrame([{
    "model"    : "BERT fine-tuned",
    "split"    : "test",
    "accuracy" : round(accuracy, 4),
    "f1_macro" : round(f1_macro, 4),
    "f1_fake"  : round(f1_fake,  4),
    "f1_real"  : round(f1_real,  4),
    "precision": round(precision, 4),
    "recall"   : round(recall,   4),
    "roc_auc"  : round(roc_auc,  4),
}])
bert_results.to_csv(RESULTS_PATH / "bert_results.csv", index=False)
print(f"\nSaved : outputs/results/bert_results.csv")

# %% [6] Build model comparison table
baseline_df = pd.read_csv(RESULTS_PATH / "baseline_results.csv")
baseline_test = baseline_df[baseline_df["split"] == "test"].iloc[0]

comparison = pd.DataFrame([
    {
        "model"    : "TF-IDF + LR (baseline)",
        "accuracy" : baseline_test["accuracy"],
        "f1_macro" : baseline_test["f1_macro"],
        "f1_fake"  : baseline_test["f1_fake"],
        "f1_real"  : baseline_test["f1_real"],
        "roc_auc"  : baseline_test["roc_auc"],
    },
    {
        "model"    : "BERT fine-tuned",
        "accuracy" : round(accuracy, 4),
        "f1_macro" : round(f1_macro, 4),
        "f1_fake"  : round(f1_fake,  4),
        "f1_real"  : round(f1_real,  4),
        "roc_auc"  : round(roc_auc,  4),
    },
])
comparison.to_csv(RESULTS_PATH / "model_comparison.csv", index=False)
print(f"Saved : outputs/results/model_comparison.csv")
print("\nModel Comparison (Test Set):")
print(comparison.to_string(index=False))

# %% [7] Chart 9 — BERT confusion matrix
print("\nGenerating charts...")
cm  = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt=",d", cmap="RdYlGn",
    xticklabels=["Predicted Real", "Predicted Fake"],
    yticklabels=["Actual Real",    "Actual Fake"],
    linewidths=0.5, linecolor="white",
    annot_kws={"size": 14, "weight": "bold"}, ax=ax,
)
ax.set_title("BERT fine-tuned\nConfusion Matrix (Test Set)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Actual Label",    fontsize=11)
ax.set_xlabel("Predicted Label", fontsize=11)
fig.text(0.5, -0.02,
         f"TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}",
         ha="center", fontsize=10, color="#555")
plt.tight_layout()
plt.savefig(CHARTS_PATH / "09_bert_confusion_matrix.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 09_bert_confusion_matrix.png")

# %% [8] Chart 10 — BERT ROC curve vs baseline
fpr_bert, tpr_bert, _ = roc_curve(y_true, y_proba)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_bert, tpr_bert, color="#e74c3c", lw=2.5,
        label=f"BERT fine-tuned  (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="#aaa", linestyle="--", lw=1.5,
        label="Random (AUC = 0.50)")
ax.fill_between(fpr_bert, tpr_bert, alpha=0.08, color="#e74c3c")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate",  fontsize=11)
ax.set_title("BERT ROC Curve (Test Set)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(CHARTS_PATH / "10_bert_roc_curve.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 10_bert_roc_curve.png")

# %% [9] Chart 11 — Model comparison bar chart
metrics   = ["accuracy", "f1_macro", "f1_fake", "f1_real", "roc_auc"]
labels    = ["Accuracy", "F1 Macro", "F1 Fake", "F1 Real", "ROC-AUC"]
baseline_vals = [baseline_test[m] for m in metrics]
bert_vals     = [round(accuracy,4), round(f1_macro,4),
                 round(f1_fake,4),  round(f1_real,4), round(roc_auc,4)]

x     = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, baseline_vals, width, label="TF-IDF + LR",
               color="#3498db", alpha=0.85, edgecolor="none")
bars2 = ax.bar(x + width/2, bert_vals,     width, label="BERT fine-tuned",
               color="#e74c3c", alpha=0.85, edgecolor="none")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Score", fontsize=11)
ax.set_title("Model Comparison: TF-IDF + LR vs BERT (Test Set)",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0.93, 1.005)
ax.legend(fontsize=11)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(CHARTS_PATH / "11_model_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 11_model_comparison.png")

# %% [10] Chart 12 — Training curves
log_df = pd.read_csv(RESULTS_PATH / "training_log.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("BERT Fine-Tuning Training Curves",
             fontsize=14, fontweight="bold")

# Loss curve
axes[0].plot(log_df["epoch"], log_df["train_loss"],
             marker="o", color="#e74c3c", lw=2, label="Train Loss")
axes[0].set_title("Training Loss per Epoch", fontsize=12)
axes[0].set_xlabel("Epoch", fontsize=11)
axes[0].set_ylabel("Cross-Entropy Loss", fontsize=11)
axes[0].legend(fontsize=10)
axes[0].set_xticks(log_df["epoch"])
axes[0].grid(True, alpha=0.3)

# F1 curves
axes[1].plot(log_df["epoch"], log_df["train_f1"],
             marker="o", color="#e74c3c", lw=2, label="Train F1")
axes[1].plot(log_df["epoch"], log_df["val_f1_macro"],
             marker="s", color="#2ecc71", lw=2, label="Val F1 Macro")
axes[1].set_title("F1 Score per Epoch", fontsize=12)
axes[1].set_xlabel("Epoch", fontsize=11)
axes[1].set_ylabel("F1 Score (macro)", fontsize=11)
axes[1].legend(fontsize=10)
axes[1].set_xticks(log_df["epoch"])
axes[1].set_ylim(0.94, 1.005)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(CHARTS_PATH / "12_training_curves.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 12_training_curves.png")

# %% [11] Final status
print("\n" + "=" * 60)
print("  Stage 6 COMPLETE")
print("=" * 60)
print(f"BERT Test Accuracy : {accuracy:.4f}")
print(f"BERT Test F1 macro : {f1_macro:.4f}")
print(f"BERT Test ROC-AUC  : {roc_auc:.4f}")
print(f"Baseline F1 macro  : {baseline_test['f1_macro']:.4f}")
print(f"Improvement        : +{(f1_macro - baseline_test['f1_macro'])*100:.2f} pp")
print("\nNext step : python scripts/07_explainability.py")
print("=" * 60)
