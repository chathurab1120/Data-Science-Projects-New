"""
scripts/07_explainability.py
Purpose : Generate LIME token-level explanations for 12 sample
          predictions from the BERT model (6 Fake, 6 Real).
          Saves explanation charts and a summary CSV for the
          Streamlit dashboard explainability page.
Inputs  : outputs/models/bert_fake_news/
          outputs/results/test.csv
Outputs : outputs/charts/lime_fake_01..06.png
          outputs/charts/lime_real_01..06.png
          outputs/results/lime_samples.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

sys.path.insert(0, str(_PROJECT_DIR))
from src.trainer import get_device

print("=" * 60)
print("  Stage 7 -- LIME Explainability")
print("=" * 60)

# %% [1] Load model
device    = get_device()
tokenizer = BertTokenizerFast.from_pretrained(str(BERT_DIR))
model     = BertForSequenceClassification.from_pretrained(str(BERT_DIR))
model     = model.to(device)
model.eval()
print("Model loaded.")

# %% [2] Define BERT predict function for LIME
# LIME calls this function with a list of perturbed text strings
# and expects a numpy array of shape (n_samples, n_classes)
def bert_predict_proba(texts: list) -> np.ndarray:
    """
    Wrapper that feeds a list of strings through BERT and returns
    class probabilities. LIME uses this to understand which words
    most influence the prediction by masking them out.
    """
    all_proba = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoding = tokenizer(
            batch_texts,
            max_length     = MAX_SEQ_LEN,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            proba = torch.softmax(outputs.logits, dim=1)
        all_proba.append(proba.cpu().numpy())
    return np.vstack(all_proba)

# %% [3] Load test samples
print("\nLoading test set...")
test_df = pd.read_csv(RESULTS_PATH / "test.csv")

# Select 6 correctly-predicted Fake and 6 correctly-predicted Real articles
# Run a quick inference pass to identify correct predictions
print("Identifying correctly predicted samples for explanation...")
sample_pool = test_df.sample(n=500, random_state=RANDOM_STATE).reset_index(drop=True)
texts_pool  = sample_pool["text"].tolist()
labels_pool = sample_pool["label"].tolist()

proba_pool  = bert_predict_proba(texts_pool)
preds_pool  = np.argmax(proba_pool, axis=1)

correct_fake = [i for i in range(len(preds_pool))
                if labels_pool[i] == 1 and preds_pool[i] == 1][:6]
correct_real = [i for i in range(len(preds_pool))
                if labels_pool[i] == 0 and preds_pool[i] == 0][:6]

print(f"Correct Fake predictions selected : {len(correct_fake)}")
print(f"Correct Real predictions selected : {len(correct_real)}")

# %% [4] LIME explainer setup
explainer = LimeTextExplainer(
    class_names    = ["Real", "Fake"],
    random_state   = RANDOM_STATE,
    bow            = True,   # bag-of-words perturbation — most stable for BERT
    mask_string    = "MASK", # token used to replace masked words
)

# %% [5] Generate and save LIME explanations
lime_records = []
N_FEATURES   = 12   # top N words to highlight per explanation
N_SAMPLES    = 500  # LIME perturbation samples — balances speed vs stability

def save_lime_chart(exp, text: str, true_label: int, pred_label: int,
                    confidence: float, out_path: Path, sample_num: int):
    """
    Render a LIME explanation as a horizontal bar chart showing which
    words pushed the prediction toward Fake vs Real.
    Green bars = evidence for Real, Red bars = evidence for Fake.
    """
    features = exp.as_list(label=exp.available_labels()[0])
    words    = [f[0] for f in features]
    weights  = [f[1] for f in features]

    colors = ["#e74c3c" if w > 0 else "#2ecc71" for w in weights]
    words_disp  = [w[:25] for w in words]  # truncate long words

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(words_disp)), weights,
                   color=colors, alpha=0.85, edgecolor="none")
    ax.set_yticks(range(len(words_disp)))
    ax.set_yticklabels(words_disp, fontsize=10)
    ax.axvline(0, color="#333", linewidth=1.2)
    ax.set_xlabel("LIME Weight  (positive = Fake, negative = Real)", fontsize=10)

    true_name = "Fake" if true_label == 1 else "Real"
    pred_name = "Fake" if pred_label == 1 else "Real"
    title = (f"LIME Explanation -- Sample {sample_num}\n"
             f"True: {true_name}  |  Predicted: {pred_name}  |  "
             f"Confidence: {confidence:.1%}")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    # Legend
    fake_patch = mpatches.Patch(color="#e74c3c", label="Evidence for Fake")
    real_patch = mpatches.Patch(color="#2ecc71", label="Evidence for Real")
    ax.legend(handles=[fake_patch, real_patch], fontsize=9,
              loc="lower right")

    # Show article snippet below chart
    snippet = text[:200].replace("\n", " ")
    fig.text(0.5, -0.04, f'"{snippet}..."',
             ha="center", fontsize=8, color="#555",
             wrap=True, style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# Generate for Fake samples
print("\nGenerating LIME explanations for Fake News samples...")
for chart_num, idx in enumerate(correct_fake, 1):
    text       = texts_pool[idx]
    true_label = labels_pool[idx]
    confidence = float(proba_pool[idx][1])
    pred_label = 1

    print(f"  Fake sample {chart_num}/6  (confidence={confidence:.3f}) ...")
    exp = explainer.explain_instance(
        text,
        bert_predict_proba,
        num_features = N_FEATURES,
        num_samples  = N_SAMPLES,
        labels       = (1,),
    )
    out_path = CHARTS_PATH / f"lime_fake_{chart_num:02d}.png"
    save_lime_chart(exp, text, true_label, pred_label=1,
                    confidence=confidence,
                    out_path=out_path, sample_num=chart_num)
    print(f"    Saved: {out_path.name}")

    available_labels = list(exp.local_exp.keys())
    lime_label = (
        pred_label if "pred_label" in locals() and pred_label in available_labels else available_labels[0]
    )
    top_words = exp.as_list(label=lime_label)[:5]
    lime_records.append({
        "sample_num"   : chart_num,
        "true_label"   : "Fake",
        "pred_label"   : "Fake",
        "confidence"   : round(confidence, 4),
        "top_words"    : str([w for w, _ in top_words]),
        "top_weights"  : str([round(w, 4) for _, w in top_words]),
        "text_snippet" : text[:150],
    })

# Generate for Real samples
print("\nGenerating LIME explanations for Real News samples...")
for chart_num, idx in enumerate(correct_real, 1):
    text       = texts_pool[idx]
    true_label = labels_pool[idx]
    confidence = float(proba_pool[idx][0])
    pred_label = 0

    print(f"  Real sample {chart_num}/6  (confidence={confidence:.3f}) ...")
    exp = explainer.explain_instance(
        text,
        bert_predict_proba,
        num_features = N_FEATURES,
        num_samples  = N_SAMPLES,
        labels       = (0,),
    )
    out_path = CHARTS_PATH / f"lime_real_{chart_num:02d}.png"
    save_lime_chart(exp, text, true_label, pred_label=0,
                    confidence=confidence,
                    out_path=out_path, sample_num=chart_num)
    print(f"    Saved: {out_path.name}")

    available_labels = list(exp.local_exp.keys())
    lime_label = (
        pred_label if "pred_label" in locals() and pred_label in available_labels else available_labels[0]
    )
    top_words = exp.as_list(label=lime_label)[:5]
    lime_records.append({
        "sample_num"   : chart_num,
        "true_label"   : "Real",
        "pred_label"   : "Real",
        "confidence"   : round(confidence, 4),
        "top_words"    : str([w for w, _ in top_words]),
        "top_weights"  : str([round(w, 4) for _, w in top_words]),
        "text_snippet" : text[:150],
    })

# %% [6] Save LIME summary CSV
lime_df = pd.DataFrame(lime_records)
lime_df.to_csv(RESULTS_PATH / "lime_samples.csv", index=False)
print(f"\nSaved : outputs/results/lime_samples.csv")
print(lime_df[["sample_num","true_label","confidence","top_words"]].to_string(index=False))

# %% [7] Final status
print("\n" + "=" * 60)
print("  Stage 7 COMPLETE")
print("=" * 60)
lime_charts = sorted(CHARTS_PATH.glob("lime_*.png"))
print(f"LIME charts saved : {len(lime_charts)}")
for c in lime_charts:
    print(f"  {c.name}")
print("\nNext step : build Streamlit dashboard")
print("  app/streamlit_app.py")
print("=" * 60)
