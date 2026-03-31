# -*- coding: utf-8 -*-
"""
scripts/05_bert_training.py
Purpose : Fine-tune bert-base-uncased on WELFake train split.
          GPU-accelerated with FP16 mixed precision.
          Saves best checkpoint by validation F1 macro.
Inputs  : outputs/results/train.csv
          outputs/results/val.csv
Outputs : outputs/models/bert_fake_news/
          outputs/results/training_log.csv
"""

# %% [0] Imports and configuration
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, accuracy_score

_SCRIPT_DIR  = Path(__file__).parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_CONFIG_PATH = _PROJECT_DIR / "configs" / "config.yaml"

with open(_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RESULTS_PATH = _PROJECT_DIR / config["paths"]["outputs_results"]
MODELS_PATH  = _PROJECT_DIR / config["paths"]["outputs_models"]
BERT_DIR     = MODELS_PATH / "bert_fake_news"
BERT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = config["project"]["random_state"]
MODEL_NAME   = config["model"]["name"]
MAX_SEQ_LEN  = config["model"]["max_seq_len"]
BATCH_SIZE   = config["model"]["batch_size"]
EPOCHS       = config["model"]["epochs"]
LR           = config["model"]["learning_rate"]
WARMUP_RATIO = config["model"]["warmup_ratio"]
WEIGHT_DECAY = config["model"]["weight_decay"]
FP16         = config["model"]["fp16"]

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

sys.path.insert(0, str(_PROJECT_DIR))
from src.dataset import FakeNewsDataset
from src.trainer import get_device, evaluate_model, print_gpu_memory

print("=" * 60)
print("  Stage 5 -- BERT Fine-Tuning")
print("=" * 60)

# %% [1] Device setup
device = get_device()
print(f"Training device : {device}")
print(f"FP16 enabled    : {FP16}")
print(f"Model           : {MODEL_NAME}")
print(f"Max seq len     : {MAX_SEQ_LEN}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Epochs          : {EPOCHS}")
print(f"Learning rate   : {LR}")

# %% [2] Load splits
print("\nLoading splits...")
train_df = pd.read_csv(RESULTS_PATH / "train.csv")
val_df   = pd.read_csv(RESULTS_PATH / "val.csv")
X_train  = train_df["text"].tolist()
y_train  = train_df["label"].tolist()
X_val    = val_df["text"].tolist()
y_val    = val_df["label"].tolist()
print(f"Train : {len(X_train):,}  |  Val : {len(X_val):,}")

# %% [3] Tokenizer and DataLoaders
print(f"\nLoading tokenizer: {MODEL_NAME} ...")
tokenizer     = BertTokenizerFast.from_pretrained(MODEL_NAME)
train_dataset = FakeNewsDataset(X_train, y_train, tokenizer, MAX_SEQ_LEN)
val_dataset   = FakeNewsDataset(X_val,   y_val,   tokenizer, MAX_SEQ_LEN)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                           shuffle=True,  num_workers=0, pin_memory=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE * 2,
                           shuffle=False, num_workers=0, pin_memory=True)
print(f"Train batches : {len(train_loader):,}")
print(f"Val batches   : {len(val_loader):,}")

# %% [4] Load model
print(f"\nLoading {MODEL_NAME} ...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2,
    id2label={0: "Real", 1: "Fake"},
    label2id={"Real": 0, "Fake": 1},
)
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters : {total_params:,}")
print_gpu_memory()

# %% [5] Optimizer and scheduler
total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
optimizer    = AdamW(model.parameters(), lr=LR,
                     weight_decay=WEIGHT_DECAY, eps=1e-8)
scheduler    = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)
scaler = GradScaler("cuda") if FP16 else None
print(f"\nTotal steps  : {total_steps:,}")
print(f"Warmup steps : {warmup_steps:,}")

# %% [6] Training loop
training_log = []
best_val_f1  = 0.0
best_epoch   = 0
patience     = 2
no_improve   = 0
t_start      = time.time()

for epoch in range(1, EPOCHS + 1):
    t_epoch      = time.time()
    model.train()
    total_loss   = 0.0
    train_preds  = []
    train_labels = []

    print(f"\n--- Epoch {epoch}/{EPOCHS} ---")

    for step, batch in enumerate(train_loader, 1):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()

        if FP16:
            with autocast("cuda"):
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

        if step % 200 == 0 or step == len(train_loader):
            avg_loss   = total_loss / step
            train_f1   = f1_score(train_labels, train_preds,
                                  average="macro", zero_division=0)
            elapsed    = time.time() - t_epoch
            eta        = (elapsed / step) * (len(train_loader) - step)
            print(f"  Step {step:>4}/{len(train_loader)}"
                  f"  loss={avg_loss:.4f}"
                  f"  train_f1={train_f1:.4f}"
                  f"  elapsed={elapsed:.0f}s"
                  f"  eta={eta:.0f}s")

    epoch_loss     = total_loss / len(train_loader)
    epoch_train_f1 = f1_score(train_labels, train_preds,
                               average="macro", zero_division=0)
    epoch_train_acc = accuracy_score(train_labels, train_preds)
    val_metrics     = evaluate_model(model, val_loader, device)
    epoch_time      = time.time() - t_epoch

    print(f"\n  Epoch {epoch} Summary:")
    print(f"    Train loss     : {epoch_loss:.4f}")
    print(f"    Train accuracy : {epoch_train_acc:.4f}")
    print(f"    Train F1 macro : {epoch_train_f1:.4f}")
    print(f"    Val accuracy   : {val_metrics['accuracy']:.4f}")
    print(f"    Val F1 macro   : {val_metrics['f1_macro']:.4f}")
    print(f"    Val F1 fake    : {val_metrics['f1_fake']:.4f}")
    print(f"    Val F1 real    : {val_metrics['f1_real']:.4f}")
    print(f"    Epoch time     : {epoch_time:.0f}s")
    print_gpu_memory()

    if val_metrics["f1_macro"] > best_val_f1:
        best_val_f1 = val_metrics["f1_macro"]
        best_epoch  = epoch
        no_improve  = 0
        model.save_pretrained(str(BERT_DIR))
        tokenizer.save_pretrained(str(BERT_DIR))
        print(f"    --> Best model saved (val F1={best_val_f1:.4f})")
    else:
        no_improve += 1
        print(f"    --> No improvement ({no_improve}/{patience})")
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    training_log.append({
        "epoch"        : epoch,
        "train_loss"   : round(epoch_loss, 4),
        "train_acc"    : round(epoch_train_acc, 4),
        "train_f1"     : round(epoch_train_f1, 4),
        "val_acc"      : round(val_metrics["accuracy"], 4),
        "val_f1_macro" : round(val_metrics["f1_macro"], 4),
        "val_f1_fake"  : round(val_metrics["f1_fake"], 4),
        "val_f1_real"  : round(val_metrics["f1_real"], 4),
        "epoch_time_s" : round(epoch_time, 1),
    })

total_time = time.time() - t_start
print(f"\nTotal training time : {total_time/60:.1f} minutes")
print(f"Best epoch          : {best_epoch}  (val F1={best_val_f1:.4f})")

# %% [7] Save training log
log_df = pd.DataFrame(training_log)
log_df.to_csv(RESULTS_PATH / "training_log.csv", index=False)
print(f"\nSaved : outputs/results/training_log.csv")
print(log_df.to_string(index=False))

# %% [8] Final status
print("\n" + "=" * 60)
print("  Stage 5 COMPLETE")
print("=" * 60)
print(f"Best model : {BERT_DIR}")
print(f"Val F1     : {best_val_f1:.4f}")
print(f"Baseline   : 0.9656")
print(f"Gain       : +{(best_val_f1 - 0.9656)*100:.2f} pp")
print("\nNext step  : python scripts/06_model_evaluation.py")
print("=" * 60)

