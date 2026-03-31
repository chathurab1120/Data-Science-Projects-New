# -*- coding: utf-8 -*-
"""
src/trainer.py
GPU-accelerated training loop with FP16 mixed precision,
early stopping, and best-checkpoint saving by validation F1.
Inputs  : model, dataloaders, optimizer, scheduler, config
Outputs : trained model saved to outputs/models/bert_fake_news/
"""

# %% [0] Imports
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from torch.cuda.amp import GradScaler, autocast


# %% [1] Training utilities

def get_device() -> torch.device:
    """
    Auto-detect the best available device.
    Prints a clear message so the user knows GPU is active.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected : {gpu_name}")
        print(f"VRAM         : {vram_gb:.1f} GB")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected — running on CPU. Training will be slow.")
    return device


def evaluate_model(model, dataloader, device) -> dict:
    """
    Run model inference on a dataloader and return metrics dict.
    Always uses torch.no_grad() and model.eval() for correctness.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "accuracy" : accuracy_score(all_labels, all_preds),
        "f1_macro" : f1_score(all_labels, all_preds, average="macro"),
        "f1_fake"  : f1_score(all_labels, all_preds, pos_label=1, average="binary"),
        "f1_real"  : f1_score(all_labels, all_preds, pos_label=0, average="binary"),
    }


def print_gpu_memory() -> None:
    """Log current GPU VRAM usage — useful for monitoring batch size headroom."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved  = torch.cuda.memory_reserved(0) / 1e9
        print(f"  GPU memory -> allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")


