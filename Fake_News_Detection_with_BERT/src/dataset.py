# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
src/dataset.py
PyTorch Dataset class for BERT tokenisation and batching.
Inputs  : list of text strings, list of labels, HuggingFace tokenizer, max_len
Outputs : tokenised tensors ready for DataLoader
"""

# %% [0] Imports
import torch
from torch.utils.data import Dataset


# %% [1] Dataset class

class FakeNewsDataset(Dataset):
    """
    Wraps a list of cleaned article strings and binary labels into
    a format compatible with PyTorch DataLoader + BERT tokenizer.

    Each __getitem__ call returns a dict of tensors:
        input_ids      : token IDs padded/truncated to max_len
        attention_mask : 1 for real tokens, 0 for padding
        label          : 0 (real) or 1 (fake)
    """

    def __init__(self, texts: list, labels: list, tokenizer, max_len: int):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length      = self.max_len,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )
        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),
            "attention_mask" : encoding["attention_mask"].squeeze(0),
            "label"          : torch.tensor(self.labels[idx], dtype=torch.long),
        }


