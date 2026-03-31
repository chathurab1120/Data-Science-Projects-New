# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
src/utils.py
Shared text cleaning and helper utilities used across pipeline scripts.
Inputs  : raw text strings
Outputs : cleaned text strings, logging helpers
"""

# %% [0] Imports
import re
import unicodedata
from pathlib import Path


# %% [1] Text cleaning helpers

def clean_text(text: str) -> str:
    """
    Clean a raw news article string for NLP modelling.
    Steps: lowercase, remove HTML tags, remove URLs, strip non-ASCII,
    collapse whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Normalise unicode accents -> ASCII equivalents
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove special characters — keep letters, digits, basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\-\'\"]+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_title_text(title: str, text: str) -> str:
    """
    Combine title and body text with a separator.
    BERT benefits from seeing the title — it is often the most
    discriminative signal for fake news classification.
    """
    title_clean = clean_text(str(title))
    text_clean  = clean_text(str(text))
    if title_clean and text_clean:
        return f"{title_clean} [SEP] {text_clean}"
    return title_clean or text_clean


def print_section(title: str) -> None:
    """Print a visible section header to the console."""
    border = "=" * 60
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")


