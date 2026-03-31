# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
scripts/01_data_loading.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Purpose : Load the WELFake dataset, validate schema and quality,
          and produce a data summary report.
Inputs  : data/WELFake_Dataset.csv
Outputs : outputs/results/data_summary.csv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# %% [0] Imports and configuration
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# Resolve paths relative to this script — works locally and on Streamlit Cloud
_SCRIPT_DIR  = Path(__file__).parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_CONFIG_PATH = _PROJECT_DIR / "configs" / "config.yaml"

with open(_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DATA_PATH    = _PROJECT_DIR / config["paths"]["data_raw"]
RESULTS_PATH = _PROJECT_DIR / config["paths"]["outputs_results"]
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_STATE  = config["project"]["random_state"]
TARGET_COL    = config["data"]["target_column"]
TEXT_COLS     = config["data"]["text_columns"]
MIN_TEXT_LEN  = config["data"]["min_text_length"]

print("=" * 60)
print("  Stage 1 — Data Loading and Validation")
print("=" * 60)
print(f"Dataset path : {DATA_PATH}")
print(f"Config loaded: {_CONFIG_PATH.name}\n")


# %% [1] Load raw CSV
# WELFake has 4 columns: serial number (unnamed index), title, text, label
# label: 0 = Real news, 1 = Fake news
print("Loading WELFake_Dataset.csv ...")
df = pd.read_csv(DATA_PATH)
print(f"Raw shape    : {df.shape}")
print(f"Columns      : {df.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())


# %% [2] Schema validation
# Confirm expected columns exist before any downstream processing
print("\n--- Schema Validation ---")
expected_cols = ["title", "text", "label"]
missing_cols  = [c for c in expected_cols if c not in df.columns]

if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")
else:
    print("All expected columns present: title, text, label")

# Confirm binary label values
label_values = sorted(df[TARGET_COL].dropna().unique().tolist())
print(f"Label values found : {label_values}")
if set(label_values) != {0, 1}:
    raise ValueError(f"Expected labels {{0, 1}}, got {label_values}")
print("Label values confirmed: 0=Real, 1=Fake")


# %% [3] Class distribution
# Understanding class balance is critical — imbalanced classes need
# different strategies (we use class_weight in baseline, balanced batching in BERT)
print("\n--- Class Distribution ---")
label_counts = df[TARGET_COL].value_counts().sort_index()
label_pct    = df[TARGET_COL].value_counts(normalize=True).sort_index() * 100

for lbl in label_counts.index:
    name = "Real" if lbl == 0 else "Fake"
    print(f"  Label {lbl} ({name}) : {label_counts[lbl]:>6,} samples  ({label_pct[lbl]:.1f}%)")

imbalance_ratio = label_counts.max() / label_counts.min()
print(f"\nImbalance ratio  : {imbalance_ratio:.2f}x")
if imbalance_ratio < 1.5:
    print("Class balance    : GOOD (ratio < 1.5x, no special handling needed)")
else:
    print("Class balance    : MODERATE — will use class_weight='balanced' in baseline")


# %% [4] Missing value analysis
print("\n--- Missing Value Analysis ---")
null_counts = df[expected_cols].isnull().sum()
null_pct    = (null_counts / len(df) * 100).round(2)

for col in expected_cols:
    print(f"  {col:<10} : {null_counts[col]:>5} nulls  ({null_pct[col]:.2f}%)")

total_nulls = null_counts.sum()
print(f"\nTotal nulls  : {total_nulls}")


# %% [5] Text length analysis
# BERT has a max sequence length of 512 tokens — understanding the raw
# character length distribution helps us choose max_seq_len (we use 256)
print("\n--- Text Length Analysis (characters) ---")

# Fill nulls temporarily for length calculation only
df["title_len"] = df["title"].fillna("").str.len()
df["text_len"]  = df["text"].fillna("").str.len()
df["combined_len"] = df["title_len"] + df["text_len"]

for col in ["title_len", "text_len", "combined_len"]:
    series = df[col]
    print(f"\n  {col}:")
    print(f"    min    : {series.min():>8,.0f}")
    print(f"    median : {series.median():>8,.0f}")
    print(f"    mean   : {series.mean():>8,.0f}")
    print(f"    p95    : {series.quantile(0.95):>8,.0f}")
    print(f"    max    : {series.max():>8,.0f}")

# Articles shorter than MIN_TEXT_LEN characters are likely corrupt rows
short_articles = (df["combined_len"] < MIN_TEXT_LEN).sum()
print(f"\nArticles shorter than {MIN_TEXT_LEN} chars : {short_articles}")


# %% [6] Duplicate detection
print("\n--- Duplicate Detection ---")
dup_full  = df.duplicated().sum()
dup_title = df["title"].dropna().duplicated().sum()
dup_text  = df["text"].dropna().duplicated().sum()

print(f"  Full row duplicates   : {dup_full}")
print(f"  Duplicate titles      : {dup_title}")
print(f"  Duplicate text bodies : {dup_text}")


# %% [7] Sample articles — qualitative inspection
# Always read a few examples to understand what the model will actually see
print("\n--- Sample Articles (first 2 per class) ---")
for label_val, label_name in [(0, "REAL"), (1, "FAKE")]:
    print(f"\n  [{label_name}]")
    samples = df[df[TARGET_COL] == label_val].head(2)
    for _, row in samples.iterrows():
        title_preview = str(row["title"])[:80] if pd.notna(row["title"]) else "[NO TITLE]"
        text_preview  = str(row["text"])[:120]  if pd.notna(row["text"])  else "[NO TEXT]"
        print(f"    Title : {title_preview}")
        print(f"    Text  : {text_preview}")
        print()


# %% [8] Save data summary to results
# This CSV becomes the source of truth for the Streamlit Overview page
print("\n--- Saving Data Summary ---")

# Drop temp length columns before summarising
df_clean = df.drop(columns=["title_len", "text_len", "combined_len"])

summary_rows = [
    {"metric": "total_samples",         "value": len(df_clean)},
    {"metric": "real_samples",          "value": int(label_counts.get(0, 0))},
    {"metric": "fake_samples",          "value": int(label_counts.get(1, 0))},
    {"metric": "real_pct",              "value": round(float(label_pct.get(0, 0)), 2)},
    {"metric": "fake_pct",              "value": round(float(label_pct.get(1, 0)), 2)},
    {"metric": "null_title",            "value": int(null_counts["title"])},
    {"metric": "null_text",             "value": int(null_counts["text"])},
    {"metric": "null_label",            "value": int(null_counts["label"])},
    {"metric": "duplicate_rows",        "value": int(dup_full)},
    {"metric": "duplicate_titles",      "value": int(dup_title)},
    {"metric": "duplicate_text",        "value": int(dup_text)},
    {"metric": "median_title_len",      "value": round(float(df["title_len"].median()), 1)},
    {"metric": "median_text_len",       "value": round(float(df["text_len"].median()), 1)},
    {"metric": "imbalance_ratio",       "value": round(float(imbalance_ratio), 3)},
    {"metric": "short_articles",        "value": int(short_articles)},
]

summary_df = pd.DataFrame(summary_rows)
summary_path = RESULTS_PATH / "data_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved : {summary_path}")
print(f"\nData summary preview:")
print(summary_df.to_string(index=False))


# %% [9] Final status report
print("\n" + "=" * 60)
print("  Stage 1 COMPLETE")
print("=" * 60)
print(f"Dataset rows     : {len(df_clean):,}")
print(f"Dataset columns  : {df_clean.shape[1]}")
print(f"Ready for EDA    : outputs/results/data_summary.csv")
print("Next step        : python scripts/02_eda.py")
print("=" * 60)

