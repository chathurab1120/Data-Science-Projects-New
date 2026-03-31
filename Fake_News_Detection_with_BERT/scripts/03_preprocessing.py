# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
scripts/03_preprocessing.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Purpose : Clean text, remove duplicates and noise, combine
          title + body, and produce stratified train/val/test
          splits with zero data leakage.
Inputs  : data/WELFake_Dataset.csv
Outputs : outputs/results/train.csv
          outputs/results/val.csv
          outputs/results/test.csv
          outputs/results/preprocessing_report.csv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# %% [0] Imports and configuration
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import sys
from sklearn.model_selection import train_test_split

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
TRAIN_SIZE    = config["data"]["train_size"]
VAL_SIZE      = config["data"]["val_size"]
TEST_SIZE     = config["data"]["test_size"]
MIN_TEXT_LEN  = config["data"]["min_text_length"]

# Import shared utilities from src/
sys.path.insert(0, str(_PROJECT_DIR))
from src.utils import clean_text, combine_title_text, print_section

print("=" * 60)
print("  Stage 3 -- Text Preprocessing and Data Splitting")
print("=" * 60)


# %% [1] Load raw data
print_section("Step 1: Load Raw Data")
df = pd.read_csv(DATA_PATH)
print(f"Raw shape : {df.shape}")
n_raw = len(df)


# %% [2] Drop rows with null labels
# Label nulls cannot be imputed — remove them first
print_section("Step 2: Drop Null Labels")
df = df.dropna(subset=[TARGET_COL])
n_after_label = len(df)
print(f"Rows removed (null label) : {n_raw - n_after_label}")
print(f"Remaining rows            : {n_after_label:,}")


# %% [3] Fill null titles and text with empty string
# We do NOT drop null title/text rows — combine_title_text handles gracefully
# by using whichever field is available. Dropping would discard valid articles.
print_section("Step 3: Handle Null Title / Text")
df["title"] = df["title"].fillna("")
df["text"]  = df["text"].fillna("")
print(f"Null titles filled with empty string : {(df['title'] == '').sum()}")
print(f"Null texts  filled with empty string : {(df['text']  == '').sum()}")


# %% [4] Combine title + text into a single input field
# BERT sees one string per article: "title [SEP] body"
# The title carries the most discriminative signal for fake news detection
# — including it consistently outperforms body-only models by ~1-2% F1
print_section("Step 4: Combine Title + Body Text")
df["combined_text"] = df.apply(
    lambda row: combine_title_text(row["title"], row["text"]), axis=1
)
print(f"Sample combined text (first article):")
print(f"  {df['combined_text'].iloc[0][:200]} ...")


# %% [5] Remove short articles
# Articles under MIN_TEXT_LEN characters are likely corrupt rows
# (e.g. single words, encoding errors, placeholder text)
print_section("Step 5: Remove Short Articles")
mask_short = df["combined_text"].str.len() < MIN_TEXT_LEN
n_short    = mask_short.sum()
df         = df[~mask_short].copy()
print(f"Removed (< {MIN_TEXT_LEN} chars) : {n_short}")
print(f"Remaining rows             : {len(df):,}")


# %% [6] Remove duplicate combined texts
# CRITICAL: deduplication must happen BEFORE splitting to prevent the same
# article appearing in both train and test (data leakage).
# We deduplicate on combined_text — identical bodies with different titles
# are kept, but exact duplicates are removed.
print_section("Step 6: Remove Duplicate Articles")
n_before_dedup = len(df)
df = df.drop_duplicates(subset=["combined_text"], keep="first").copy()
n_removed_dedup = n_before_dedup - len(df)
print(f"Duplicate combined texts removed : {n_removed_dedup:,}")
print(f"Remaining rows                   : {len(df):,}")


# %% [7] Reset index and keep only needed columns
df = df[["combined_text", TARGET_COL]].reset_index(drop=True)
df.columns = ["text", "label"]
print(f"\nFinal cleaned dataframe shape : {df.shape}")
print(f"Label distribution after cleaning:")
label_counts = df["label"].value_counts().sort_index()
for lbl, cnt in label_counts.items():
    name = "Real" if lbl == 0 else "Fake"
    pct  = cnt / len(df) * 100
    print(f"  {lbl} ({name}) : {cnt:,}  ({pct:.1f}%)")


# %% [8] Stratified train / val / test split
# Stratified split ensures class ratio is preserved in all three sets.
# We split in two steps:
#   Step A : 80% train | 20% temp
#   Step B : temp -> 50% val | 50% test  (giving 10% val, 10% test overall)
# This approach prevents any leakage between splits.
print_section("Step 7: Stratified Train / Val / Test Split")

X = df["text"]
y = df["label"]

# Step A — train vs temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size    = 1.0 - TRAIN_SIZE,   # 0.20
    stratify     = y,
    random_state = RANDOM_STATE,
)

# Step B — val vs test (50/50 split of temp = 10% each overall)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size    = TEST_SIZE / (VAL_SIZE + TEST_SIZE),  # 0.50
    stratify     = y_temp,
    random_state = RANDOM_STATE,
)

train_df = pd.DataFrame({"text": X_train, "label": y_train}).reset_index(drop=True)
val_df   = pd.DataFrame({"text": X_val,   "label": y_val  }).reset_index(drop=True)
test_df  = pd.DataFrame({"text": X_test,  "label": y_test }).reset_index(drop=True)

# Verify split sizes and class ratios
print(f"{'Split':<10} {'Rows':>8}  {'Real':>8}  {'Fake':>8}  {'Real%':>7}  {'Fake%':>7}")
print("-" * 58)
for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    real_n = (split["label"] == 0).sum()
    fake_n = (split["label"] == 1).sum()
    real_p = real_n / len(split) * 100
    fake_p = fake_n / len(split) * 100
    print(f"{name:<10} {len(split):>8,}  {real_n:>8,}  {fake_n:>8,}  "
          f"{real_p:>6.1f}%  {fake_p:>6.1f}%")


# %% [9] Save splits to outputs/results/
print_section("Step 8: Save Split CSVs")
train_df.to_csv(RESULTS_PATH / "train.csv", index=False)
val_df.to_csv(  RESULTS_PATH / "val.csv",   index=False)
test_df.to_csv( RESULTS_PATH / "test.csv",  index=False)
print(f"Saved : outputs/results/train.csv   ({len(train_df):,} rows)")
print(f"Saved : outputs/results/val.csv     ({len(val_df):,} rows)")
print(f"Saved : outputs/results/test.csv    ({len(test_df):,} rows)")


# %% [10] Save preprocessing report
report_rows = [
    {"step": "raw_rows",               "value": n_raw},
    {"step": "after_drop_null_label",  "value": n_after_label},
    {"step": "after_remove_short",     "value": n_before_dedup},
    {"step": "after_dedup",            "value": len(df)},
    {"step": "train_rows",             "value": len(train_df)},
    {"step": "val_rows",               "value": len(val_df)},
    {"step": "test_rows",              "value": len(test_df)},
    {"step": "train_real",             "value": int((train_df["label"]==0).sum())},
    {"step": "train_fake",             "value": int((train_df["label"]==1).sum())},
    {"step": "val_real",               "value": int((val_df["label"]==0).sum())},
    {"step": "val_fake",               "value": int((val_df["label"]==1).sum())},
    {"step": "test_real",              "value": int((test_df["label"]==0).sum())},
    {"step": "test_fake",              "value": int((test_df["label"]==1).sum())},
]
report_df = pd.DataFrame(report_rows)
report_df.to_csv(RESULTS_PATH / "preprocessing_report.csv", index=False)
print(f"Saved : outputs/results/preprocessing_report.csv")


# %% [11] Final status
print("\n" + "=" * 60)
print("  Stage 3 COMPLETE")
print("=" * 60)
print(f"Clean dataset    : {len(df):,} articles")
print(f"Train            : {len(train_df):,}")
print(f"Val              : {len(val_df):,}")
print(f"Test             : {len(test_df):,}")
print("\nNext step        : python scripts/04_baseline_model.py")
print("=" * 60)

