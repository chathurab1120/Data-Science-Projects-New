# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
scripts/02_eda.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Purpose : Exploratory data analysis on WELFake dataset.
          Produces 6 publication-quality charts for the
          Streamlit dashboard and portfolio README.
Inputs  : data/WELFake_Dataset.csv
Outputs : outputs/charts/01_class_distribution.png
          outputs/charts/02_text_length_distribution.png
          outputs/charts/03_text_length_by_class.png
          outputs/charts/04_title_wordcloud_fake.png
          outputs/charts/05_title_wordcloud_real.png
          outputs/charts/06_top_words_comparison.png
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# %% [0] Imports and configuration
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required for server/cloud
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import yaml
import warnings
warnings.filterwarnings("ignore")

_SCRIPT_DIR  = Path(__file__).parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_CONFIG_PATH = _PROJECT_DIR / "configs" / "config.yaml"

with open(_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DATA_PATH    = _PROJECT_DIR / config["paths"]["data_raw"]
CHARTS_PATH  = _PROJECT_DIR / config["paths"]["outputs_charts"]
CHARTS_PATH.mkdir(parents=True, exist_ok=True)

# Chart style — consistent across all portfolio projects
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE     = {"Real": "#2ecc71", "Fake": "#e74c3c"}
FIG_DPI     = 150
TITLE_SIZE  = 14
LABEL_SIZE  = 11

print("=" * 60)
print("  Stage 2 -- Exploratory Data Analysis")
print("=" * 60)


# %% [1] Load data
print("\nLoading dataset ...")
df = pd.read_csv(DATA_PATH)
df["title"]    = df["title"].fillna("")
df["text"]     = df["text"].fillna("")
df["label_name"] = df["label"].map({0: "Real", 1: "Fake"})

# Compute text lengths for visualisation
df["title_len"]    = df["title"].str.len()
df["text_len"]     = df["text"].str.len()
df["combined_len"] = df["title_len"] + df["text_len"]

print(f"Loaded {len(df):,} articles  |  "
      f"Real: {(df['label']==0).sum():,}  |  "
      f"Fake: {(df['label']==1).sum():,}")


# %% [2] Chart 1 — Class distribution bar chart
# This is the first chart any reader looks at — make it clear and clean
print("\nGenerating chart 1: class distribution ...")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("WELFake Dataset — Class Distribution", fontsize=TITLE_SIZE + 1,
             fontweight="bold", y=1.01)

counts    = df["label_name"].value_counts()
colors    = [PALETTE[c] for c in counts.index]

# Left: bar chart with counts
bars = axes[0].bar(counts.index, counts.values, color=colors,
                   edgecolor="white", linewidth=1.5, width=0.5)
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                 f"{val:,}", ha="center", va="bottom", fontsize=LABEL_SIZE,
                 fontweight="bold")
axes[0].set_title("Sample Count per Class", fontsize=TITLE_SIZE)
axes[0].set_ylabel("Number of Articles", fontsize=LABEL_SIZE)
axes[0].set_ylim(0, counts.max() * 1.15)
axes[0].tick_params(axis="x", labelsize=LABEL_SIZE)

# Right: pie chart with percentages
pct       = counts / counts.sum() * 100
pie_colors = [PALETTE[c] for c in counts.index]
wedges, texts, autotexts = axes[1].pie(
    counts.values,
    labels=counts.index,
    colors=pie_colors,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": LABEL_SIZE},
)
for at in autotexts:
    at.set_fontsize(LABEL_SIZE)
    at.set_fontweight("bold")
axes[1].set_title("Class Split (%)", fontsize=TITLE_SIZE)

plt.tight_layout()
out_path = CHARTS_PATH / "01_class_distribution.png"
plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print(f"  Saved: {out_path.name}")


# %% [3] Chart 2 — Text length distribution (combined title + text)
# Shows where BERT truncation will occur — justifies our max_seq_len=256 choice
print("Generating chart 2: text length distribution ...")

fig, ax = plt.subplots(figsize=(11, 5))
ax.set_title("Combined Text Length Distribution (Title + Body)",
             fontsize=TITLE_SIZE, fontweight="bold")

# Cap at p99 for readability — extreme outliers distort the histogram
p99 = df["combined_len"].quantile(0.99)
df_plot = df[df["combined_len"] <= p99]

for label_name, color in PALETTE.items():
    subset = df_plot[df_plot["label_name"] == label_name]["combined_len"]
    ax.hist(subset, bins=80, alpha=0.65, color=color,
            label=f"{label_name} (n={len(subset):,})", edgecolor="none")

# Add vertical line at approximate BERT token boundary
# 256 tokens ~ 1,280 characters (avg 5 chars/token)
bert_approx_chars = 256 * 5
ax.axvline(bert_approx_chars, color="#2c3e50", linestyle="--", linewidth=1.8,
           label=f"BERT max_seq_len=256 (~{bert_approx_chars:,} chars)")

ax.set_xlabel("Character Count (Title + Body)", fontsize=LABEL_SIZE)
ax.set_ylabel("Number of Articles", fontsize=LABEL_SIZE)
ax.legend(fontsize=LABEL_SIZE - 1)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
out_path = CHARTS_PATH / "02_text_length_distribution.png"
plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print(f"  Saved: {out_path.name}")


# %% [4] Chart 3 — Text length by class (box + violin)
# Checks whether Real and Fake articles differ in verbosity
print("Generating chart 3: text length by class ...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Text Length by Class", fontsize=TITLE_SIZE + 1,
             fontweight="bold", y=1.01)

# Cap at p95 to reduce outlier distortion
p95 = df["combined_len"].quantile(0.95)
df_box = df[df["combined_len"] <= p95].copy()

# Left: violin plot
sns.violinplot(
    data=df_box, x="label_name", y="combined_len",
    palette=PALETTE, inner="quartile",
    order=["Real", "Fake"], ax=axes[0]
)
axes[0].set_title("Violin Plot (capped at p95)", fontsize=TITLE_SIZE)
axes[0].set_xlabel("Class", fontsize=LABEL_SIZE)
axes[0].set_ylabel("Character Count", fontsize=LABEL_SIZE)
axes[0].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

# Right: box plot
sns.boxplot(
    data=df_box, x="label_name", y="combined_len",
    palette=PALETTE, order=["Real", "Fake"],
    width=0.5, ax=axes[1]
)
axes[1].set_title("Box Plot (capped at p95)", fontsize=TITLE_SIZE)
axes[1].set_xlabel("Class", fontsize=LABEL_SIZE)
axes[1].set_ylabel("Character Count", fontsize=LABEL_SIZE)
axes[1].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
out_path = CHARTS_PATH / "03_text_length_by_class.png"
plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print(f"  Saved: {out_path.name}")


# %% [5] Helper — extract top words from a text series
# Stop words manually defined to avoid nltk dependency
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "that", "this", "it", "its", "as", "not", "he",
    "she", "they", "we", "you", "i", "his", "her", "their", "our", "said",
    "will", "would", "could", "should", "may", "also", "more", "than",
    "about", "up", "out", "into", "than", "so", "if", "which", "who",
    "after", "before", "when", "there", "s", "t", "do", "did", "does",
    "been", "all", "new", "one", "two", "three", "us", "no", "mr", "ms",
    "trump", "clinton",  # too dominant — mask to reveal other patterns
}

def get_top_words(series: pd.Series, n: int = 30) -> list:
    """Return top n words from a Series of text strings, excluding stop words."""
    all_text = " ".join(series.dropna().str.lower().tolist())
    words    = re.findall(r"\b[a-z]{3,}\b", all_text)
    filtered = [w for w in words if w not in STOP_WORDS]
    return Counter(filtered).most_common(n)


# %% [6] Charts 4 & 5 — Word clouds for Fake vs Real titles
print("Generating chart 4: word cloud (Fake titles) ...")

for label_val, label_name, chart_num in [(1, "Fake", "04"), (0, "Real", "05")]:
    subset_text = df[df["label"] == label_val]["title"]
    text_blob   = " ".join(subset_text.dropna().str.lower().tolist())
    # Remove stop words from cloud
    text_blob   = " ".join([w for w in text_blob.split()
                            if w not in STOP_WORDS and len(w) > 2])

    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        colormap="Reds" if label_name == "Fake" else "Greens",
        max_words=150,
        collocations=False,
        prefer_horizontal=0.85,
    ).generate(text_blob)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Most Frequent Title Words — {label_name} News",
                 fontsize=TITLE_SIZE + 1, fontweight="bold", pad=12)

    out_path = CHARTS_PATH / f"{chart_num}_title_wordcloud_{label_name.lower()}.png"
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# %% [7] Chart 6 — Top words side-by-side comparison
# Visualises which words are most predictive — sets up the BERT story
print("Generating chart 6: top words comparison ...")

fake_words = get_top_words(df[df["label"] == 1]["title"], n=20)
real_words = get_top_words(df[df["label"] == 0]["title"], n=20)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("Top 20 Title Words by Class (stop words removed)",
             fontsize=TITLE_SIZE + 1, fontweight="bold", y=1.01)

for ax, word_counts, label_name, color in [
    (axes[0], fake_words, "Fake News", "#e74c3c"),
    (axes[1], real_words, "Real News", "#2ecc71"),
]:
    words  = [w for w, _ in word_counts][::-1]
    counts = [c for _, c in word_counts][::-1]
    bars   = ax.barh(words, counts, color=color, alpha=0.85, edgecolor="none")
    ax.set_title(label_name, fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Frequency", fontsize=LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=9)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(0, max(counts) * 1.18)

plt.tight_layout()
out_path = CHARTS_PATH / "06_top_words_comparison.png"
plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print(f"  Saved: {out_path.name}")


# %% [8] Print EDA summary statistics
print("\n--- EDA Summary Statistics ---")
for label_val, label_name in [(0, "Real"), (1, "Fake")]:
    subset = df[df["label"] == label_val]
    print(f"\n  {label_name} News ({len(subset):,} articles):")
    print(f"    Median combined length : {subset['combined_len'].median():>8,.0f} chars")
    print(f"    Mean combined length   : {subset['combined_len'].mean():>8,.0f} chars")
    print(f"    Empty titles           : {(subset['title_len'] == 0).sum():>8,}")
    print(f"    Empty text bodies      : {(subset['text_len'] == 0).sum():>8,}")


# %% [9] Final status
print("\n" + "=" * 60)
print("  Stage 2 COMPLETE")
print("=" * 60)
charts = sorted(CHARTS_PATH.glob("0*.png"))
print(f"Charts saved ({len(charts)} files):")
for c in charts:
    print(f"  {c.name}")
print("\nNext step : python scripts/03_preprocessing.py")
print("=" * 60)

