"""
app/streamlit_app.py
Multi-page Streamlit dashboard for Fake News Detection with BERT.
Pages:
    1. Overview       -- project summary and dataset stats
    2. EDA            -- exploratory analysis charts
    3. Live Detector  -- paste any article, get prediction + LIME
    4. Performance    -- model evaluation charts and comparison
    5. Dataset        -- browse sample articles
"""

from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import torch
import sys

# ── Path resolution ───────────────────────────────────────────
_APP_DIR     = Path(__file__).parent
_PROJECT_DIR = _APP_DIR.parent
_CONFIG_PATH = _PROJECT_DIR / "configs" / "config.yaml"

with open(_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

CHARTS  = _PROJECT_DIR / config["paths"]["outputs_charts"]
RESULTS = _PROJECT_DIR / config["paths"]["outputs_results"]
MODELS  = _PROJECT_DIR / config["paths"]["outputs_models"]
BERT_DIR = MODELS / "bert_fake_news"

sys.path.insert(0, str(_PROJECT_DIR))

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Fake News Detector | BERT",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Sidebar navigation ────────────────────────────────────────
st.sidebar.title("🔍 Fake News Detection")
st.sidebar.markdown("**BERT fine-tuned on WELFake**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 EDA", "🤖 Live Detector",
     "📈 Performance", "🗂 Dataset Explorer"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model**: bert-base-uncased  \n"
    "**Dataset**: WELFake (63,606 articles)  \n"
    "**Test F1**: 0.9908  \n"
    "**Test AUC**: 0.9996"
)


# ── Helper: load image safely ─────────────────────────────────
def load_chart(filename: str):
    path = CHARTS / filename
    if path.exists():
        return Image.open(path)
    return None


# ══════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🔍 Fake News Detection with BERT")
    st.markdown(
        "A production-grade NLP pipeline that fine-tunes **bert-base-uncased** "
        "on the WELFake dataset (72,134 news articles) to classify news as "
        "**Real** or **Fake** with 99.1% accuracy."
    )

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Accuracy",  "99.09%", "+2.50% vs baseline")
    col2.metric("F1 Macro",       "0.9908",  "+0.0252 vs baseline")
    col3.metric("ROC-AUC",        "0.9996",  "+0.0051 vs baseline")
    col4.metric("Training Time",  "17.8 min", "FP16 + CUDA 12.8")

    st.markdown("---")

    # Model comparison table
    st.subheader("Model Comparison")
    try:
        cmp_df = pd.read_csv(RESULTS / "model_comparison.csv")
        st.dataframe(
            cmp_df.style.highlight_max(
                subset=["accuracy","f1_macro","f1_fake","f1_real","roc_auc"],
                color="#d4edda"
            ),
            use_container_width=True,
        )
    except Exception:
        st.info("Run scripts/06_model_evaluation.py to generate comparison data.")

    st.markdown("---")

    # Pipeline stages
    st.subheader("Pipeline")
    stages = [
        ("01", "Data Loading",     "72,134 WELFake articles validated"),
        ("02", "EDA",              "6 exploratory charts generated"),
        ("03", "Preprocessing",    "63,606 clean articles, 80/10/10 split"),
        ("04", "Baseline Model",   "TF-IDF + LR — 96.56% F1 macro"),
        ("05", "BERT Fine-Tuning", "bert-base-uncased, 4 epochs, FP16, RTX 5080"),
        ("06", "Evaluation",       "Test F1 = 0.9908, AUC = 0.9996"),
        ("07", "Explainability",   "LIME token-level explanations"),
    ]
    for num, name, detail in stages:
        st.markdown(f"**{num}. {name}** — {detail}")

    st.markdown("---")
    st.subheader("Tech Stack")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Model**  \nbert-base-uncased  \nHuggingFace Transformers")
    c2.markdown("**Training**  \nPyTorch + CUDA 12.8  \nFP16 Mixed Precision")
    c3.markdown("**Explainability**  \nLIME token importance  \n12 sample explanations")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Key insights from the WELFake dataset before modelling.")

    tab1, tab2, tab3 = st.tabs(
        ["Class Distribution", "Text Length", "Word Analysis"])

    with tab1:
        img = load_chart("01_class_distribution.png")
        if img:
            st.image(img, use_container_width=True)
        st.markdown(
            "The dataset is **near-perfectly balanced** (54.7% Real / 45.3% Fake) "
            "with an imbalance ratio of 1.06x — no oversampling required."
        )

    with tab2:
        img2 = load_chart("02_text_length_distribution.png")
        if img2:
            st.image(img2, use_container_width=True)
        img3 = load_chart("03_text_length_by_class.png")
        if img3:
            st.image(img3, use_container_width=True)
        st.markdown(
            "Median article length is ~2,500 characters. The BERT truncation "
            "boundary at max_seq_len=256 tokens (~1,280 chars) captures the "
            "most discriminative content for most articles."
        )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            img4 = load_chart("04_title_wordcloud_fake.png")
            if img4:
                st.image(img4, caption="Fake News — Top Title Words",
                         use_container_width=True)
        with col2:
            img5 = load_chart("05_title_wordcloud_real.png")
            if img5:
                st.image(img5, caption="Real News — Top Title Words",
                         use_container_width=True)
        img6 = load_chart("06_top_words_comparison.png")
        if img6:
            st.image(img6, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — Live Detector
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Live Detector":
    st.title("🤖 Live Fake News Detector")
    st.markdown(
        "Paste any news article (title + body) and the fine-tuned BERT model "
        "will classify it as **Real** or **Fake** with a confidence score."
    )

    # Load model (cached so it only loads once per session)
    @st.cache_resource(show_spinner="Loading BERT model...")
    def load_model():
        from transformers import BertTokenizerFast, BertForSequenceClassification
        HF_MODEL_ID = "chathurab1120/bert-fake-news-detector"
        # Try local model first (works locally), fall back to HF Hub (Streamlit Cloud)
        if BERT_DIR.exists() and any(BERT_DIR.iterdir()):
            model_source = str(BERT_DIR)
        else:
            model_source = HF_MODEL_ID
        tokenizer = BertTokenizerFast.from_pretrained(model_source)
        model     = BertForSequenceClassification.from_pretrained(model_source)
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model     = model.to(device)
        model.eval()
        return tokenizer, model, device

    try:
        tokenizer, model, device = load_model()
        st.success(f"Model loaded on: {device}")
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    # Input
    sample_texts = {
        "-- Paste your own article --": "",
        "Sample Fake Article": (
            "BREAKING: Scientists confirm that the moon landing was staged "
            "in a Hollywood studio. Leaked NASA documents reveal the shocking "
            "truth that has been hidden from the public for decades. "
            "Share this before it gets deleted!"
        ),
        "Sample Real Article": (
            "Reuters - The Federal Reserve held interest rates steady on "
            "Wednesday, as policymakers await further evidence that inflation "
            "is cooling before considering cuts to borrowing costs. The "
            "decision was unanimous among voting members of the FOMC."
        ),
    }

    selected = st.selectbox("Choose a sample or paste your own:", list(sample_texts.keys()))
    default_text = sample_texts[selected]

    article_text = st.text_area(
        "Article text (title + body):",
        value=default_text,
        height=200,
        placeholder="Paste your news article here...",
    )

    if st.button("🔍 Analyse Article", type="primary"):
        if not article_text.strip():
            st.warning("Please enter some text to analyse.")
        else:
            with st.spinner("Running BERT inference..."):
                from src.utils import clean_text
                cleaned = clean_text(article_text)
                encoding = tokenizer(
                    cleaned,
                    max_length     = config["model"]["max_seq_len"],
                    padding        = "max_length",
                    truncation     = True,
                    return_tensors = "pt",
                )
                input_ids      = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask)
                    proba   = torch.softmax(outputs.logits, dim=1)[0]
                    pred    = torch.argmax(proba).item()

                real_prob = float(proba[0])
                fake_prob = float(proba[1])

            # Display result
            st.markdown("---")
            if pred == 1:
                st.error(f"## 🚨 FAKE NEWS  ({fake_prob:.1%} confidence)")
            else:
                st.success(f"## ✅ REAL NEWS  ({real_prob:.1%} confidence)")

            col1, col2 = st.columns(2)
            col1.metric("Real probability", f"{real_prob:.4f}")
            col2.metric("Fake probability", f"{fake_prob:.4f}")

            # Confidence bar
            st.markdown("**Confidence breakdown:**")
            st.progress(fake_prob, text=f"Fake: {fake_prob:.1%}")

            # Token count info
            tokens = tokenizer.tokenize(cleaned)
            st.caption(
                f"Article length: {len(article_text)} chars | "
                f"{len(tokens)} tokens | "
                f"Truncated to {config['model']['max_seq_len']} tokens for BERT"
            )

    st.markdown("---")
    st.subheader("Sample LIME Explanations")
    st.markdown(
        "These charts show which words most influenced BERT's prediction "
        "on held-out test articles. Red bars = evidence for Fake, "
        "Green bars = evidence for Real."
    )

    lime_tab1, lime_tab2 = st.tabs(["Fake News Explanations", "Real News Explanations"])
    with lime_tab1:
        cols = st.columns(2)
        for i in range(1, 7):
            img = load_chart(f"lime_fake_{i:02d}.png")
            if img:
                cols[(i-1) % 2].image(img, caption=f"Fake Sample {i}",
                                       use_container_width=True)
    with lime_tab2:
        cols = st.columns(2)
        for i in range(1, 7):
            img = load_chart(f"lime_real_{i:02d}.png")
            if img:
                cols[(i-1) % 2].image(img, caption=f"Real Sample {i}",
                                       use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — Performance
# ══════════════════════════════════════════════════════════════
elif page == "📈 Performance":
    st.title("📈 Model Performance")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Confusion Matrices", "ROC Curves",
         "Model Comparison", "Training Curves"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            img = load_chart("07_baseline_confusion_matrix.png")
            if img:
                st.image(img, caption="Baseline: TF-IDF + LR",
                         use_container_width=True)
        with col2:
            img = load_chart("09_bert_confusion_matrix.png")
            if img:
                st.image(img, caption="BERT fine-tuned",
                         use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            img = load_chart("08_baseline_roc_curve.png")
            if img:
                st.image(img, caption="Baseline ROC",
                         use_container_width=True)
        with col2:
            img = load_chart("10_bert_roc_curve.png")
            if img:
                st.image(img, caption="BERT ROC",
                         use_container_width=True)

    with tab3:
        img = load_chart("11_model_comparison.png")
        if img:
            st.image(img, use_container_width=True)
        try:
            cmp_df = pd.read_csv(RESULTS / "model_comparison.csv")
            st.dataframe(cmp_df, use_container_width=True)
        except Exception:
            pass

    with tab4:
        img = load_chart("12_training_curves.png")
        if img:
            st.image(img, use_container_width=True)
        try:
            log_df = pd.read_csv(RESULTS / "training_log.csv")
            st.dataframe(log_df, use_container_width=True)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
# PAGE 5 — Dataset Explorer
# ══════════════════════════════════════════════════════════════
elif page == "🗂 Dataset Explorer":
    st.title("🗂 Dataset Explorer")
    st.markdown("Browse the preprocessed WELFake articles used for training.")

    try:
        @st.cache_data
        def load_splits():
            splits = {}
            for name in ["train", "val", "test"]:
                path = RESULTS / f"{name}.csv"
                if path.exists():
                    splits[name] = pd.read_csv(path)
                else:
                    # On Streamlit Cloud the large CSVs are not available
                    # Show a small sample from the results we do have
                    splits[name] = pd.DataFrame({
                        "text": ["Dataset splits not available on Streamlit Cloud. "
                                 "Run scripts/03_preprocessing.py locally to generate them."],
                        "label": [0]
                    })
            return splits["train"], splits["val"], splits["test"]

        train_df, val_df, test_df = load_splits()

        split = st.selectbox("Select split:", ["Train", "Val", "Test"])
        df    = {"Train": train_df, "Val": val_df, "Test": test_df}[split]

        label_filter = st.radio("Filter by label:", ["All", "Real (0)", "Fake (1)"],
                                horizontal=True)
        if label_filter == "Real (0)":
            df = df[df["label"] == 0]
        elif label_filter == "Fake (1)":
            df = df[df["label"] == 1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total articles", f"{len(df):,}")
        col2.metric("Real", f"{(df['label']==0).sum():,}")
        col3.metric("Fake", f"{(df['label']==1).sum():,}")

        st.markdown("---")
        n_show = st.slider("Articles to display:", 5, 50, 10)
        sample = df.sample(n=min(n_show, len(df)),
                           random_state=42).reset_index(drop=True)

        for _, row in sample.iterrows():
            label_badge = "🚨 FAKE" if row["label"] == 1 else "✅ REAL"
            with st.expander(f"{label_badge} — {row['text'][:80]}..."):
                st.write(row["text"])

    except Exception as e:
        st.error(f"Could not load dataset splits: {e}")
        st.info("Run scripts/03_preprocessing.py to generate split files.")
