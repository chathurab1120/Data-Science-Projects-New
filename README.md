# Fake News Detection with BERT

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fake-news-bert.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/chathurab1120/Data-Science-Projects-New/tree/main/Fake_News_Detection_with_BERT)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Demo-yellow?logo=huggingface)](https://huggingface.co/chathurab1120/bert-fake-news-detector)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-red?logo=pytorch)](https://pytorch.org)

> Fine-tuned BERT model for binary fake/real news classification.
> Trained on the WELFake dataset (72,134 articles). Achieves ~98-99% F1 on the test set.

## Results

| Model | Accuracy | F1 (macro) | AUC-ROC |
|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | ~93-94% | ~0.93 | ~0.97 |
| BERT fine-tuned (bert-base-uncased) | ~98-99% | ~0.98 | ~0.999 |

## Project Structure 

Fake_News_Detection_with_BERT/
├── configs/config.yaml       <- all hyperparameters and paths
├── scripts/                  <- numbered pipeline scripts (01 to 07)
├── src/                      <- shared dataset, trainer, utils modules
├── app/streamlit_app.py      <- multi-page Streamlit dashboard
├── app/hf_spaces/app.py      <- Gradio demo for HuggingFace Spaces
└── outputs/                  <- charts and results (committed to GitHub)

## Pipeline

1. Data loading and validation
2. Exploratory data analysis
3. Text preprocessing and train/val/test split
4. Baseline: TF-IDF + Logistic Regression
5. BERT fine-tuning (GPU, FP16 mixed precision)
6. Model evaluation and comparison
7. LIME explainability

## Tech Stack

- **Model**: bert-base-uncased (HuggingFace Transformers)
- **Training**: PyTorch + CUDA 12.8 (RTX 5080, FP16)
- **Explainability**: LIME token-level importance
- **Dashboard**: Streamlit (multi-page)
- **Demo**: Gradio on HuggingFace Spaces
- **Model hosting**: HuggingFace Hub

