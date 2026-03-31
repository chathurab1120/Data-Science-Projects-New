# Fake News Detection with BERT

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fake-news-detection-with-bert.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/chathurab1120/Data-Science-Projects-New/tree/main/Fake_News_Detection_with_BERT)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)](https://huggingface.co/chathurab1120/bert-fake-news-detector)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-red?logo=pytorch)](https://pytorch.org)

> Fine-tuned BERT model for binary fake/real news classification.
> Trained on the WELFake dataset (72,134 articles).
> Achieves 99.09% accuracy and 0.9908 F1 macro on the test set.

## Results

| Model | Accuracy | F1 (macro) | AUC-ROC |
|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 96.59% | 0.9656 | 0.9945 |
| **BERT fine-tuned (bert-base-uncased)** | **99.09%** | **0.9908** | **0.9996** |
| **Improvement** | **+2.50pp** | **+0.0252** | **+0.0051** |

## Live Demo

- **Streamlit Dashboard**: https://fake-news-detection-with-bert.streamlit.app
- **Model weights**: https://huggingface.co/chathurab1120/bert-fake-news-detector

## Project Structure
Fake_News_Detection_with_BERT/
├── configs/config.yaml       <- all hyperparameters and paths
├── scripts/                  <- numbered pipeline scripts (01-07)
├── src/                      <- shared dataset, trainer, utils modules
├── app/streamlit_app.py      <- multi-page Streamlit dashboard
└── outputs/                  <- charts and results

## Pipeline

| Stage | Script | Output |
|-------|--------|--------|
| 1 | Data Loading | 72,134 articles validated |
| 2 | EDA | 6 exploratory charts |
| 3 | Preprocessing | 63,606 clean articles, 80/10/10 split |
| 4 | Baseline Model | TF-IDF + LR, F1=0.9656 |
| 5 | BERT Fine-Tuning | bert-base-uncased, 4 epochs, FP16 |
| 6 | Evaluation | Test F1=0.9908, AUC=0.9996 |
| 7 | Explainability | LIME token-level explanations |

## Key Technical Decisions

- **Baseline first**: TF-IDF + Logistic Regression establishes a strong 96.6% F1 benchmark before investing in deep learning
- **Deduplication before splitting**: 8,471 duplicate articles removed before train/val/test split to prevent data leakage
- **FP16 mixed precision**: Cuts GPU memory usage and speeds training ~2x
- **LIME explainability**: Token-level feature importance shows which words drove each prediction
- **Model hosting**: Fine-tuned weights hosted on HuggingFace Hub, downloaded at runtime on Streamlit Cloud

## Tech Stack

- **Model**: bert-base-uncased (HuggingFace Transformers)
- **Training**: PyTorch + CUDA 12.8, FP16 mixed precision, 17.8 min
- **Explainability**: LIME token importance, 12 sample explanations
- **Dashboard**: Streamlit multi-page app (5 pages)
- **Model hosting**: HuggingFace Hub
