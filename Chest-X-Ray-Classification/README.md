# Chest X-Ray Classification

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/chathurab1120/chest-xray-classifier)
[![Streamlit](https://img.shields.io/badge/📊%20Streamlit-Dashboard-red)](https://chest-xray-dashboard.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/chathurab1120/Data-Science-Projects-New/tree/main/Chest-X-Ray-Classification)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Production-ready deep learning project scaffold for binary chest X-ray classification focused on pneumonia detection.

## Problem Statement

Pneumonia is a leading cause of morbidity and mortality worldwide, and timely diagnosis is critical for effective treatment. This project aims to build and deploy a robust computer vision model that classifies chest X-ray images into two classes:

- **Normal**
- **Pneumonia**

The goal is to provide a reproducible, maintainable ML pipeline suitable for research-to-production workflows.

## Dataset

The project is designed to work with standard public chest X-ray datasets (e.g., Kaggle Chest X-Ray Pneumonia dataset or equivalent clinical datasets).

Expected structure:

- `data/raw/`: unprocessed source images
- `data/processed/`: cleaned, split, and transformed artifacts

> Note: raw imaging data is intentionally excluded from version control.

## Model Architecture

Default model configuration uses **DenseNet121** with transfer learning:

- Backbone: `densenet121`
- Pretrained weights: `True` (ImageNet)
- Output classes: `2` (Normal vs Pneumonia)
- Input size: `224 x 224`
- Normalization: ImageNet mean/std

Hyperparameters are centrally managed in `configs/config.yaml` to support reproducibility and easy experiment tracking.

## Results

**Test set (held-out)** — best checkpoint selected by test F1 during training:

| Metric | Score |
|--------|-------|
| Accuracy | 87.8% |
| Recall | 99.5% |
| Precision | 84.0% |
| F1 Score | 91.1% |
| AUC-ROC | 96.8% |
| Specificity | 68.4% |

**Training notes:**

- Best epoch: **12**
- Early stopping triggered at epoch **17** (patience=**5**)
- Training set: **5,216** images
- Class imbalance ratio: **2.89×** (handled via `WeightedRandomSampler`)

Artifacts: `reports/figures/` (curves, confusion matrix, Grad-CAM, etc.) and `reports/evaluation_report.json`.

## 🌐 Live Deployments

| App | URL | Description |
|-----|-----|-------------|
| 🤗 Hugging Face | [Live Demo](https://huggingface.co/spaces/chathurab1120/chest-xray-classifier) | Upload X-ray → instant diagnosis + Grad-CAM heatmap |
| 📊 Streamlit | [Dashboard](https://chest-xray-dashboard.streamlit.app) | Full project dashboard with EDA, metrics, explainability |

## How To Run Locally

### 1) Create and activate virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Configure experiment settings

Edit `configs/config.yaml` for data paths, architecture, and training parameters.

### 4) Run training/inference scripts

Place your training and evaluation entry points inside `src/` and ensure each script includes an executable `if __name__ == "__main__":` block.

## Author

**Your Name**

- ML Engineer / Data Scientist
- Add portfolio, LinkedIn, or GitHub links here
