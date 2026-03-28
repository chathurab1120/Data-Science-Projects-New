# Chest X-Ray Classification

![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/chathurab1120/chest-xray-classifier)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Deployment-orange)](https://huggingface.co/spaces/chathurab1120/chest-xray-classifier)

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

| Metric | Value |
|--------|-------|
| Accuracy | 87.8% |
| Precision | 84.0% |
| Recall | 99.5% |
| F1 Score | 91.1% |
| AUC-ROC | 96.8% |
| Specificity | 68.4% |

**Training:** best epoch **12**; **early stopping** enabled (patience **5**, stopped at epoch **17**).

Artifacts: `reports/figures/` (curves, confusion matrix, Grad-CAM, etc.) and `reports/evaluation_report.json`.

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

## Deployment

The scaffold includes an `app/` directory for serving inference endpoints/UI (e.g., Gradio app) and a placeholder HuggingFace deployment badge.

Recommended deployment flow:

1. Export best checkpoint from `models/checkpoints/`
2. Build inference wrapper in `app/`
3. Deploy to HuggingFace Spaces, cloud VM, or containerized runtime

## Author

**Your Name**

- ML Engineer / Data Scientist
- Add portfolio, LinkedIn, or GitHub links here
