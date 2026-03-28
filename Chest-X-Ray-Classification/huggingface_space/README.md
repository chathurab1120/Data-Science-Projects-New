---
title: Chest X-Ray Pneumonia Classifier
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
license: mit
---

# 🫁 Chest X-Ray Pneumonia Classifier

AI-powered pneumonia detection from chest X-rays using DenseNet121.

## 🎯 Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 87.8% |
| Recall | 99.5% |
| Precision | 84.0% |
| F1 Score | 91.1% |
| AUC-ROC | 96.8% |
| Specificity | 68.4% |

## 🏗️ Architecture
- Model: DenseNet121 pretrained on ImageNet
- Two-phase transfer learning
- Trained on 5,216 chest X-ray images
- Weighted sampling for class imbalance (2.89x)

## ⚠️ Disclaimer
This tool is for research and educational purposes only.
It is NOT a substitute for professional medical diagnosis.

## 📁 Full Project
GitHub: https://github.com/chathurab1120/Data-Science-Projects-New/tree/main/Chest-X-Ray-Classification

📊 Streamlit Dashboard: https://chest-xray-dashboard.streamlit.app
