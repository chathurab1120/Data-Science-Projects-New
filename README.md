# Data Science Projects

A curated collection of end-to-end machine learning and deep learning projects.

## 1. Credit Card Fraud Detection
- **Domain:** FinTech + Anomaly Detection
- **Project Folder:** `Credit-Card-Fraud-Detection`

## 🫁 2. Chest X-Ray Pneumonia Classification
**Domain:** Healthcare + Deep Learning  
**Model:** DenseNet121 (Transfer Learning)  
**Dataset:** Kaggle Chest X-Ray Images (5,216 images)

| Metric | Score |
|--------|-------|
| Accuracy | 87.8% |
| Precision | 84.0% |
| Recall | 99.5% |
| F1 Score | 91.1% |
| AUC-ROC | 96.8% |
| Specificity | 68.4% |

**Key Features:**
- Two-phase transfer learning (feature extraction → fine-tuning)
- Weighted sampling + loss for class imbalance (2.89x)
- Grad-CAM explainability heatmaps
- Deployed on Hugging Face Spaces + Streamlit Cloud

**Training note:** Early stopping at epoch 17; best checkpoint at epoch 12 (test F1).

**Deployments:**
- 🤗 [Live Demo - Hugging Face](https://huggingface.co/spaces/chathurab1120/chest-xray-classifier)
- 📊 [Dashboard - Streamlit](https://chest-xray-dashboard.streamlit.app)

[📁 View Project](https://github.com/chathurab1120/Data-Science-Projects-New/tree/main/Chest-X-Ray-Classification)
