---
title: Chest X-Ray Pneumonia Classifier
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.7.1
app_file: gradio_app.py
pinned: false
---

# Chest X-Ray Pneumonia Classifier

This Hugging Face Space deploys a DenseNet121-based deep learning model for binary chest X-ray classification:

- `NORMAL`
- `PNEUMONIA`

The app includes Grad-CAM visual explanations so users can inspect where the model focuses when making predictions.

## Model Architecture

- Backbone: DenseNet121 (transfer learning)
- Custom classifier head:
  - Linear(1024 -> 512)
  - ReLU
  - Dropout(0.4)
  - Linear(512 -> 2)
- Input pipeline:
  - Resize(256), CenterCrop(224)
  - Grayscale converted to RGB
  - Normalize with dataset stats:
    - Mean: `[0.4693, 0.4693, 0.4693]`
    - Std: `[0.2270, 0.2270, 0.2270]`

## Performance Metrics

Evaluated on the held-out test set:

- Accuracy: **84.5%**
- Recall: **99.7%**
- AUC-ROC: **95.0%**
- F1 Score: **88.9%**

High recall is especially important for screening use cases to reduce missed pneumonia-positive cases.

## Dataset Credit

- Kaggle Chest X-Ray Images (Pneumonia)

## Disclaimer

This project is for **research and educational use only**.  
It is **not** a medical device and must not be used as a substitute for professional radiology or clinical diagnosis.

## Links

- GitHub project: [GitHub URL](https://github.com/your-username/your-repo)
- Streamlit dashboard: [Streamlit URL](https://your-streamlit-app-url.streamlit.app)
