# Streamlit Cloud Deployment Guide

## Prerequisites
- GitHub repo pushed ✅
- Streamlit account at share.streamlit.io

## Steps
1. Go to https://share.streamlit.io
2. Click "New app"
3. Repository: chathurab1120/Data-Science-Projects-New
4. Branch: main
5. Main file path: Chest-X-Ray-Classification/app/streamlit_app.py
6. Click "Deploy"

## Environment Variables to set in Streamlit Cloud:
None required for basic deployment.

## Important Notes:
- Model weights (best_model.pth) must be committed to repo OR
  loaded from Hugging Face Hub
- Data files are NOT needed for the dashboard pages 1, 2, and 4
- Only Page 3 (Live Prediction) needs the model weights

## Adding model weights to GitHub LFS:
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add models/checkpoints/best_model.pth
git commit -m "feat: add trained model weights via Git LFS"
git push origin main
