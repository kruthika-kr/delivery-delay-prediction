# Delivery Delay Prediction

This project predicts whether an e-commerce order will be delayed using engineered logistics and product features.

It started as a notebook-based analysis project and was later turned into a working machine learning app with:
- feature engineering
- LightGBM model training
- SHAP explainability
- FastAPI backend
- Streamlit dashboard

## Problem Statement

Predict whether an order will be delivered late using order, seller, product, and delivery-related signals.

## Project Workflow

1. Loaded and cleaned the raw order data
2. Created a binary target: `is_late`
3. Engineered time, seller, and product features
4. Trained and evaluated ML models
5. Tuned the classification threshold
6. Saved the trained model
7. Built a FastAPI prediction service
8. Built a Streamlit dashboard for live predictions

## Model Performance

- LightGBM ROC-AUC: 0.739
- Recall at threshold 0.50: 0.417
- Recall at threshold 0.30: 0.789

## How to Run

### 1. Start the FastAPI backend

```bash
uvicorn api.main:app --reload
