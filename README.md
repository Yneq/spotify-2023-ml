🎵 Spotify 2023 Top Songs — End-to-End ML Regression System

This repository implements a production-style machine learning system for predicting Spotify song popularity (log-transformed stream counts) from numerical audio features.
The project emphasizes reproducibility, experiment tracking, model versioning, and deployable inference services, making it suitable for ML Engineer, Backend, and MLOps-oriented portfolios.

🔥 System Overview

This project covers the full ML lifecycle:

Data → Features → Training → Evaluation → Model Registry → Inference API → Containerization

Config-driven training pipeline (YAML)

Experiment tracking & model versioning with MLflow

Feature schema validation to prevent training-serving skew

REST inference service (FastAPI)

Dockerized deployment for portable execution

📊 Dataset

Source: Kaggle — Top Spotify Songs 2023
The dataset contains ~953 songs and 24 columns, including numerical audio features and metadata.

Raw data is excluded from version control

Processed features are generated through a reproducible data pipeline

Target variable: streams → transformed into log_streams for regression stability

🔧 Features Used

Only numerical audio features are used to avoid data leakage from popularity-based or identity-based fields:

bpm

danceability_%

energy_%

valence_%

acousticness_%

instrumentalness_%

liveness_%

speechiness_%

🏗️ Project Structure
🔥 System Overview

This project covers the full ML lifecycle:

Data → Features → Training → Evaluation → Model Registry → Inference API → Containerization

Config-driven training pipeline (YAML)

Experiment tracking & model versioning with MLflow

Feature schema validation to prevent training-serving skew

REST inference service (FastAPI)

Dockerized deployment for portable execution

📊 Dataset

Source: Kaggle — Top Spotify Songs 2023
The dataset contains ~953 songs and 24 columns, including numerical audio features and metadata.

Raw data is excluded from version control

Processed features are generated through a reproducible data pipeline

Target variable: streams → transformed into log_streams for regression stability

🔧 Features Used

Only numerical audio features are used to avoid data leakage from popularity-based or identity-based fields:

bpm

danceability_%

energy_%

valence_%

acousticness_%

instrumentalness_%

liveness_%

speechiness_%

🏗️ Project Structure

🚀 Quick Start
1. Environment Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Download Dataset
kaggle datasets download -d nelgiriyewithana/top-spotify-songs-2023 -p data/raw --unzip

3. Train Model (Config-Driven)
python -m src.training.train --config configs/baseline.yaml

This will:

Clean and transform the dataset
Train the model
Evaluate performance
Log experiments and register the model in MLflow

📈 Baseline Results
Model	Target	RMSE
Linear Regression	log_streams	~1.01
Random Forest	log_streams	~1.03

These results highlight that audio features alone provide limited predictive signal, and external factors such as artist popularity and platform exposure likely dominate streaming outcomes.

🔥 Inference API (Production Style)

The trained model can be served as a REST API using FastAPI.

Run Locally
docker build -t spotify-ml .
docker run -p 8000:8000 spotify-ml

Example Request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bpm": 120,
    "danceability_%": 60,
    "valence_%": 50,
    "energy_%": 70,
    "acousticness_%": 10,
    "instrumentalness_%": 0,
    "liveness_%": 15,
    "speechiness_%": 5
  }'

🛠️ Tech Stack

Python

pandas, numpy, scikit-learn

MLflow (Experiments & Model Registry)

FastAPI (Inference Service)

Docker (Containerized Deployment)

Git & GitHub

pre-commit (black, ruff)

⭐ Highlights

Reproducible, config-driven training pipeline

MLflow-based experiment tracking and model versioning

Feature schema validation and alignment at inference time

Containerized inference service for portable deployment

Clean separation of data, training, and serving layers

📂 Future Work

NLP-based lyric feature extraction

Hyperparameter tuning (Optuna)

CI/CD for model retraining and deployment

Streamlit interactive demo

Vector search & semantic similarity (FAISS)