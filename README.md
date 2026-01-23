# 🎵 Spotify 2023 Top Songs — ML Regression Pipeline

This project builds a reproducible machine learning pipeline to predict Spotify song popularity (stream counts) using audio features. It focuses on clean engineering practices, experiment tracking, and model evaluation, making it suitable for portfolios and technical interviews.

---

## 📌 Project Overview

The goal of this project is to predict the popularity of a song, measured by the log-transformed number of Spotify streams, using numerical audio features such as danceability, energy, and tempo. Since raw stream counts follow a long-tailed distribution, a log transformation is applied to stabilize variance and improve regression performance.

---

## 📊 Dataset

Source: Kaggle — Top Spotify Songs 2023  
The dataset contains approximately 953 songs and 24 columns, including audio characteristics and metadata. The target variable is `streams`, which is transformed into `log_streams` for training.

---

## 🔧 Features Used

The model uses only numerical audio features to avoid data leakage from popularity-based or identity-based fields:

- danceability_%  
- energy_%  
- valence_%  
- acousticness_%  
- instrumentalness_%  
- liveness_%  
- speechiness_%  
- bpm  

---

## 🏗️ Project Structure

spotify-2023-ml/
├─ .gitignore
├─ configs/
│ └─ baseline.yaml
├─ data/
│ ├─ raw/ (ignored)
│ └─ processed/ (ignored)
├─ src/
│ ├─ data/
│ ├─ features/
│ ├─ models/
│ ├─ training/
│ └─ evaluation/
├─ notebooks/
├─ .pre-commit-config.yaml
├─ requirements.txt
└─ README.md


---

## 🚀 How to Run

### 1. Installation

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Download Dataset (Kaggle)

Generate a Kaggle API token and place kaggle.json in ~/.kaggle/. Then download the dataset:

kaggle datasets download -d nelgiriyewithana/top-spotify-songs-2023 -p data/raw --unzip

3. Training with Config

Train the baseline model using a YAML configuration file:

python -m src.training.train --config configs/baseline.yaml


This will load the data, clean and transform features, train the model, evaluate performance, and log the experiment using MLflow.

📈 Baseline Results
Model	Target	RMSE
Linear Regression	log_streams	~1.008
Random Forest	log_streams	~1.029

These results indicate that audio features alone provide a weak-to-moderate signal for predicting popularity, and that non-audio factors such as artist recognition and platform exposure likely play a significant role.

🛠️ Tech Stack

Python

pandas, numpy

scikit-learn

MLflow

Git & GitHub

pre-commit (black, ruff)

Jupyter Notebook

⭐ Highlights

Reproducible, config-driven training pipeline

Clean separation of data, features, training, and evaluation code

Experiment tracking with MLflow

Feature importance analysis and baseline model comparison

Structured project layout suitable for ML engineering roles

📂 Future Work

Add NLP-based lyric features

Hyperparameter tuning (Grid Search / Optuna)

Model deployment with FastAPI

Interactive demo with Streamlit

Vector search and semantic similarity (FAISS)

This will load the data, clean and transform features, train the model, evaluate performance, and log the experiment using MLflow.

📈 Baseline Results
Model	Target	RMSE
Linear Regression	log_streams	~1.008
Random Forest	log_streams	~1.029

These results indicate that audio features alone provide a weak-to-moderate signal for predicting popularity, and that non-audio factors such as artist recognition and platform exposure likely play a significant role.

🛠️ Tech Stack

Python

pandas, numpy

scikit-learn

MLflow

Git & GitHub

pre-commit (black, ruff)

Jupyter Notebook

⭐ Highlights

Reproducible, config-driven training pipeline

Clean separation of data, features, training, and evaluation code

Experiment tracking with MLflow

Feature importance analysis and baseline model comparison

Structured project layout suitable for ML engineering roles

📂 Future Work

Add NLP-based lyric features

Hyperparameter tuning (Grid Search / Optuna)

Model deployment with FastAPI

Interactive demo with Streamlit

Vector search and semantic similarity (FAISS)


