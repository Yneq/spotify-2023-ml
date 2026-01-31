import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "spotify_clean.csv"

EXPERIMENT_NAME = "spotify-streams-regression"


def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df[
        [
            "bpm",
            "danceability_%",
            "valence_%",
            "energy_%",
            "acousticness_%",
            "liveness_%",
            "speechiness_%",
            "instrumentalness_%",
        ]
    ]
    y = df["log_streams"]

    return X, y


def train():
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        X, y = load_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log params + metrics
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("rmse", rmse)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"RMSE: {rmse:.4f}")
        print("Run logged to MLflow")


if __name__ == "__main__":
    train()

