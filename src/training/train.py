from pathlib import Path
import argparse
import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int):
    np.random.seed(seed)


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data(processed_path: Path, features: list, target: str):
    df = pd.read_csv(processed_path)
    X = df[features]
    y = df[target]
    return X, y


def build_model(model_cfg: dict):
    model_type = model_cfg["type"]

    if model_type == "random_forest":
        return RandomForestRegressor(**model_cfg["params"])

    raise ValueError(f"Unknown model type: {model_type}")


def train(config_path: Path):
    cfg = load_config(config_path)

    set_seed(cfg["project"]["seed"])

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    mlflow_cfg = cfg["mlflow"]

    X, y = load_data(
        PROJECT_ROOT / data_cfg["processed_path"],
        data_cfg["features"],
        data_cfg["target"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_cfg["test_size"],
        random_state=cfg["project"]["seed"],
    )

    model = build_model(model_cfg)

    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run():
        # Log params
        mlflow.log_params(model_cfg["params"])
        mlflow.log_param("model_type", model_cfg["type"])
        mlflow.log_param("test_size", train_cfg["test_size"])

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print("Training complete")
        print("RMSE:", rmse)
        print("R2:", r2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML",
    )
    args = parser.parse_args()

    train(Path(args.config))
