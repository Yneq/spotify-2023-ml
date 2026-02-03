from pathlib import Path
import mlflow
import pandas as pd

MODEL_NAME = "spotify-streams-regressor"
MODEL_ALIAS = "production"


def load_production_model():
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    return mlflow.sklearn.load_model(model_uri)


def validate_and_align_schema(model, df: pd.DataFrame) -> pd.DataFrame:
    expected = list(model.feature_names_in_)

    print("Expected features:", expected)
    print("Input features:", df.columns.tolist())

    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 強制照模型訓練時的順序重排
    return df[expected]


def predict(input_csv: str):
    model = load_production_model()
    df = pd.read_csv(input_csv)

    X = validate_and_align_schema(model, df)

    preds = model.predict(X)

    df["prediction_log_streams"] = preds
    print(df.head())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    predict(args.input)
