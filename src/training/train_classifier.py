from pathlib import Path
import argparse
import yaml
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

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
    
    if model_type == "random_forest_classifier":
        return RandomForestClassifier(**model_cfg["params"])
    elif model_type == "gradient_boosting":  # ← 新增
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(**model_cfg["params"])
    
    raise ValueError(f"Unknown model type: {model_type}")


def train(config_path: Path):
    cfg = load_config(config_path)
    
    set_seed(cfg["project"]["seed"])
    
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    mlflow_cfg = cfg["mlflow"]
    
    # 載入資料
    X, y = load_data(
        PROJECT_ROOT / data_cfg["processed_path"],
        data_cfg["features"],
        data_cfg["target"],
    )
    
    print(f"資料形狀：X={X.shape}, y={y.shape}")
    print(f"目標分佈：\n{y.value_counts()}")
    
    # 分割資料
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_cfg["test_size"],
        random_state=cfg["project"]["seed"],
        stratify=y,  # 確保訓練/測試集的類別比例相同
    )
    
    # 建立模型
    model = build_model(model_cfg)
    
    # MLflow 記錄
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    
    with mlflow.start_run():
        # 記錄參數
        mlflow.log_params(model_cfg["params"])
        mlflow.log_param("model_type", model_cfg["type"])
        mlflow.log_param("test_size", train_cfg["test_size"])
        mlflow.log_param("n_features", len(data_cfg["features"]))
        
        # 訓練模型
        model.fit(X_train, y_train)
        
        # 預測
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # 正類別的機率
        
        # 計算指標
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # 記錄指標
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)
        
        # 記錄模型
        signature = mlflow.models.signature.infer_signature(
            X_train, model.predict(X_train)
        )
        
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="spotify-popularity-classifier",
            signature=signature,
        )
        
        # 印出結果
        print("\n" + "="*60)
        print("訓練完成！")
        print("="*60)
        print(f"準確率 (Accuracy):  {accuracy:.4f}")
        print(f"精確率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall):    {recall:.4f}")
        print(f"F1 分數:            {f1:.4f}")
        print(f"AUC:                {auc:.4f}")
        print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    train(Path(args.config))