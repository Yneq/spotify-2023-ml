# src/data/make_dataset.py

from pathlib import Path
import pandas as pd
import numpy as np
from src.data.load_data import load_spotify_data

# 專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "spotify-2023.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "spotify_clean.csv"

# 基礎特徵（數值型）
NUMERIC_FEATURES = [
    "bpm",
    "danceability_%",
    "valence_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%",
    "artist_count",
    "released_year",
    "released_month",
    "released_day",
]

# 類別特徵（文字型，需要編碼）
CATEGORICAL_FEATURES = [
    "key",
    "mode",
]

TARGET_COL = "streams"


def load_raw():
    """載入原始資料"""
    print("Loading raw data from:", RAW_PATH)
    return load_spotify_data(RAW_PATH)


def clean_streams(series: pd.Series) -> pd.Series:
    """清洗 streams 欄位"""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """將類別特徵轉換成數字"""
    
    df = df.copy()
    
    # 1. 處理 key（調性）
    if "key" in df.columns:
        # Label Encoding：C=0, C#=1, D=2, ...
        key_mapping = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        }
        df['key'] = df['key'].map(key_mapping)
        print(f"✅ 編碼 key：{sorted(df['key'].dropna().unique())}")
    
    # 2. 處理 mode（大調/小調）
    if "mode" in df.columns:
        # Label Encoding：Major=1, Minor=0
        mode_mapping = {'Major': 1, 'Minor': 0}
        df['mode'] = df['mode'].map(mode_mapping)
        print(f"✅ 編碼 mode：{sorted(df['mode'].dropna().unique())}")
    
    return df


def main():
    """主要處理流程"""
    
    # 1. 載入原始資料
    df = load_raw()
    print(f"原始資料：{df.shape}")

    # 2. 選擇需要的欄位
    keep_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]
    
    # 檢查欄位是否存在
    missing = set(keep_cols) - set(df.columns)
    if missing:
        print(f"⚠️ 警告：以下欄位不存在，將被忽略：{missing}")
        keep_cols = [c for c in keep_cols if c in df.columns]
    
    df = df[keep_cols]
    print(f"選擇欄位後：{df.shape}")

    # 3. 清洗 streams
    df[TARGET_COL] = clean_streams(df[TARGET_COL])

    # 4. 編碼類別特徵
    print("\n編碼類別特徵...")
    df = encode_categorical(df)

    # 5. 移除缺失值
    print(f"\n移除缺失值前：{len(df)} 筆")
    df = df.dropna()
    print(f"移除缺失值後：{len(df)} 筆")

    # 6. 計算 log_streams
    df["log_streams"] = np.log1p(df[TARGET_COL])

    # 7. 移除原始 streams，只保留特徵和目標
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    # 只保留存在的欄位
    all_features = [f for f in all_features if f in df.columns]
    final_cols = all_features + ["log_streams"]
    df = df[final_cols]
    
    # 8. 驗證
    print(f"\n{'='*60}")
    print("最終資料：")
    print(f"{'='*60}")
    print(f"欄位數：{len(df.columns)}")
    print(f"欄位名：{df.columns.tolist()}")
    print(f"形狀：{df.shape}")
    
    print(f"\n前 3 筆資料：")
    print(df.head(3))
    
    print(f"\n資料型態檢查：")
    print(df.dtypes)
    
    # 確認沒有非數值欄位
    non_numeric = df.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        print(f"\n❌ 錯誤：仍有非數值欄位：{non_numeric}")
        raise ValueError("資料中仍有文字欄位，無法訓練模型")
    else:
        print(f"\n✅ 確認：所有欄位都是數值型")
    
    assert "streams" not in df.columns, "❌ streams 不應該在最終資料中！"
    print("✅ 確認：已移除 streams 欄位")

    # 9. 儲存
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    
    print(f"\n✅ 已儲存到：{PROCESSED_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()