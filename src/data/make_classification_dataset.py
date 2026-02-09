from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def preprocess_for_classification():
    """將資料處理成分類問題：預測歌曲是否高人氣"""
    
    # 讀取原始資料
    raw_path = PROJECT_ROOT / "data/raw/spotify-2023.csv"
    print(f"讀取原始資料：{raw_path}")
    df = pd.read_csv(raw_path, encoding="ISO-8859-1")
    print(f"原始資料：{df.shape}")
    
    # 處理 streams 欄位
    df["streams"] = pd.to_numeric(df["streams"], errors="coerce")
    df = df.dropna(subset=["streams"])
    df = df[df["streams"] > 0]
    print(f"清理 streams 後：{df.shape}")
    
    # 新增交互特徵
    df["dance_energy"] = df["danceability_%"] * df["energy_%"]  # 舞曲 × 能量
    df["valence_energy"] = df["valence_%"] * df["energy_%"]    # 愉悅 × 能量
    df["total_vibe"] = (
        df["danceability_%"] + 
        df["energy_%"] + 
        df["valence_%"]
    ) / 3  # 平均「氛圍分數」
    
    # 更新特徵列表
    feature_cols = [
        "bpm",
        "danceability_%",
        "valence_%",
        "energy_%",
        "acousticness_%",
        "instrumentalness_%",
        "liveness_%",
        "speechiness_%",
        "dance_energy",      # 新特徵
        "valence_energy",    # 新特徵
        "total_vibe",        # 新特徵
    ]
    
    # 檢查欄位是否存在
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(f"警告：缺少欄位 {missing}")
    
    # 選擇存在的欄位
    available_features = [col for col in feature_cols if col in df.columns]
    
    # 建立目標變數：是否為高人氣歌曲（前 30%）
    threshold = df["streams"].quantile(0.90)  # 前 10% 為高人氣
    df["is_popular"] = (df["streams"] >= threshold).astype(int)
    
    print(f"\n高人氣歌曲定義：播放次數 >= {threshold:,.0f}")
    print(f"高人氣歌曲數量：{df['is_popular'].sum()} ({df['is_popular'].mean()*100:.1f}%)")
    print(f"低人氣歌曲數量：{(1-df['is_popular']).sum()} ({(1-df['is_popular'].mean())*100:.1f}%)")
    
    # 選擇最終欄位
    final_cols = available_features + ["is_popular"]
    df_final = df[final_cols].copy()
    
    # 移除缺失值
    print(f"\n移除缺失值前：{len(df_final)} 筆")
    df_final = df_final.dropna()
    print(f"移除缺失值後：{len(df_final)} 筆")
    
    # 儲存
    output_path = PROJECT_ROOT / "data/processed/spotify_classification.csv"
    df_final.to_csv(output_path, index=False)
    print(f"\n✅ 已儲存到：{output_path}")
    
    # 顯示資料摘要
    print("\n" + "="*60)
    print("最終資料摘要：")
    print("="*60)
    print(f"特徵數：{len(available_features)}")
    print(f"特徵名：{available_features}")
    print(f"資料筆數：{len(df_final)}")
    print(f"\n前 3 筆資料：")
    print(df_final.head(3))
    print(f"\n目標變數分佈：")
    print(df_final["is_popular"].value_counts())
    

if __name__ == "__main__":
    preprocess_for_classification()