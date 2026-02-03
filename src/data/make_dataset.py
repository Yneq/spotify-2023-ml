from pathlib import Path
import pandas as pd
import numpy as np

# 專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "spotify-2023.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "spotify_clean.csv"

FEATURE_COLS = [
    "bpm",
    "danceability_%",
    "valence_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%",
]

TARGET_COL = "streams"


from src.data.load_data import load_spotify_data

def load_raw():
    print("Loading raw data from:", RAW_PATH)
    return load_spotify_data(RAW_PATH)



def clean_streams(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )


def main():
    df = load_raw()

    # 保留必要欄位
    keep_cols = FEATURE_COLS + [TARGET_COL]
    df = df[keep_cols]

    # 清洗 streams
    df[TARGET_COL] = clean_streams(df[TARGET_COL])

    # 丟掉壞資料
    df = df.dropna()

    # log target
    df["log_streams"] = np.log1p(df[TARGET_COL])

    # 確保資料夾存在
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 存檔
    df.to_csv(PROCESSED_PATH, index=False)
    print("Saved clean data to:", PROCESSED_PATH)
    print("Shape:", df.shape)


if __name__ == "__main__":
    main()
