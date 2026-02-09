# check_features.py
import pandas as pd

df = pd.read_csv("data/processed/spotify_clean.csv")

# 檢查目標變數分佈
print("log_streams 統計：")
print(df["log_streams"].describe())

# 檢查各特徵與目標的相關性
features = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 
            'acousticness_%', 'instrumentalness_%', 'liveness_%', 
            'speechiness_%', 'artist_count', 'released_year', 
            'released_month', 'released_day', 'key', 'mode']

print("\n特徵與 log_streams 的相關性：")
for feat in features:
    corr = df[feat].corr(df["log_streams"])
    print(f"{feat:20s}: {corr:6.3f}")