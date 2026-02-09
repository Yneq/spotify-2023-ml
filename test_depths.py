import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 載入資料
df = pd.read_csv("data/processed/spotify_clean.csv")

features = [
    'bpm', 'danceability_%', 'valence_%', 'energy_%', 
    'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%'
]

X = df[features]
y = df['log_streams']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("測試不同的 max_depth：")
print("="*70)
print(f"{'max_depth':<12} {'訓練 R²':<12} {'測試 R²':<12} {'差距':<12} {'評價'}")
print("="*70)

for depth in [2, 3, 4, 5, 6, 8, 10, None]:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=depth,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    gap = r2_train - r2_test
    
    # 評價
    if r2_test < 0:
        grade = "❌ 很差"
    elif gap > 0.2:
        grade = "⚠️ 過擬合"
    elif r2_test < 0.3:
        grade = "🔸 普通"
    else:
        grade = "✅ 良好"
    
    depth_str = str(depth) if depth else "None"
    print(f"{depth_str:<12} {r2_train:<12.4f} {r2_test:<12.4f} {gap:<12.4f} {grade}")

print("="*70)
print("\n建議：選擇「測試 R²」最高且「差距」最小的 max_depth")