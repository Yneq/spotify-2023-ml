# 建立 debug_train.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. 載入資料
print("="*60)
print("步驟 1：載入資料")
print("="*60)

df = pd.read_csv("data/processed/spotify_clean.csv")

print(f"CSV 欄位：{df.columns.tolist()}")
print(f"資料形狀：{df.shape}")
print(f"\n前 3 筆資料：")
print(df.head(3))

# 2. 準備特徵和目標
print(f"\n{'='*60}")
print("步驟 2：準備 X 和 y")
print("="*60)

features = [
    'bpm', 'danceability_%', 'valence_%', 'energy_%', 
    'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%'
]

X = df[features]
y = df['log_streams']

print(f"X 形狀：{X.shape}")
print(f"y 形狀：{y.shape}")
print(f"\nX 前 3 筆：")
print(X.head(3))
print(f"\ny 前 3 筆：{y.head(3).tolist()}")
print(f"\ny 統計：")
print(y.describe())

# 3. 分割資料
print(f"\n{'='*60}")
print("步驟 3：分割訓練/測試集")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"訓練集 X：{X_train.shape}")
print(f"測試集 X：{X_test.shape}")
print(f"訓練集 y：{y_train.shape}")
print(f"測試集 y：{y_test.shape}")

print(f"\n訓練集 y 統計：")
print(y_train.describe())
print(f"\n測試集 y 統計：")
print(y_test.describe())

# 4. 訓練模型
print(f"\n{'='*60}")
print("步驟 4：訓練模型")
print("="*60)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✅ 訓練完成")

# 5. 預測
print(f"\n{'='*60}")
print("步驟 5：預測")
print("="*60)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(f"訓練集預測值範圍：{y_pred_train.min():.2f} ~ {y_pred_train.max():.2f}")
print(f"測試集預測值範圍：{y_pred_test.min():.2f} ~ {y_pred_test.max():.2f}")

# 6. 評估
print(f"\n{'='*60}")
print("步驟 6：評估")
print("="*60)

# 訓練集
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = r2_score(y_train, y_pred_train)

# 測試集
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print(f"訓練集 RMSE: {rmse_train:.4f}")
print(f"訓練集 R²:   {r2_train:.4f}")
print(f"\n測試集 RMSE: {rmse_test:.4f}")
print(f"測試集 R²:   {r2_test:.4f}")

# 7. Baseline 比較
print(f"\n{'='*60}")
print("步驟 7：與 Baseline 比較")
print("="*60)

# Baseline：永遠預測平均值
y_mean = y_train.mean()
y_pred_baseline = np.full_like(y_test, y_mean)

rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
r2_baseline = r2_score(y_test, y_pred_baseline)

print(f"Baseline（永遠預測平均值 {y_mean:.2f}）：")
print(f"  RMSE: {rmse_baseline:.4f}")
print(f"  R²:   {r2_baseline:.4f}")

print(f"\n比較：")
print(f"  模型 RMSE: {rmse_test:.4f} vs Baseline RMSE: {rmse_baseline:.4f}")
print(f"  模型 R²:   {r2_test:.4f} vs Baseline R²:   {r2_baseline:.4f}")

if r2_test < 0:
    print(f"\n❌ 模型比 Baseline 還差！")
else:
    print(f"\n✅ 模型比 Baseline 好")

# 8. 看幾個實際預測
print(f"\n{'='*60}")
print("步驟 8：實際預測範例（前 10 筆測試集）")
print("="*60)

comparison = pd.DataFrame({
    '真實值': y_test.iloc[:10].values,
    '預測值': y_pred_test[:10],
    '誤差': y_test.iloc[:10].values - y_pred_test[:10]
})
print(comparison)

# 9. 特徵重要性
print(f"\n{'='*60}")
print("步驟 9：特徵重要性")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)