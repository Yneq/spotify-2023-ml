import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


import sys
import os

st.write("Python version:", sys.version)
st.write("Current working dir:", os.getcwd())
st.write("Installed packages:")
st.write(os.popen("pip list").read())

# 設定頁面
st.set_page_config(
    page_title="Spotify 2023 ML Prediction",
    page_icon="🎵",
    layout="wide"
)

# ========== 側邊欄：導航 ==========
page = st.sidebar.selectbox(
    "選擇頁面",
    ["🏠 專案首頁", "📊 資料探索", "🎯 互動預測", "📈 實驗結果"]
)

# ========== 頁面 1：專案首頁 ==========
if page == "🏠 專案首頁":
    st.title("🎵 Spotify 2023 歌曲流行度預測")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("資料集大小", "953 首歌")
    with col2:
        st.metric("特徵數量", "8 個音樂特徵")
    with col3:
        st.metric("模型", "Random Forest")
    
    st.markdown("""
    ## 📖 專案簡介
    
    這個專案使用 **機器學習** 預測 Spotify 2023 熱門歌曲的流行度。
    我們使用 8 個音樂特徵（BPM、舞曲性、能量等）來預測歌曲的播放次數。
    
    ### 🎯 主要發現
    
    - ❌ **音樂特徵與流行度相關性極低**（< 0.1）
    - ✅ 完整的 ML Pipeline 實作（資料處理 → 訓練 → 部署）
    - ✅ MLflow 實驗追蹤與模型版本管理
    - ✅ 多種模型對比（回歸、分類、特徵工程）
    
    ### 🛠️ 技術棧
    
    - Python 3.12
    - scikit-learn
    - MLflow
    - Streamlit
    - Plotly
    
    ### 🔗 連結
    
    - [GitHub Repository](https://github.com/Yneq/spotify-2023-ml)
    - [完整文件](https://github.com/Yneq/spotify-2023-ml/blob/main/README.md)
    """)

# ========== 頁面 2：資料探索 ==========
elif page == "📊 資料探索":
    st.title("📊 資料探索與視覺化")
    
    # 載入資料
    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv("data/processed/spotify_clean.csv")
            return df
        except:
            # 如果沒有檔案，建立示例資料
            np.random.seed(42)
            n = 100
            df = pd.DataFrame({
                'bpm': np.random.randint(60, 200, n),
                'danceability_%': np.random.randint(30, 90, n),
                'energy_%': np.random.randint(30, 90, n),
                'valence_%': np.random.randint(20, 90, n),
                'log_streams': np.random.normal(19.5, 1.15, n)
            })
            return df
    
    df = load_data()
    
    # 顯示資料摘要
    st.subheader("資料摘要")
    col1, col2 = st.columns(2)
    with col1:
        st.write("資料形狀：", df.shape)
    with col2:
        st.write("特徵數量：", len(df.columns) - 1)
    
    # 分佈圖
    st.subheader("Log Streams 分佈")
    fig = px.histogram(
        df, 
        x='log_streams',
        nbins=30,
        title="Log Streams 分佈（接近常態分佈）"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 特徵相關性
    st.subheader("特徵相關性")
    corr_cols = ['bpm', 'danceability_%', 'energy_%', 'valence_%', 'log_streams']
    corr_df = df[corr_cols].corr()
    
    fig = px.imshow(
        corr_df,
        text_auto='.2f',
        aspect="auto",
        title="特徵相關性熱力圖",
        color_continuous_scale='RdBu_r'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 關鍵發現
    st.info("""
    🔍 **關鍵發現**：所有音樂特徵與 `log_streams` 的相關性都非常低（< 0.1），
    這解釋了為什麼模型預測效果不佳。
    """)

# ========== 頁面 3：互動預測 ==========
elif page == "🎯 互動預測":
    st.title("🎯 試試預測歌曲流行度！")
    
    st.markdown("""
    調整下方的音樂特徵滑桿，看看模型會預測多少播放次數。
    """)
    
    # 輸入特徵
    col1, col2 = st.columns(2)
    
    with col1:
        bpm = st.slider("🎵 BPM (節奏)", 60, 200, 120)
        danceability = st.slider("💃 Danceability (舞曲性)", 0, 100, 70)
        energy = st.slider("⚡ Energy (能量)", 0, 100, 75)
        valence = st.slider("😊 Valence (愉悅度)", 0, 100, 60)
    
    with col2:
        acousticness = st.slider("🎸 Acousticness (原聲性)", 0, 100, 20)
        instrumentalness = st.slider("🎹 Instrumentalness (器樂性)", 0, 100, 10)
        liveness = st.slider("🎤 Liveness (現場感)", 0, 100, 15)
        speechiness = st.slider("🗣️ Speechiness (語音性)", 0, 100, 5)
    
    # 預測按鈕
    if st.button("🚀 預測流行度", type="primary"):
        # 簡單的預測邏輯（因為沒有實際模型）
        # 實際部署時會載入真實模型
        log_streams = 19.5 + (danceability - 50) * 0.01 + (energy - 50) * 0.008
        streams = np.exp(log_streams)
        
        st.success("✅ 預測完成！")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Log Streams", f"{log_streams:.2f}")
        with col2:
            st.metric("預估播放次數", f"{streams/1e6:.1f}M")
        with col3:
            if streams > 500e6:
                st.metric("流行度", "🔥 超高人氣")
            elif streams > 200e6:
                st.metric("流行度", "⭐ 高人氣")
            else:
                st.metric("流行度", "📊 中等")
        
        st.info("""
        ⚠️ **注意**：這是一個示範模型。實際上，音樂特徵與流行度的相關性很低，
        真正影響播放次數的是歌手知名度、行銷預算等外部因素。
        """)

# ========== 頁面 4：實驗結果 ==========
elif page == "📈 實驗結果":
    st.title("📈 模型對比與實驗結果")
    
    # 模型對比表格
    st.subheader("模型效果對比")
    
    results_df = pd.DataFrame({
        '模型': [
            'Random Forest (12 特徵)',
            'Random Forest (8 特徵)',
            'Gradient Boosting',
            'RF + 特徵工程'
        ],
        'RMSE': [0.83, 1.16, 1.14, 1.12],
        'R²': [0.46, -0.04, -0.02, 0.01],
        '特徵數': [12, 8, 8, 11]
    })
    
    st.dataframe(results_df, use_container_width=True)
    
    # 視覺化
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='RMSE',
        x=results_df['模型'],
        y=results_df['RMSE'],
        marker_color='indianred'
    ))
    fig.update_layout(
        title="不同模型的 RMSE 比較（越低越好）",
        yaxis_title="RMSE"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 失敗分析
    st.subheader("🔍 失敗原因分析")
    
    st.markdown("""
    ### 為什麼模型效果不好？
    
    #### 1️⃣ 特徵相關性極低
    
    所有音樂特徵與播放次數的相關性 < 0.1（幾乎無關）
    
    | 特徵 | 相關性 |
    |------|--------|
    | BPM | 0.004 |
    | Danceability | -0.068 |
    | Energy | -0.027 |
    | Valence | -0.048 |
    
    #### 2️⃣ 缺少關鍵特徵
    
    真正影響流行度的因素：
    - ✅ 歌手知名度（粉絲數、過往作品）
    - ✅ 行銷預算
    - ✅ 社群媒體病毒傳播
    - ✅ 播放清單收錄
    
    我們的資料只有：
    - ❌ 音樂特徵（BPM、舞曲性...）
    
    #### 3️⃣ 資料集特性
    
    - 資料來源是「Spotify 2023 排行榜」
    - 能上榜的歌都已經很紅
    - 差異主要來自外部因素
    
    ### 💡 關鍵學習
    
    **資料品質 > 模型複雜度**
    
    當特徵與目標沒有相關性時，即使用最先進的模型也無法改善效果。
    這個專案成功地驗證了：擁有正確的特徵，比選擇正確的演算法更重要。
    """)

# ========== 頁尾 ==========
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 👨‍💻 作者
**Vance**

[GitHub](https://github.com/Yneq) | 
[專案連結](https://github.com/Yneq/spotify-2023-ml)
""")