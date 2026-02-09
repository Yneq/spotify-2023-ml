# 🎵 Spotify 2023 Top Songs — ML Prediction System

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green.svg)](https://scikit-learn.org/)

This repository implements a **complete machine learning pipeline** for predicting Spotify song popularity using audio features. The project demonstrates:

- 🔄 **Full ML lifecycle**: Data processing → Training → Evaluation → Deployment
- 📊 **Experiment tracking** with MLflow
- 🧪 **Multiple problem formulations**: Regression & Classification
- 🔧 **Feature engineering** and model comparison
- 📦 **Production-ready** structure with config-driven workflows

---

## 📊 Dataset

**Source**: [Kaggle — Top Spotify Songs 2023](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023)

- **Size**: 953 songs, 24 features
- **Target**: `streams` (transformed to `log_streams` for stability)
- **Features**: 8 numerical audio attributes

### Audio Features Used

| Feature | Description |
|---------|-------------|
| `bpm` | Beats per minute |
| `danceability_%` | How suitable for dancing (0-100) |
| `energy_%` | Intensity and activity (0-100) |
| `valence_%` | Musical positiveness (0-100) |
| `acousticness_%` | Acoustic vs electronic (0-100) |
| `instrumentalness_%` | Vocal vs instrumental (0-100) |
| `liveness_%` | Presence of audience (0-100) |
| `speechiness_%` | Spoken words presence (0-100) |

---

## 🏗️ Project Structure
```
spotify-2023-ml/
├── data/
│   ├── raw/                    # Original dataset (not in git)
│   └── processed/              # Cleaned data
├── configs/
│   ├── baseline.yaml           # Regression config
│   ├── classifier.yaml         # Classification config
│   ├── experiment_1.yaml       # Experiment: 8 features
│   └── experiment_2.yaml       # Experiment: 9 features
├── src/
│   ├── data/
│   │   ├── make_dataset.py              # Regression preprocessing
│   │   └── make_classification_dataset.py  # Classification preprocessing
│   ├── training/
│   │   ├── train.py                     # Regression training
│   │   └── train_classifier.py          # Classification training
│   └── inference/
│       └── predict.py                   # Model inference
├── mlruns/                     # MLflow experiment logs
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
# Option A: Manual download from Kaggle
# Place 'spotify-2023.csv' in data/raw/

# Option B: Using Kaggle CLI (requires kaggle.json)
kaggle datasets download -d nelgiriyewithana/top-spotify-songs-2023 -p data/raw --unzip
```

### 3. Data Processing
```bash
# For regression task
python -m src.data.make_dataset

# For classification task
python -m src.data.make_classification_dataset
```

### 4. Train Models
```bash
# Regression: Predict log(streams)
python -m src.training.train --config configs/baseline.yaml

# Classification: Predict high popularity (top 10%)
python -m src.training.train_classifier --config configs/classifier.yaml
```

### 5. View Experiments
```bash
# Launch MLflow UI
mlflow ui

# Open browser: http://localhost:5000
```

### 6. Make Predictions
```bash
# Create test data: test_input.csv
python -m src.inference.predict --input test_input.csv
```

---

## 📈 Experimental Results

### Regression Task (Predicting Stream Count)

**Objective**: Predict `log_streams` using audio features

| Model | Features | RMSE | R² | Notes |
|-------|----------|------|----|-------|
| Random Forest | 12 features | 0.83 | 0.46 | Baseline with all features |
| Random Forest | 8 features | 1.16 | -0.04 | Audio features only |
| Random Forest | 9 features | 1.14 | -0.02 | + artist_count |

**Key Finding**: Audio features alone have **very weak correlation** (<0.1) with streaming popularity.

### Classification Task (Predicting High Popularity)

**Objective**: Classify songs as "high popularity" (top 10% of streams)

**Data Distribution**:
- High popularity: 96 songs (10.1%)
- Low popularity: 856 songs (89.9%)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Random Forest (30% threshold) | 64.4% | 35.1% | 22.8% | 27.7% | 55.9% |
| Random Forest (10% threshold) | 89.5% | 0% | 0% | 0% | 67.4% |
| Gradient Boosting | 89.5% | 33.3% | 5.3% | 9.1% | 65.9% |
| RF + Feature Engineering | 90.1% | 50.0% | 5.3% | 9.5% | 62.7% |

**Key Finding**: Models struggle to identify popular songs, often defaulting to predicting "low popularity" due to class imbalance.

---

## 🔍 Feature Correlation Analysis

Correlation of audio features with `log_streams`:
```
bpm                 :  0.004  (essentially no correlation)
danceability_%      : -0.068
valence_%           : -0.048
energy_%            : -0.027
acousticness_%      : -0.013
instrumentalness_%  : -0.021
liveness_%          : -0.055
speechiness_%       : -0.097
artist_count        : -0.158  (strongest, but still weak)
released_year       : -0.248  (potential data leakage)
```

**Interpretation**: Audio features show **negligible correlation** with streaming success.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.12 |
| **ML Framework** | scikit-learn |
| **Experiment Tracking** | MLflow |
| **Data Processing** | pandas, numpy |
| **Configuration** | YAML |
| **Version Control** | Git |

---

## 💡 Key Learnings

### What Worked ✅

1. **Complete ML Pipeline**: Implemented full workflow from data → training → inference
2. **Experiment Tracking**: MLflow for reproducible experiments
3. **Multiple Approaches**: Tried regression, classification, and feature engineering
4. **Config-Driven**: Easy to modify hyperparameters without code changes

### What Didn't Work ❌

1. **Poor Predictive Performance**: Audio features alone cannot predict popularity
2. **Class Imbalance**: Classification models default to majority class
3. **Feature Engineering**: Combining weak features doesn't create strong ones

### Root Cause Analysis 🔬

**Why did the models fail?**

1. **Missing Critical Features**
   - Artist fame/follower count
   - Marketing budget
   - Social media virality (TikTok, Instagram)
   - Playlist inclusion
   
2. **Dataset Bias**
   - All songs are already "popular" (on Spotify charts)
   - Limited variance in target variable
   
3. **Low Feature-Target Correlation**
   - Audio features correlate <0.1 with streams
   - True drivers of popularity are external factors

**Analogy**: Trying to predict stock prices using only the company's logo colors — the features simply don't contain the relevant information.

---

## 🚀 Future Improvements

To meaningfully predict song popularity, we would need:

1. **Artist Metadata**
   - Spotify follower count
   - Previous album performance
   - Social media presence
   
2. **Temporal Features**
   - Release timing (holidays, events)
   - Playlist add velocity
   - First-week streaming trends
   
3. **External Data**
   - TikTok challenge participation
   - YouTube music video views
   - Media coverage (news, blogs)

4. **Advanced Techniques**
   - Time series analysis for trend prediction
   - NLP on lyrics
   - Multi-modal learning (audio + text + metadata)

---

## 📂 MLflow Experiment Tracking

All experiments are logged in MLflow with:

- **Parameters**: model type, hyperparameters, feature list
- **Metrics**: RMSE, R², Accuracy, F1, AUC
- **Artifacts**: trained model, feature schema, input examples

**View experiments**:
```bash
mlflow ui
# Navigate to http://localhost:5000
```

**Compare runs**:
1. Select multiple runs (checkbox)
2. Click "Compare" button
3. View parameter/metric differences

---

## 📝 Project Conclusion

This project demonstrates a **complete ML engineering workflow** while highlighting an important lesson: **data quality and feature relevance matter more than model complexity**.

### Achievements 🏆

- ✅ End-to-end ML pipeline implementation
- ✅ Experiment tracking and version control
- ✅ Multiple modeling approaches (regression, classification)
- ✅ Proper evaluation and failure analysis

### Key Takeaway 💡

**Not all problems are solvable with machine learning**. When features have no correlation with the target, even the best algorithms will fail. This project successfully identified that:

1. Audio features alone cannot predict streaming popularity
2. External factors (artist fame, marketing, social virality) dominate
3. The right features are more important than the right algorithm

This is a **valuable learning experience** that reflects real-world ML challenges where feature engineering and data acquisition are often the bottleneck, not model selection.

---

## 🌟 Portfolio Value

This project showcases:

- **ML Engineering**: Complete pipeline from data to deployment
- **Problem-Solving**: Multiple approaches to tackle the problem
- **Critical Thinking**: Identifying why models fail and proposing solutions
- **Production Skills**: Config management, experiment tracking, code structure
- **Communication**: Clear documentation of successes AND failures

---

## 📧 Contact

**Author**: Vance  
**GitHub**: [Yneq](https://github.com/Yneq)  
**Project**: [spotify-2023-ml](https://github.com/Yneq/spotify-2023-ml)

---

## 中文結論 (Chinese Summary)

### 專案目標

使用音樂特徵（節奏、舞曲性、能量等）預測 Spotify 歌曲的流行程度。

### 實驗結果

#### 回歸問題（預測播放次數）
- **最佳模型**: Random Forest
- **RMSE**: 0.83
- **R²**: 0.46
- **結論**: 效果不佳，只能解釋 46% 的變異

#### 分類問題（預測是否高人氣）
- **最佳模型**: Random Forest + 特徵工程
- **準確率**: 90.1%（但這是誤導性的）
- **召回率**: 5.3%（只找到 5% 的高人氣歌曲）
- **F1 分數**: 9.5%
- **結論**: 模型幾乎無法識別高人氣歌曲

### 失敗原因分析

1. **特徵相關性極低**
   - 所有音樂特徵與播放次數的相關性 < 0.1
   - 音樂特徵對流行度的影響微乎其微

2. **缺少關鍵特徵**
   - 真正影響播放次數的因素：
     - ✅ 歌手知名度（粉絲數、過往作品）
     - ✅ 行銷預算（廣告、宣傳）
     - ✅ 社群媒體病毒式傳播（TikTok、Instagram）
     - ✅ 播放清單收錄
   - 我們的資料只有：
     - ❌ 音樂特徵（BPM、舞曲性、能量等）

3. **資料集特性**
   - 資料來源已經是「篩選後的熱門歌曲」
   - 能上榜的歌都已經很紅
   - 差異主要來自外部因素，而非音樂本身

### 學習成果

雖然模型效果不好，但這是一個**非常成功的學習專案**：

#### 技術面 ✅
- 完整的機器學習工作流程（資料處理 → 訓練 → 評估）
- MLflow 實驗追蹤與模型版本管理
- 多種模型與方法嘗試（回歸、分類、特徵工程）
- 專業的專案結構（模組化、設定檔管理）

#### 概念面 ✅
- 理解特徵工程的重要性
- 認識類別不平衡問題
- 學習模型評估指標（RMSE、R²、Accuracy、F1、AUC）
- 體會「資料品質 > 模型複雜度」的真理

#### 實務面 ✅
- 學會使用 Git 版本控制
- 實作完整的資料科學專案
- 撰寫專業的技術文件
- 分析失敗原因並提出改進方向

### 核心領悟 💡

**並非所有問題都能用機器學習解決**

當特徵與目標變數沒有相關性時，即使使用最先進的演算法也無法改善效果。這個專案成功地驗證了：

1. 音樂特徵本身無法預測流行度
2. 外部因素（歌手名氣、行銷、社群傳播）才是關鍵
3. **擁有正確的特徵，比選擇正確的演算法更重要**

這是一個寶貴的學習經驗，反映了真實世界中機器學習專案的挑戰：特徵工程和資料獲取往往是瓶頸，而非模型選擇。

### 專案價值

這個專案展示了：

- ✅ **完整的 ML 工程能力**：從資料到部署的全流程
- ✅ **問題解決能力**：嘗試多種方法應對挑戰
- ✅ **批判性思維**：識別失敗原因並提出解決方案
- ✅ **專業技能**：設定管理、實驗追蹤、程式架構
- ✅ **溝通能力**：清楚記錄成功與失敗

**失敗的實驗也是成功的學習** — 這正是真實機器學習專案的寫照。

---

## 📜 License

MIT License - Feel free to use this project for learning and portfolio purposes.

🚀 執行指令
bash# 1. 建立 README.md（複製上面的內容）
# 用你喜歡的編輯器（VS Code、vim、nano）

# 2. 加入並提交
git add README.md
git add .  # 加入其他所有檔案
git commit -m "Update README with complete project documentation and Chinese summary"

# 3. 推送到 GitHub
git push origin main