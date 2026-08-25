# 🏠 House Prices — Advanced Regression Techniques

> Predict residential home sale prices in Ames, Iowa using a deep-learning regression pipeline.

| Detail | Value |
|---|---|
| **Competition** | [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) |
| **Task** | Tabular Regression |
| **Target** | `SalePrice` (continuous, USD) |
| **Training Set** | 1,460 homes × 81 columns |
| **Test Set** | 1,459 homes × 80 columns |
| **Evaluation Metric** | RMSLE (Root Mean Squared Logarithmic Error) |
| **Kaggle Leaderboard Rank** | 🏅 **3,388** |
| **Kaggle Score (RMSLE)** | **1.00545** |

---

## 📂 Project Structure

```
Kaggle_house_prices_Prediction/
├── README.md                  # This file
├── house_prices_dl.ipynb      # Full notebook — EDA, cleaning, modelling, submission
├── train.csv                  # Training data (1,460 rows, 81 cols)
├── test.csv                   # Test data (1,459 rows, 80 cols)
├── sample_submission.csv      # Kaggle-provided submission template
├── data_description.txt       # Official feature documentation
└── Screenshot 2026-08-25 110019.png   # Leaderboard screenshot
```

---

## 🔍 Pipeline Overview

### 1. Exploratory Data Analysis (EDA)

| Step | Details |
|---|---|
| **Missing values** | Identified 19 columns with nulls; heaviest: PoolQC (1,453), MiscFeature (1,406), Alley (1,369), Fence (1,179) |
| **Column types** | 45 categorical + 34 numeric (after reclassifying `MSSubClass` and `MoSold` as categorical) |
| **Categorical EDA** | Count-plots for all 45 categorical features |
| **Numeric EDA** | Histograms for all 32 numeric features |
| **Target distribution** | Histogram of `SalePrice` — right-skewed, motivating log-transform |

### 2. Data Cleaning & Missing Value Imputation

All processing was done on a **combined train + test DataFrame** to ensure consistency.

| Strategy | Columns |
|---|---|
| Fill with `"None"` | PoolQC, MiscFeature, Alley, Fence, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, BsmtExposure, BsmtFinType2, BsmtQual, BsmtCond, BsmtFinType1, MasVnrType |
| Fill with `0` | GarageYrBlt, MasVnrArea, GarageCars, GarageArea, BsmtFullBath, BsmtHalfBath, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF |
| Median (per Neighborhood) | LotFrontage |
| Mode | MSZoning, Utilities, Exterior1st, Exterior2nd, Electrical, KitchenQual, Functional, SaleType |

**Result:** 0 missing values after imputation.

### 3. Data Type Fixes

- `MSSubClass` → `str` (numeric code, not a real quantity)
- `MoSold` → `str` (month sold — categorical, not ordinal)
- `GarageYrBlt` → `int`

### 4. Feature Engineering

| New Feature | Formula |
|---|---|
| `HouseAge` | `YrSold − YearBuilt` (clipped ≥ 0) |
| `YearsSinceRemod` | `YrSold − YearRemodAdd` (clipped ≥ 0) |
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` |

### 5. Encoding & Scaling

- **Categorical encoding:** `LabelEncoder` on all remaining `object`-type columns
- **Numerical scaling:** `StandardScaler` (fit on train, transform both train & test)
- **Target transform:** `log1p(SalePrice)` for training; `expm1` to invert at prediction time

### 6. Train / Validation Split

```
85% train  ·  15% validation  ·  random_state = 42
→ 1,241 train samples  ·  219 validation samples
```

---

## 🧠 Model Architecture

A **Keras Sequential** deep neural network:

```
Input (82 features)
  ↓
Dense(128, relu)
  ↓
Dropout(0.3)
  ↓
Dense(64, relu)
  ↓
Dropout(0.2)
  ↓
Dense(32, relu)
  ↓
Dense(1)  ← linear output (log-price)
```

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (lr = 0.001) |
| Loss | MSE |
| Metrics | MAE |
| Epochs | up to 200 |
| Batch size | 32 |
| Early stopping patience | 20 (restore best weights) |

---

## 📊 Results

| Metric | Value |
|---|---|
| **Val MSE** (log target) | 0.0418 |
| **Val MAE** (log target) | 0.1408 |
| **Kaggle RMSLE** | 1.00545 |
| **Leaderboard Rank** | 3,388 |

---

## 🛠️ Tech Stack

| Component | Library / Tool |
|---|---|
| Data wrangling | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Preprocessing | `scikit-learn` (LabelEncoder, StandardScaler, train_test_split) |
| Deep Learning | `TensorFlow / Keras` |
| Environment | Kaggle Notebooks (Python 3.14) |

---

## 🚀 How to Reproduce

1. **Download the data** from the [competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place `train.csv`, `test.csv`, and `sample_submission.csv` in this directory.
2. **Open** `house_prices_dl.ipynb` in Jupyter / Kaggle / VS Code.
3. **Run all cells** — the notebook is fully self-contained.
4. The final cell writes `submission.csv` ready for Kaggle upload.

---

*"Every leaderboard rank starts with a messy `train.csv` and a stubborn refusal to skip the EDA."* 🚀
