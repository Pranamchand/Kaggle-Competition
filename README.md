## 📌 Repository Overview

This repository is a collection of five **Kaggle "Getting Started" competition** solutions, each built as a complete, self-contained ML pipeline — from raw data to a submitted `submission.csv`. The notebooks follow a consistent, disciplined workflow: explore the data first, engineer features that actually matter, benchmark multiple models honestly, and tune where appropriate.

|  #  | Project                                                                  | Task Type                             | Kaggle Leaderboard Rank |
| :-: | :----------------------------------------------------------------------- | :------------------------------------ | :---------------------: |
|  1  | 🌪️ [Disaster Tweets Classification](#-1-disaster-tweets-classification) | Binary NLP Classification             |        🏅 **628**       |
|  2  | 🚀 [Spaceship Titanic Prediction](#-2-spaceship-titanic-prediction)      | Binary Tabular Classification         |       🏅 **1733**       |
|  3  | 🔢 [Digit Recognizer (MNIST)](#-3-digit-recognizer-mnist-classification) | Multiclass Image Classification (CNN) |        🏅 **667**       |
|  4  | 🏪 [Store Sales Time Series Forecasting](#-4-store-sales-time-series-forecasting) | Time Series Regression | 🏅 **1022** |
|  5  | 🏠 [House Prices Prediction](#-5-house-prices--advanced-regression-techniques) | Tabular Regression (Deep Learning) | 🏅 **3388** |

`🧮 3,060,654+ training records analyzed`  •  `🤖 13+ models/approaches explored`  •  `🔬 2+ GridSearchCV tuning runs`  •  `🎯 5 leaderboard submissions`

## 🌪️ 1. Disaster Tweets Classification

**Goal:** Given a tweet, predict whether it is describing a real disaster (`target = 1`) or not (`target = 0`) — a classic NLP text-classification problem on the [Kaggle "Natural Language Processing with Disaster Tweets"](https://www.kaggle.com/competitions/nlp-getting-started) dataset (7,613 training tweets, 3,263 test tweets).

## 🚀 2. Spaceship Titanic Prediction

**Goal:** Predict whether a passenger aboard the *Spaceship Titanic* was transported to an alternate dimension after the ship's collision with a spacetime anomaly — a tabular binary classification problem on the [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) dataset (8,693 training passengers).

## 🔢 3. Digit Recognizer (MNIST) Classification

**Goal:** Given a 28×28 grayscale image of a handwritten digit (flattened into 784 pixel columns), predict which digit (0–9) it represents — a classic multiclass image-classification problem on the [Kaggle "Digit Recognizer"](https://www.kaggle.com/competitions/digit-recognizer) dataset (42,000 training images, 28,000 test images).

**Approach:** Pixel values were scaled to a 0–1 range and reshaped from flat 784-length vectors into 28×28×1 image tensors, with labels one-hot encoded across 10 classes. A **Convolutional Neural Network** built in Keras — two `Conv2D` + `MaxPooling2D` blocks followed by a dense layer with dropout for regularization and a softmax output layer — was trained for 5 epochs, reaching **98.60% accuracy** on the held-out validation split.

## 🏪 4. Store Sales Time Series Forecasting

**Goal:** Predict daily sales for stores and product families using historical sales data and additional information such as promotions, store details, oil prices, holidays, and transactions — a time series regression problem on the [Kaggle "Store Sales - Time Series Forecasting"](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) dataset.

**Dataset:** 3,000,888 training records and 28,512 test records.

**Approach:** The project included data cleaning, date-based feature engineering, merging multiple datasets, handling missing oil prices, holiday feature engineering, categorical encoding, and numerical feature scaling. **Linear Regression** and **XGBoost Regressor** were evaluated, with XGBoost performing substantially better.


## 🏠 5. House Prices — Advanced Regression Techniques

**Goal:** Given 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, predict the final sale price of each home — a tabular regression problem on the [Kaggle "House Prices - Advanced Regression Techniques"](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) dataset (1,460 training homes, 1,459 test homes).

**Approach:** The project included thorough EDA (countplots for 45 categorical columns and histograms for 32 numeric columns), aggressive missing-value imputation ("None" fills for absent features like PoolQC/Alley/Fence, median-per-neighborhood for LotFrontage, mode fills for MSZoning/Electrical, etc.), semantic dtype fixes (MSSubClass and MoSold cast to categorical), and feature engineering (HouseAge, YearsSinceRemod, TotalSF). All categorical features were label-encoded and numerics scaled with StandardScaler. A **Keras deep neural network** — four Dense layers (128 → 64 → 32 → 1) with ReLU activations, Dropout regularization (0.3 / 0.2), Adam optimizer (lr = 0.001), and EarlyStopping (patience 20) — was trained on log-transformed sale prices for up to 200 epochs, achieving a **validation MSE of 0.0418** on the log target.

*"Every leaderboard rank starts with a messy* `train.csv` *and a stubborn refusal to skip the EDA."* 🚀