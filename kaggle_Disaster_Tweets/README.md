<div align="center">

# 🌪️ Disaster Tweet Classification
### An End-to-End, Production-Style NLP Pipeline for Real-Time Disaster Detection

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=23&duration=3000&pause=1000&color=FF4B4B&center=true&vCenter=true&width=750&lines=Modular+Python+Architecture;TF-IDF+Feature+Engineering;Logistic+Regression+%7C+LinearSVC+%7C+XGBoost;One+Command+%E2%86%92+Full+Pipeline+%E2%86%92+Kaggle+Submission" />

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F3D?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-EC0000?style=for-the-badge&logo=xgboost&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

</div>



## 📌 Working link
[🔗 click here to view app](https://disaster-tweets-pranamchand.streamlit.app/)




## 📌 Project Overview

This repository solves the Kaggle **"Natural Language Processing with Disaster Tweets"** competition — classifying whether a tweet describes a **real disaster** (`1`) or **not** (`0`).

What started as a single exploratory notebook has been **refactored into a fully modular, production-style Python pipeline**. Instead of running cells top-to-bottom in a notebook, the entire workflow — data loading, cleaning, vectorization, training, evaluation, and Kaggle submission — is now orchestrated by a single `main.py` entry point, with each responsibility cleanly separated into its own module inside `src/`.

> 🔁 Run one script. Get a trained model, evaluation report, and a ready-to-submit `submission.csv` — automatically.

---

## ✨ What Makes This Version Different

| Before (Notebook) | Now (Modular Pipeline) |
|---|---|
| One long `.ipynb` file | Clean `src/` package with single-responsibility modules |
| Manual cell-by-cell execution | Single command: `python main.py` |
| Hardcoded values scattered everywhere | Centralized `config.py` |
| Model/vectorizer lost after kernel restart | Persisted as `model.pkl` / `vectoriser.pkl` via `models/` |
| Hard to reuse or test | Each stage is an importable, testable function |
| Notebook kept for EDA only | Notebook preserved separately for exploratory analysis, pipeline handles production logic |

---

## 🚀 Pipeline Architecture

Running `main.py` triggers a fully automated 5-stage pipeline:

```text
┌────────────────────────────┐
│   [1/5] Load Dataset       │   src/data_loader.py
│   load_and_clean_data()    │   → loads raw train/test CSVs, cleans text, saves processed data
└──────────────┬─────────────┘
               ▼
┌────────────────────────────┐
│   [2/5] Preprocessing      │   src/preprocessing.py
│   split_and_vectorise_data()│  → train/validation split + TF-IDF vectorization
└──────────────┬─────────────┘
               ▼
┌────────────────────────────┐
│   [3/5] Model Training     │   src/model_train.py
│   train_final_model()      │   → fits the final tuned Logistic Regression model
└──────────────┬─────────────┘
               ▼
┌────────────────────────────┐
│   [4/5] Evaluation         │   src/evaluate.py
│   evaluate_model()         │   → Accuracy, Precision, Recall, F1 on validation set
└──────────────┬─────────────┘
               ▼
┌────────────────────────────┐
│   [5/5] Submission         │   src/submission.py
│   submit_predictions()     │   → predicts on test set, writes Kaggle-ready submission.csv
└────────────────────────────┘
```

Each stage prints a live progress indicator (`[1/5]`, `[2/5]` …) to the console, so the entire run is transparent from start to finish.

---

## 📂 Repository Structure

```text
📦 Disaster-Tweet-Classification
│
├── main.py                      # Pipeline orchestrator — single entry point
├── requirements.txt              # Project dependencies
├── README.md
│
├── src/                          # Core pipeline package
│   ├── __init__.py
│   ├── config.py                 # Centralized paths, params & constants
│   ├── data_loader.py             # Load + clean raw data
│   ├── preprocessing.py           # Train/valid split + TF-IDF vectorization
│   ├── model_train.py             # Model training logic
│   ├── evaluate.py                # Metric computation & reporting
│   └── submission.py              # Test prediction + Kaggle submission file
│
├── data/
│   ├── raw/                       # Original Kaggle train.csv / test.csv
│   ├── processed/                 # Cleaned, pipeline-ready data
│   └── submission/                # Generated submission.csv files
│
├── models/
│   ├── model.pkl                  # Serialized trained model
│   └── vectoriser.pkl             # Serialized fitted TF-IDF vectorizer
│
├── notebook/
│   └── Disaster Tweets Notebook.ipynb   # Original EDA & experimentation notebook
│
└── screenshort/                   # Reference screenshots
```

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/Pranamchand/Disaster-Tweet-Classification.git
cd Disaster-Tweet-Classification

# 2. Create and activate a virtual environment
python -m venv myenv
myenv\Scripts\activate      # Windows
source myenv/bin/activate    # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the entire pipeline — from raw data to Kaggle submission — with a single command:

```bash
python main.py
```

**Console output:**

```text
[1/5] Loading dataset...
[2/5] Preprocessing data...
[3/5] Training Logistic Regression model...
[4/5] Evaluating model...
[5/5] Submitting predictions...
```

At the end of the run:
- ✅ A trained model & vectorizer are saved to `models/`
- ✅ Validation metrics are printed to the console
- ✅ A Kaggle-ready `submission.csv` is generated in `data/submission/`

---

## 🔍 Exploratory Data Analysis

The `notebook/` directory preserves the original research phase, covering:

- Missing Value Analysis
- Target Class Distribution
- Tweet Length Analysis
- Keyword Frequency & Disaster Rate by Keyword
- Disaster Rate by Location
- Word Frequency Visualization

These insights directly informed the cleaning and feature-engineering decisions built into `src/data_loader.py` and `src/preprocessing.py`.

---

## 🧹 Text Preprocessing Pipeline

Handled inside `src/data_loader.py` and `src/preprocessing.py`:

- Lowercasing text
- URL removal
- Punctuation stripping
- Numeric character removal
- Extra whitespace normalization
- Stopword removal
- Porter Stemming

---

## 🧠 Feature Engineering — TF-IDF Vectorization

Implemented in `src/preprocessing.py` via `split_and_vectorise_data()`:

- Unigram & Bigram feature extraction
- Maximum feature cap for dimensionality control
- Vocabulary learned **only** from training data (no leakage into validation/test)
- Fitted vectorizer serialized to `models/vectoriser.pkl` for reuse at inference time

---

## 🤖 Models Implemented

| Model | Purpose |
|---|---|
| **Logistic Regression** | Final tuned production model (`train_final_model`) |
| Linear Support Vector Classifier (LinearSVC) | Benchmark comparison |
| Multinomial Naive Bayes | Benchmark comparison |
| Random Forest Classifier | Benchmark comparison |
| XGBoost Classifier | Benchmark comparison |

---

## 🎯 Hyperparameter Tuning

- **Method:** `GridSearchCV`
- **Tuned Models:** Logistic Regression, LinearSVC
- **Optimization Metric:** F1 Score
- **Cross-Validation:** 5-Fold Stratified CV

The winning configuration is what `src/model_train.py` trains as the final model in the pipeline.

---

## 📈 Model Evaluation

`src/evaluate.py` computes and reports, on the held-out validation split:

- Accuracy
- Precision
- Recall
- F1 Score

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| NLP | NLTK |
| Feature Engineering | TF-IDF (Scikit-Learn) |
| Machine Learning | Scikit-Learn, XGBoost |
| Model Selection | GridSearchCV |
| Serialization | Pickle |
| Environment | Virtualenv (`myenv`) |

---

## 📚 Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
xgboost
```

---

<div align="center">

## ⭐ Thank You for Checking Out This Project!

<img src="https://capsule-render.vercel.app/api?type=waving&color=FF4B4B&height=120&section=footer"/>

</div>