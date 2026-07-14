<div align="center">

# 🌪️ Disaster Tweet Classification using NLP
### Predicting Real Disaster Tweets with Traditional Machine Learning

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&duration=3000&pause=1000&color=FF4B4B&center=true&vCenter=true&width=700&lines=Natural+Language+Processing;TF-IDF+Feature+Engineering;Text+Classification;Logistic+Regression+%7C+LinearSVC+%7C+XGBoost" />

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F3D?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

</div>

---

# 📌 Project Overview

This project focuses on solving the **Natural Language Processing with Disaster Tweets** Kaggle competition.

The objective is to classify whether a tweet refers to a **real disaster** or **not** using Natural Language Processing (NLP) techniques and classical Machine Learning models.

The project follows a complete end-to-end NLP pipeline, including exploratory data analysis, text preprocessing, TF-IDF feature engineering, model comparison, hyperparameter tuning, and Kaggle submission generation.

---

# 🚀 Workflow

```text
Raw Tweets
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Text Cleaning
     │
     ▼
TF-IDF Vectorization
     │
     ▼
Train / Validation Split
     │
     ▼
Machine Learning Models
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Prediction
     │
     ▼
Kaggle Submission
```

---

# 📂 Dataset

- Source: Kaggle
- Competition: **Natural Language Processing with Disaster Tweets**

Dataset contains:

- Tweet ID
- Keyword
- Location
- Tweet Text
- Target Label
    - **1 → Real Disaster**
    - **0 → Non-Disaster**

---

# 🔍 Exploratory Data Analysis

Performed extensive EDA including:

- Missing Value Analysis
- Target Distribution
- Tweet Length Analysis
- Keyword Frequency
- Disaster Rate by Keyword
- Disaster Rate by Location
- Word Frequency Visualization

---

# 🧹 Text Preprocessing Pipeline

The following preprocessing steps were applied:

- Convert text to lowercase
- Remove URLs
- Remove punctuation
- Remove numeric characters
- Remove extra whitespaces
- Remove stopwords
- Apply Porter Stemming

---

# ⚙️ Feature Engineering

Text data was transformed into numerical vectors using

## TF-IDF Vectorizer

Configuration included:

- TF-IDF Encoding
- Unigram & Bigram Features
- Maximum Feature Selection
- Vocabulary Learning from Training Data only

---

# 🤖 Models Implemented

The following machine learning algorithms were evaluated:

- Logistic Regression
- Linear Support Vector Classifier (LinearSVC)
- Multinomial Naive Bayes
- Random Forest Classifier
- XGBoost Classifier

---

# 🎯 Hyperparameter Tuning

GridSearchCV was applied for model optimization.

Tuned models include:

- Logistic Regression
- Linear Support Vector Classifier

Evaluation Metric:

- **F1 Score**

Cross Validation:

- **5-Fold Stratified Cross Validation**

---

# 📈 Model Evaluation 

Models were evaluated using

- Accuracy
- Precision
- Recall
- F1 Score

---

# 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| NLP | NLTK |
| Feature Engineering | TF-IDF |
| Machine Learning | Scikit-Learn, XGBoost |
| Model Selection | GridSearchCV |

---

# 📁 Repository Structure

```text
📦 Disaster-Tweet-Classification
│
├── Disaster Tweets Notebook.ipynb
├── submission.csv
├── README.md
└── requirements.txt
```

---

# 📊 Machine Learning 

```python
Tweet Text
    ↓
Text Cleaning
    ↓
Stopword Removal
    ↓
Stemming
    ↓
TF-IDF Vectorizer
    ↓
Train/Test Split
    ↓
Model Training
    ↓
Hyperparameter Tuning
    ↓
Prediction
```


---

# 📚 Libraries

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
xgboost
re
```

---

<div align="center">

## ⭐ Thank You!

<img src="https://capsule-render.vercel.app/api?type=waving&color=FF4B4B&height=120&section=footer"/>

</div>