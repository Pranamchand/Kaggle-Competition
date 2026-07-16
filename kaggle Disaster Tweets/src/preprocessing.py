"""
Spliting and vectorissation of train and test Data

"""

# --- import libraries ---
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import TARGET_COLUMN, TRAIN_COLUMN, VECTORISED_PATH

# --- splitting train---
def split_train_data(train):
    x = train[TRAIN_COLUMN]
    y = train[TARGET_COLUMN]

    x_train, x_valid, y_train, y_valid = train_test_split( x, y, test_size=0.2, random_state=42, stratify=y)
    vectorizer  = TfidfVectorizer( max_features=7000,
                                    ngram_range=(1, 2), stop_words = "english")
    return x_train, x_valid, y_train, y_valid, vectorizer

# --- vectorising train data ---
def vectorise_data(x_train, x_valid, vectorizer):
    x_train = vectorizer.fit_transform(x_train)
    x_valid = vectorizer.transform(x_valid)

    return x_train, x_valid

# --- split and vectorise train data ---
def split_and_vectorise_data(train):
    x_train, x_valid, y_train, y_valid, vectorizer = split_train_data(train)
    x_train, x_valid = vectorise_data(x_train, x_valid, vectorizer)

    return x_train, x_valid, y_train, y_valid, vectorizer

