'''
End to End pipelne for Disaster Tweets prediction using NLP and ML.
- Load the raw Dataset of Disaster tweet .
- Cleaning text column.
- Split the train data into train and validation set.
- Vectorise the train and validation set using TFIDF vectoriser.
- Train the model using Logistic Regression.
- Evaluate the model performance on the validation set using accuracy, f1 score, recall score and precision score.

Run command : python main.py

'''

import pandas as pd
from src.data_loader import load_and_clean_data
from src.preprocessing import split_and_vectorise_data 
from src.model_train import train_final_model
from src.evaluate import evaluate_model
from src.submission import submit_predictions

def main():

    # -------------------------------------------------
    # Step 1 : Load Dataset
    # -------------------------------------------------
    print("\n[1/5] Loading dataset...")
    train, test = load_and_clean_data(Save=True)

    # -------------------------------------------------
    # Step 2 : Data Preprocessing
    # -------------------------------------------------
    print("\n[2/5] Preprocessing data...")
    x_train, x_valid, y_train, y_valid, vectorizer = split_and_vectorise_data(train)


    # -------------------------------------------------
    # Step 3 : Train Model
    # -------------------------------------------------
    print("\n[3/4] Training Logistic Regression model...")
    model = train_final_model(x_train, y_train)


    # -------------------------------------------------
    # Step 4 : Evaluate Model
    # -------------------------------------------------
    print("\n[4/5] Evaluating model...")
    evaluate_model(model, x_valid, y_valid)

    # -------------------------------------------------
    # Step 5 : Submit Predictions
    # -------------------------------------------------
    print("\n[5/5] Submitting predictions...")
    submit_predictions(model, test, vectorizer)


if __name__ == "__main__":
    main()

