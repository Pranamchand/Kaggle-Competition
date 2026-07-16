'''
Train the final Logistic Regression model on the entire training dataset and save the model to disk.
'''

# --- Import Libraries ---
from sklearn.linear_model import LogisticRegression
from src.config import PARAM_LR, MODEL_PATH
import joblib

def train_final_model(x_train, y_train, Save:bool = True):
    model = LogisticRegression(**PARAM_LR)
    model.fit(x_train, y_train)

    if Save:
        joblib.dump(model, MODEL_PATH)
    return model
