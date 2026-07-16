'''
make submission.csv file for kaggle competition
'''

import pandas as pd
import joblib
from src.config import VECTORISED_PATH
from src.config import TRAIN_COLUMN, SUBMISSION_PATH

# --- predict target feature on test Data and submit ---
def submit_predictions(model, test, vectorizer):
    X_test = test[TRAIN_COLUMN]
    X_test = vectorizer.transform(X_test)

    predictions = model.predict(X_test)
    joblib.dump(vectorizer, VECTORISED_PATH)

    # --- making submission Dataframe --- 
    submission = pd.DataFrame({
        "id": test["id"],
        "target": predictions
    })

    # --- save submission.csv --- 
    submission.to_csv(SUBMISSION_PATH, index=False)







