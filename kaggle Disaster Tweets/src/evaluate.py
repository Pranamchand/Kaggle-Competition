'''
Viewing the Model performance on the validation set using accuracy, f1 score and classification report.
'''

from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score

# --- Evaluate Model ---
def evaluate_model(model, x_valid, y_valid):

    y_pred = model.predict(x_valid)

    tests = {
        "accuracy": accuracy_score(y_valid, y_pred),
        "precision": precision_score(y_valid, y_pred),
        "recall": recall_score(y_valid, y_pred),
        "f1_score": f1_score(y_valid, y_pred)}

    print("=" * 50)
    print("Final Model Evaluation")
    print("=" * 50)
    for key, value in tests.items():
        print(f"{key} : {value}")