'''
Place for Storing file PATH, CONSTANTS and HYPERPARAMETERS
'''

# --- paths ---
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data' / 'raw'
RAW_TRAIN_DIR = ROOT_DIR / 'data' / 'raw' / 'train.csv'
RAW_TEST_DIR = ROOT_DIR / 'data' / 'raw' / 'test.csv'
PROCESSED_TRAIN_DIR = ROOT_DIR / 'data' / 'processed' / 'processed_train.csv'
PROCESSED_TEST_DIR = ROOT_DIR / 'data' / 'processed' / 'processed_test.csv'
MODEL_PATH = ROOT_DIR / 'models' / 'model.pkl'
VECTORISED_PATH = ROOT_DIR / 'models' / 'vectoriser.pkl'
SUBMISSION_PATH = ROOT_DIR / 'data' / 'submission' / 'submission.csv'

# --- target variable ---
TARGET_COLUMN = 'target'
TRAIN_COLUMN = 'text_only'

# --- parameter for Logistic Regression ---
PARAM_LR = {'max_iter': 1000, 'random_state': 42}
