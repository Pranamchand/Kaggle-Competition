'''
Load the raw Dataset of Disaster tweet . 
- Delete unnecessary column.
- Cleaning text column. 
'''

# --- import libreary ---
import pandas as pd
from src.config import RAW_TRAIN_DIR, RAW_TEST_DIR, PROCESSED_TRAIN_DIR, PROCESSED_TEST_DIR, DATA_DIR
import re
import nltk

from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download("wordnet")


# --- import Data ---
def load_data(path_train = RAW_TRAIN_DIR, path_test = RAW_TEST_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not path_train.exists() or not path_test.exists():
            raise FileNotFoundError(
                f" Couldn't Find The DataSet at {DATA_DIR}"
                f" Download The File at https://www.kaggle.com/competitions/nlp-getting-started/data?select=train.csv"
                f" And Plase It There"
            )
        return pd.read_csv(path_train), pd.read_csv(path_test)


# --- Clean text ---
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df["clean_text"] = df["text"].str.lower()
    df["clean_text"] = df["clean_text"].apply(
        lambda x: re.sub(r"http\S+|www\S+|https\S+", "", x)
    )

    df["clean_text"] = df["clean_text"].str.replace(r"[^\w\s]", "", regex=True)

    df["clean_text"] = df["clean_text"].str.replace(
        r"\d+", "", regex=True
    )

    df["clean_text"] = df["clean_text"].str.strip()
    df["clean_text"] = df["clean_text"].str.replace(r"\s+", " ", regex=True)

    stop_words = set(stopwords.words("english"))

    def remove_stopwords(text):
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return " ".join(words)

    df["clean_text"] = df["clean_text"].apply(remove_stopwords)
    df['text_only'] = df['clean_text'].copy()
    return df

# --- import and clean data ---
def load_and_clean_data(Save: bool= True) -> pd.DataFrame:
    train, test = load_data()

    train = clean_data(train)
    test = clean_data(test)

    if Save:
        train.to_csv(PROCESSED_TRAIN_DIR, index=False)
        test.to_csv(PROCESSED_TEST_DIR, index=False)

    return train, test


if __name__ == "__main__":
    train, test = load_and_clean_data()
    print(f"Train Data Shape: {train.shape}")
    print(f"Test Data Shape: {test.shape}")