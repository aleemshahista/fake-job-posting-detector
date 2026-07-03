"""
feature_extraction.py
---------------------
Handles text combination, cleaning, and TF-IDF vectorization.
"""

import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import VECTORIZER_PATH


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def combine_text_columns(df):
    """
    Merge the key text columns into a single 'combined_text' column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: title, company_profile, description,
        requirements, benefits.

    Returns
    -------
    pd.Series
        A Series where each row is the concatenation of the five text fields.
    """
    text_columns = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits",
    ]

    # Fill any remaining NaN values with empty strings
    for col in text_columns:
        df[col] = df[col].fillna("")

    combined = df[text_columns].apply(lambda row: " ".join(row), axis=1)
    return combined


def clean_text(text):
    """
    Apply basic cleaning to a single text string.

    Steps:
        1. Lower-case the text.
        2. Remove HTML tags.
        3. Remove URLs.
        4. Remove non-alphabetic characters (keep spaces).
        5. Collapse multiple spaces into one.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Cleaned text.
    """
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)           # strip HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text) # strip URLs
    text = re.sub(r"[^a-z\s]", " ", text)         # keep only letters & spaces
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace
    return text


# ---------------------------------------------------------------------------
# TF-IDF fitting / transforming
# ---------------------------------------------------------------------------

def fit_tfidf(texts, max_features=5000):
    """
    Fit a TfidfVectorizer on the given texts and save it to disk.

    Parameters
    ----------
    texts : iterable of str
        Training-set texts (already cleaned).
    max_features : int, optional
        Maximum number of features (default 5000).

    Returns
    -------
    tfidf_matrix : sparse matrix
        TF-IDF feature matrix for the input texts.
    vectorizer : TfidfVectorizer
        The fitted vectorizer object.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    save_vectorizer(vectorizer)
    print(f"[feature_extraction] TF-IDF vectorizer fitted — "
          f"vocabulary size: {len(vectorizer.vocabulary_)}")

    return tfidf_matrix, vectorizer


def transform_tfidf(texts, vectorizer=None):
    """
    Transform texts using an already-fitted TF-IDF vectorizer.

    Parameters
    ----------
    texts : iterable of str
        Texts to transform (already cleaned).
    vectorizer : TfidfVectorizer or None
        If None, the saved vectorizer is loaded from disk.

    Returns
    -------
    sparse matrix
        TF-IDF feature matrix.
    """
    if vectorizer is None:
        vectorizer = load_vectorizer()

    return vectorizer.transform(texts)


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_vectorizer(vectorizer):
    """Save the fitted TfidfVectorizer to the path defined in config."""
    VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[feature_extraction] Vectorizer saved --> {VECTORIZER_PATH}")


def load_vectorizer():
    """Load a previously saved TfidfVectorizer from disk."""
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"[feature_extraction] Vectorizer loaded <-- {VECTORIZER_PATH}")
    return vectorizer
