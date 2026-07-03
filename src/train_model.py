"""
train_model.py
--------------
Trains Logistic Regression, Multinomial Naive Bayes, and Random Forest on
TF-IDF features.  Compares them on F1-score and saves the best model.
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.config import PROCESSED_DATA_PATH, MODEL_PATH
from src.feature_extraction import combine_text_columns, clean_text, fit_tfidf, transform_tfidf


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_processed_data():
    """
    Load the cleaned CSV and return features (combined text) and labels.

    Returns
    -------
    X : pd.Series   – cleaned, combined text for every row.
    y : pd.Series   – binary target (0 = real, 1 = fake).
    """
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Combine the five text columns into one string per row
    df["combined_text"] = combine_text_columns(df)

    # Clean every combined text string
    df["combined_text"] = df["combined_text"].apply(clean_text)

    X = df["combined_text"]
    y = df["fraudulent"]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Stratified train / test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models():
    """
    Return a dictionary of model_name → model_instance.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }
    return models


# ---------------------------------------------------------------------------
# Training & comparison
# ---------------------------------------------------------------------------

def train_and_compare(X_train_tfidf, X_test_tfidf, y_train, y_test):
    """
    Train every model, compute F1-score on the test set, and return
    a dict of results together with the best model object.

    Returns
    -------
    results : dict
        {model_name: {"model": fitted_model, "f1": float}}
    best_name : str
    best_model : fitted estimator
    """
    models = get_models()
    results = {}

    print("\n" + "=" * 60)
    print("  MODEL TRAINING & COMPARISON")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n> Training {name} ...")
        model.fit(X_train_tfidf, y_train)

        y_pred = model.predict(X_test_tfidf)
        score = f1_score(y_test, y_pred)

        results[name] = {"model": model, "f1": score}
        print(f"  F1-score (test): {score:.4f}")

    # Pick the winner
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]["model"]

    print("\n" + "-" * 60)
    print(f"  BEST MODEL: {best_name}  (F1 = {results[best_name]['f1']:.4f})")
    print("-" * 60)

    return results, best_name, best_model


def save_model(model):
    """Save the best model to the path defined in config."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n[train_model] Best model saved --> {MODEL_PATH}")


def load_model():
    """Load the saved best model from disk."""
    model = joblib.load(MODEL_PATH)
    print(f"[train_model] Model loaded <-- {MODEL_PATH}")
    return model


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main():
    """Full training pipeline: load -> split -> vectorize -> train -> save."""
    # 1. Load data
    print("[train_model] Loading processed data ...")
    X, y = load_processed_data()
    print(f"  Total samples : {len(X)}")
    print(f"  Fraudulent (1): {y.sum()}  |  Real (0): {(y == 0).sum()}")

    # 2. Stratified split
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"\n  Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # 3. TF-IDF — fit on train, transform test
    X_train_tfidf, vectorizer = fit_tfidf(X_train)
    X_test_tfidf = transform_tfidf(X_test, vectorizer)

    # 4. Train all models and compare
    results, best_name, best_model = train_and_compare(
        X_train_tfidf, X_test_tfidf, y_train, y_test
    )

    # 5. Save the best model
    save_model(best_model)

    print("\n[OK] Training pipeline complete.\n")


if __name__ == "__main__":
    main()
