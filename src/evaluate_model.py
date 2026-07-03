"""
evaluate_model.py
-----------------
Loads the saved best model and TF-IDF vectorizer, recreates the test split,
and prints a full evaluation report (Accuracy, Precision, Recall, F1-score,
Confusion Matrix, Classification Report).
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.train_model import load_processed_data, split_data, load_model
from src.feature_extraction import load_vectorizer, transform_tfidf


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model, X_test_tfidf, y_test):
    """
    Compute and print all evaluation metrics.

    Parameters
    ----------
    model : fitted estimator
    X_test_tfidf : sparse matrix
        TF-IDF-transformed test features.
    y_test : array-like
        True labels.
    """
    y_pred = model.predict(X_test_tfidf)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION REPORT")
    print("=" * 60)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")

    print("\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]}   FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}   TP={cm[1][1]}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main():
    """Recreate the test split, load model & vectorizer, and evaluate."""
    # 1. Reload data & recreate the same stratified split
    X, y = load_processed_data()
    _, X_test, _, y_test = split_data(X, y)

    # 2. Load saved vectorizer & transform test text
    vectorizer = load_vectorizer()
    X_test_tfidf = transform_tfidf(X_test, vectorizer)

    # 3. Load best model
    model = load_model()

    # 4. Evaluate
    evaluate(model, X_test_tfidf, y_test)

    print("[OK] Evaluation complete.\n")


if __name__ == "__main__":
    main()
