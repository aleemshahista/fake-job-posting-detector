"""
predict.py
----------
Provides a single function to classify a job posting as Real or Fake
using the saved best model and TF-IDF vectorizer.
"""

from src.feature_extraction import clean_text, load_vectorizer
from src.train_model import load_model


def predict_job_posting(job_fields):
    """
    Predict whether a job posting is real or fake.

    Parameters
    ----------
    job_fields : dict
        Must contain keys: title, company_profile, description,
        requirements, benefits.  Missing keys default to "".

    Returns
    -------
    dict
        {
            "label"      : "Fake" or "Real",
            "confidence" : float   (probability of the predicted class),
            "probability": float   (probability of being Fake),
        }
    """
    # 1. Combine fields into a single string
    text_keys = ["title", "company_profile", "description",
                 "requirements", "benefits"]
    combined = " ".join(job_fields.get(k, "") for k in text_keys)

    # 2. Clean the text
    cleaned = clean_text(combined)

    # 3. Vectorize with the saved TF-IDF vectorizer
    vectorizer = load_vectorizer()
    tfidf_vector = vectorizer.transform([cleaned])

    # 4. Load model and predict
    model = load_model()
    prediction = model.predict(tfidf_vector)[0]

    # 5. Get probability (some models support predict_proba)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(tfidf_vector)[0]
        fake_probability = proba[1]      # probability of class 1 (Fake)
        confidence = proba[prediction]   # probability of the chosen class
    else:
        fake_probability = float(prediction)
        confidence = 1.0

    label = "Fake" if prediction == 1 else "Real"

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probability": round(fake_probability, 4),
    }


# ---------------------------------------------------------------------------
# Quick command-line test
# ---------------------------------------------------------------------------

def main():
    """Run a sample prediction to verify the pipeline works."""
    sample = {
        "title": "Data Entry Clerk - Work From Home",
        "company_profile": "",
        "description": (
            "We are looking for a data entry clerk to work from home. "
            "No experience needed. Earn $5000 per week! "
            "Send your bank details to get started immediately."
        ),
        "requirements": "No requirements",
        "benefits": "Unlimited income potential",
    }

    print("\n[predict] Running sample prediction ...")
    result = predict_job_posting(sample)

    print(f"\n  Prediction : {result['label']}")
    print(f"  Confidence : {result['confidence']:.2%}")
    print(f"  P(Fake)    : {result['probability']:.2%}")
    print()


if __name__ == "__main__":
    main()
