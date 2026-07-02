from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "fake_job_postings.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_data.csv"

# Model paths
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"