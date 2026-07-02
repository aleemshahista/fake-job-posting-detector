import pandas as pd

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def load_dataset():
    """
    Load the raw dataset.
    """
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def clean_dataset(df):
    """
    Basic preprocessing of the dataset.
    """

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Fill missing values in text columns
    text_columns = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits"
    ]

    for column in text_columns:
        df[column] = df[column].fillna("")

    return df


def save_dataset(df):
    """
    Save cleaned dataset.
    """
    df.to_csv(PROCESSED_DATA_PATH, index=False)


def main():
    df = load_dataset()
    df = clean_dataset(df)
    save_dataset(df)

    print("Dataset cleaned successfully!")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()