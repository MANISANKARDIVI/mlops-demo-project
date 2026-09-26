import pandas as pd


def load_data(path):
    """Load CSV and ensure required columns exist."""
    df = pd.read_csv(path)

    if "quality" not in df.columns:
        raise ValueError("CSV must contain a 'quality' column.")

    return df


def features_and_target(df):
    """Return X (features) and y (target), excluding the ID column."""
    columns_to_drop = ["quality"]

    if "Id" in df.columns:
        columns_to_drop.append("Id")

    X = df.drop(columns=columns_to_drop)
    y = df["quality"]

    return X, y
