"""
Preprocessing utilities: cleaning, encoding, scaling.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Basic cleaning: drop irrelevant columns, handle missing values, encode target.
    # drop customerID
    if "customerID" in df.columns:
      df = df.drop(columns=["customerID"])  

    # numeric conversion
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

     # encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})


    # one-hot encode all other categorical fields
    df = pd.get_dummies(df, drop_first=True)

    # drop NaNs
    df = df.dropna()
    return df


def split_features(df: pd.DataFrame, target_col: str):
    """Split into X (features) and y (target)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def scale_numeric(X_train, X_test, numeric_cols):
    """Scale numeric columns using StandardScaler."""
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train, X_test, scaler