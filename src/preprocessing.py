"""
Preprocessing utilities: row-wise cleaning and the column preprocessor.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise cleaning that does not learn anything from the data.

    - drop customerID (unique identifier, no predictive value)
    - convert TotalCharges to numeric (blank strings become NaN)
    - encode the target Churn as 0/1
    - drop rows with missing values

    Scaling and one-hot encoding of the features are NOT done here: they live
    inside the sklearn Pipeline (see build_preprocessor), so they are fitted
    only on training data.
    """
    df = df.copy()
    df = df.drop(columns="customerID", errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.dropna()
    return df


def split_features(df: pd.DataFrame, target_col: str):
    """Split into X (features) and y (target)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scale numeric columns and one-hot encode the rest.

    Column types are detected automatically, so SeniorCitizen (int64) is
    treated as numeric, exactly as in the modeling notebook.
    """
    numeric_features = X.select_dtypes(include="number").columns
    categorical_features = X.select_dtypes(exclude="number").columns

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )