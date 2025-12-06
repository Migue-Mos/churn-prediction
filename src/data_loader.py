"""
Simple utilities for loading and inspecting datasets.
"""

import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def basic_info(df: pd.DataFrame):
    """Print basic dataset stats."""
    print(df.head())
    print(df.info())
    print(df.describe())