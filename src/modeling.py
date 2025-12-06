"""
Model training utilities.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def train_logistic(X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced")
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """Train a Support Vector Machine classifier."""
    model = SVC(kernel="rbf", probability=True, class_weight="balanced")
    model.fit(X_train, y_train)
    return model
