"""
Evaluation utilities: accuracy, report, confusion matrix.
"""

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, RocCurveDisplay
)
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """Print accuracy and classification report."""
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))


def plot_roc(model, X_test, y_test, name="Model"):
    """Plot ROC curve if probabilities are available."""
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, y_prob, name=name)
        plt.show()
    else:
        print("This model does not support probability predictions.")
        