"""
Evaluation utilities.

- compare_models: cross-validated comparison on the TRAIN set (model selection)
- plot_oof_roc_grid: out-of-fold ROC curves on the TRAIN set, one panel per model
- evaluate_model / plot_roc: final evaluation on the TEST set (use once, on the
  model already chosen)
"""

import math

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict

DECIMALS = 4  # decimals shown in tables, reports and plot legends

SCORING = {
    "accuracy": "accuracy",
    "recall": "recall",           # class 1 (churn)
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def make_cv(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """Stratified K-fold shared by every cross-validated step."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def compare_models(candidates: dict, X_train, y_train,
                   n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Cross-validate every candidate on the training set.

    Returns a table with the mean and std of each metric across folds.
    The test set is never touched here.
    """
    cv = make_cv(n_splits, random_state)
    rows = []
    for name, model in candidates.items():
        res = cross_validate(model, X_train, y_train, cv=cv,
                             scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for metric in SCORING:
            scores = res[f"test_{metric}"]
            row[metric] = scores.mean()
            row[f"{metric}_std"] = scores.std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").round(DECIMALS)


def plot_oof_roc_grid(candidates: dict, X_train, y_train,
                      n_splits: int = 5, random_state: int = 42, ncols: int = 2):
    """ROC curves from out-of-fold predictions on the training set.

    Every client is predicted by a model that did not see it during training.
    """
    cv = make_cv(n_splits, random_state)
    nrows = math.ceil(len(candidates) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()

    for ax, (name, model) in zip(axes, candidates.items()):
        y_prob = cross_val_predict(model, X_train, y_train, cv=cv,
                                   method="predict_proba", n_jobs=-1)[:, 1]
        _roc_on_ax(y_train, y_prob, name, ax, plot_chance_level=True)
        ax.set_title(name)
        ax.grid(alpha=0.3)

    for ax in axes[len(candidates):]:
        ax.set_visible(False)

    fig.suptitle("ROC curves (out-of-fold predictions on train)", fontsize=14)
    fig.tight_layout()
    plt.show()


def _roc_on_ax(y_true, y_prob, name, ax=None, plot_chance_level=False):
    """Draw a ROC curve with the AUC in the legend using DECIMALS decimals.

    RocCurveDisplay.from_predictions always writes the AUC with 2 decimals,
    so the label is built here and roc_auc is not passed to the display.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    label = f"{name} (AUC = {auc:.{DECIMALS}f})"
    return RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=None, name=label).plot(
        ax=ax, plot_chance_level=plot_chance_level
    )


def evaluate_model(model, X_test, y_test):
    """Print accuracy, ROC-AUC, classification report and confusion matrix."""
    preds = model.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, preds), DECIMALS))
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), DECIMALS))
    print(classification_report(y_test, preds, digits=DECIMALS))
    print(confusion_matrix(y_test, preds))


def plot_roc(model, X_test, y_test, name="Model", ax=None):
    """Plot the ROC curve on the test set if probabilities are available."""
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        _roc_on_ax(y_test, y_prob, name, ax)
        if ax is None:
            plt.show()
    else:
        print("This model does not support probability predictions.")