# Telco Customer Churn Prediction

Binary classification project to predict which telecom customers are likely to cancel their contract (churn), built with scikit-learn.

The focus is on a **leakage-safe, reproducible workflow**: preprocessing lives inside a scikit-learn `Pipeline`, models are compared with cross-validation on the training set only, and the test set is used once, on the final model.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset): 7,043 customers and 21 columns (demographics, services contracted, account information and the `Churn` label). About 26 % of customers churn, so the classes are imbalanced.

Cleaning applied (see `src/preprocessing.py`):

- `customerID` is dropped: it is a unique identifier with no predictive value, and one-hot encoding it would add ~7,000 noise columns.
- `TotalCharges` is converted to numeric. Eleven rows with blank values (customers with `tenure = 0`) are removed, leaving 7,032 rows.
- `Churn` is encoded as 0/1.

## Project structure

```
.
├── data/                  # Telco-Customer-Churn.csv goes here (not tracked)
├── models/                # saved pipeline (not tracked)
├── notebooks/
│   ├── 01_EDA.ipynb       # exploratory analysis and cleaning
│   └── 02_modeling.ipynb  # model comparison and final evaluation
├── src/
│   ├── data_loader.py     # load and inspect data
│   ├── preprocessing.py   # cleaning and column preprocessor
│   ├── modeling.py        # model pipelines and model saving
│   └── evaluation.py      # cross-validation, ROC curves, test metrics
├── main.py                # end-to-end script
└── requirements.txt
```

## Methodology

1. **Split first.** Stratified 80/20 train/test split (`random_state=42`) before any fitting.
2. **Preprocessing inside the pipeline.** A `ColumnTransformer` scales numeric columns (`StandardScaler`) and one-hot encodes categorical ones (`OneHotEncoder`). Because it is part of each `Pipeline`, it is fitted only on training data, and again inside every cross-validation fold.
3. **Model selection with cross-validation.** Four candidates (logistic regression, random forest, SVM, XGBoost) are compared with 5-fold stratified cross-validation on the training set. Class imbalance is handled with `class_weight="balanced"` (`scale_pos_weight` for XGBoost) so the models are comparable.
4. **Final evaluation once.** The chosen model is trained on the full training set and evaluated a single time on the test set. The saved file contains the whole pipeline, preprocessing included.

## Results

### Cross-validation on the training set (5 folds, mean)

| Model | Accuracy | Recall (churn) | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.752 | 0.802 | 0.632 | 0.846 | 0.659 |
| Random Forest | 0.769 | 0.765 | 0.637 | 0.847 | 0.663 |
| SVM | 0.748 | 0.785 | 0.623 | 0.826 | 0.592 |
| XGBoost | 0.758 | 0.791 | 0.634 | 0.848 | 0.668 |

Standard deviations across folds are around 0.005 for ROC-AUC, so logistic regression, random forest and XGBoost are statistically tied; only the SVM is clearly behind (PR-AUC). Accuracy is not used to choose between models: it depends on the decision threshold, which `class_weight` shifts.

### Final model: Logistic Regression

Logistic regression was chosen because it matches the best models on ROC-AUC and PR-AUC, has the highest recall, and is the simplest and most interpretable option. XGBoost did not improve on it.

| Metric (test set) | Value |
|---|---|
| Accuracy | TODO |
| Recall (churn) | TODO |
| ROC-AUC | TODO |

<!-- Fill in with the output of `python main.py` -->

## Why recall

The goal of a churn model is to flag customers who are about to leave so the company can act (a call, a retention offer). Missing a customer who leaves (false negative) loses their future revenue; flagging one who would have stayed (false positive) costs an unnecessary offer. Under the assumption that losing a customer costs more than a retention action, recall on the churn class is prioritised. The dataset contains no cost information, so this is an assumption, not a measured fact. The trade-off can be tuned later by moving the decision threshold.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Download the dataset from Kaggle and save it as `data/Telco-Customer-Churn.csv`.
2. Run the full workflow:

```bash
python main.py
```

This prints the cross-validation table, draws the out-of-fold ROC curves, evaluates the final model on the test set and saves it to `models/best_churn_model.pkl`.

To use the saved model:

```python
import joblib

model = joblib.load("models/best_churn_model.pkl")
model.predict_proba(new_customers)[:, 1]   # churn probability
```

`new_customers` is a DataFrame with the original columns (without `customerID` and `Churn`); the pipeline applies the preprocessing itself.

## Possible next steps

- Choose the decision threshold from cross-validated predictions and a cost assumption.
- Hyperparameter tuning with `RandomizedSearchCV` on the training set.
<<<<<<< HEAD
- Model explainability (coefficients of the logistic regression, SHAP for tree models).
=======
- Model explainability (coefficients of the logistic regression, SHAP for tree models).
>>>>>>> 7c7fee7 (XGBoost model added, new pipeline of modeling, improving notebooks, now we use cross validation to evaluate models. Modified README)
