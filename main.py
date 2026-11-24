"""
Model definitions and persistence.

Each build_* function returns an UNFITTED Pipeline (preprocessor + classifier),
so it can be used both for cross-validation and for the final fit.
"""

from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data_loader import load_data, basic_info
from src.preprocessing import clean_data, split_features, build_preprocessor
from src.modeling import get_candidates, save_model
from src.evaluation import evaluate_model, plot_roc

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "best_churn_model.pkl"

TARGET = "Churn"
RANDOM_STATE = 42
FINAL_MODEL = "Logistic Regression"

def main():
    # Load data
    df = load_data(DATA_PATH)
    print("\n--- Dataset Info ---")
    basic_info(df)

    # Clean (row-wise cleaning, no learning)
    df = clean_data(df)

    # Split X, y
    X, y = split_features(df, TARGET)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
        stratify=y  # maintain class distribution
    )

    # Models are Pipelines: scaling and encoding are fitted on train only
    preprocessor = build_preprocessor(X)

    # Get candidate models
    candidates = get_candidates(preprocessor, y_train)
    
    # Train and evaluate each model
    for name, model in candidates.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        print(f"\n\n--- {name} Evaluation ---")
        evaluate_model(model, X_test, y_test)
        plot_roc(model, X_test, y_test, name)

    # Save the final model
    final_model = candidates[FINAL_MODEL]
    save_model(final_model, MODEL_PATH)
    print(f"\nFinal model ({FINAL_MODEL}) saved to {MODEL_PATH}")

    print("\nDone.")

if __name__ == "__main__":
    main()

