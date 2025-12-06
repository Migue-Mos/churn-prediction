
from src.data_loader import load_data, basic_info
from src.preprocessing import clean_data, split_features, scale_numeric
from src.modeling import train_logistic, train_svm, train_random_forest
from src.evaluation import evaluate_model, plot_roc
from sklearn.model_selection import train_test_split

# Load data
df = load_data("data/Telco-Customer-Churn.csv")
print("\n--- Dataset Info ---")
basic_info(df)

# Clean
df = clean_data(df)

# Define target
target = "Churn"

# Split X, y
X, y = split_features(df, target)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Numeric columns to scale
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

# Scale
X_train, X_test, scaler = scale_numeric(X_train, X_test, numeric_cols)

# --- Train models ---
print("\nTraining Logistic Regression...")
log_model = train_logistic(X_train, y_train)

print("\nTraining Random Forest...")
rf_model = train_random_forest(X_train, y_train)

print("\nTraining SVM...")
svm_model = train_svm(X_train, y_train)

# --- Evaluate ---
print("\n\n--- Logistic Regression Evaluation ---")
evaluate_model(log_model, X_test, y_test)
plot_roc(log_model, X_test, y_test, "Logistic Regression")

print("\n\n--- Random Forest Evaluation ---")
evaluate_model(rf_model, X_test, y_test)
plot_roc(rf_model, X_test, y_test, "Random Forest")

print("\n\n--- SVM Evaluation ---")
evaluate_model(svm_model, X_test, y_test)
plot_roc(svm_model, X_test, y_test, "SVM")

print("\nDone.")

