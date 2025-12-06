# Customer Churn Prediction

## Project Overview
The goal of this project is to predict customer churn for a telecom company using Machine Learning. This allows identifying at-risk clients and applying targeted retention strategies.

---

## Dataset
- Source: Telco Customer Churn dataset  
- Features: customer demographics, account information, and services subscribed  
- Target variable: `Churn` (Yes/No)

---

## Project Structure

churn_project/
├── data/ # Original and cleaned dataset for notebook experiments
├── src/ # Source code: data loading, preprocessing, modeling, evaluation
├── notebooks/ # Exploratory Data Analysis (EDA) and modeling
├── models/ # Trained models (optional)
├── main.py # Main script to run the pipeline
├── README.md # This file


> Note: `models/` is optional and generated when training models.

---

## Data Preprocessing
- Remove `customerID` column (not informative)  
- Convert `TotalCharges` to numeric  
- Encode `Churn` as 0/1  
- One-Hot Encoding for categorical variables  
- Remove missing values  
- Separate target column from features  
- Scale numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`)  

---

## Trained Models
- **Logistic Regression**: interpretable, linear  
- **Random Forest**: robust, handles non-linear relationships, best baseline performance  
- **SVM (Support Vector Machine)**: accurate on clean, scaled data  

---

## Evaluation Metrics
- Accuracy  
- Classification Report (Precision, Recall, F1-score)  
- Confusion Matrix  
- ROC Curve (if the model supports probabilities)  

---

## Key Findings
- **Random Forest** is the most robust model.  
- Features most associated with churn:
  - Tenure (months with the company)  
  - Contract type (monthly contracts have higher churn)  
  - Services subscribed (lack of additional services increases churn)  
  - MonthlyCharges  
- Data cleaning steps (removing ID, fixing TotalCharges, encoding Churn) were critical.  

---

## Business Recommendations
- Focus retention on new customers (low tenure)  
- Encourage long-term contracts  
- Offer service bundles to increase loyalty  
- Monitor high monthly charges and provide personalized incentives  

---

## Next Steps
- Hyperparameter optimization (GridSearch or Bayesian)  
- Feature engineering (ratios, interactions)  
- Cross-validation for robust metrics  
- Deploy the model in an API or dashboard for real-time monitoring  

---

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py