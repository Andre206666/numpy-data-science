# Diabetes Prediction Model

## Problem
Predicting whether a patient has diabetes based on health
metrics like glucose levels, BMI, and age.

## Approach
- Compared 5 models: Logistic Regression, Decision Tree,
  Random Forest, GridSearchCV-tuned RF, XGBoost
- Applied feature engineering (BMI categories, age groups)
- Handled class imbalance with balanced weights
- Evaluated using precision-recall tradeoff analysis

## Results
- Best model: Logistic Regression (77.3% accuracy)
- Key insight: Simpler models sometimes outperform complex
  ones when relationships in data are fairly linear

## Tools
Python, Pandas, Scikit-learn, XGBoost, Matplotlib