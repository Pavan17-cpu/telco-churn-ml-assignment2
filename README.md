# Telco Customer Churn – Classification Models with Streamlit

## a. Problem Statement
A telecom company loses revenue when customers cancel their subscription (churn). The goal is to predict whether a customer will churn (`Yes`) or stay (`No`) from their account, service and billing details. This is a binary classification problem. Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) are trained on the same data, compared on six metrics, and exposed through an interactive Streamlit app.

## b. Dataset Description
- **Source:** Kaggle – *Telco Customer Churn* (IBM sample data), file `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Task:** Binary classification; target column `Churn` (Yes = positive class)
- **Instances:** 7,043 | **Features:** 19 (after dropping the `customerID` identifier)
- **Feature types:** 16 categorical (gender, Partner, Dependents, PhoneService, InternetService, OnlineSecurity, TechSupport, Contract, PaymentMethod, etc.) and 3 numeric (`tenure`, `MonthlyCharges`, `TotalCharges`) plus the binary `SeniorCitizen` flag
- **Class distribution:** No 73.5% / Yes 26.5% (imbalanced)
- **Preprocessing:** `customerID` dropped; `TotalCharges` converted to numeric (11 blank values for tenure-0 customers imputed with the median); numeric features standard-scaled; categorical features one-hot encoded; stratified 80/20 train/test split (`random_state=42`). All preprocessing is saved inside each model pipeline, so the app accepts raw test CSVs.

## c. GitHub Repository
https://github.com/Pavan17-cpu/telco-churn-ml-assignment2

Live app: https://telco-churn-mll.streamlit.app/

## d. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8055 | 0.8419 | 0.6572 | 0.5588 | 0.6040 | 0.4790 |
| Decision Tree | 0.7750 | 0.7978 | 0.5882 | 0.5080 | 0.5452 | 0.3987 |
| KNN | 0.7637 | 0.7895 | 0.5527 | 0.5749 | 0.5636 | 0.4018 |
| Naive Bayes | 0.6948 | 0.8074 | 0.4589 | 0.8369 | 0.5928 | 0.4245 |
| Support Vector Machine | 0.7913 | 0.7905 | 0.6418 | 0.4840 | 0.5518 | 0.4264 |

Precision, Recall and F1 are reported for the positive class (`Churn = Yes`). Evaluated on the 1,409-row held-out test split.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall: highest accuracy (0.806), AUC (0.842) and MCC (0.479). Churn depends largely on linear, additive effects (contract type, tenure, charges), which suits a linear model; it is also stable and well calibrated. |
| Decision Tree | Weakest on MCC (0.399) and lower AUC (0.798). A single depth-limited tree has high variance and cannot match the smooth probability ranking of the linear model; it gains interpretability, not accuracy. |
| KNN | Moderate (accuracy 0.764, MCC 0.402). Recall (0.575) is slightly higher than most models, but one-hot encoding of many categorical features makes distances noisy in high dimensions, and AUC is the second lowest. |
| Naive Bayes | Lowest accuracy (0.695) and precision (0.459) but by far the highest recall (0.837). The independence assumption is violated (many correlated service features), which inflates churn probabilities, so it flags most churners but with many false alarms. Useful when missing a churner is costlier than a false alert. |
| Support Vector Machine | Accuracy (0.791) close to Logistic Regression, but the lowest recall (0.484): with imbalanced classes the RBF-SVM favours the majority class. Its AUC (0.791) and MCC (0.426) trail Logistic Regression. |

**Overall winner for this dataset: Logistic Regression**: best accuracy, AUC and MCC, and the best F1 (0.604), with the most balanced precision/recall trade-off. Naive Bayes is the pick only if recall on churners matters more than precision.

## Project Structure
```
app.py                  # Streamlit app (loads saved models, no retraining)
train.py                # trains the 5 models, saves them + metrics + test_data.csv
prep_data.py            # cleans the raw Kaggle CSV -> telco_churn_clean.csv
requirements.txt
test_data.csv           # held-out test split (features + Churn) for the app
model/                  # *.joblib pipelines, meta.joblib, metrics.csv
```

## How to Run
```
pip install -r requirements.txt
python prep_data.py
python train.py --data telco_churn_clean.csv --target Churn
streamlit run app.py
```
