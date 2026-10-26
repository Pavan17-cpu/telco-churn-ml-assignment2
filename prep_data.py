"""Clean the raw Telco churn CSV -> telco_churn_clean.csv"""
import pandas as pd

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.drop(columns=["customerID"])                       # identifier, not a feature
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # 11 blanks -> NaN (imputed in pipeline)
df.to_csv("telco_churn_clean.csv", index=False)
print(df.shape)
