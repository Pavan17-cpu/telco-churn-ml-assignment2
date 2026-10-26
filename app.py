import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from train import compute_metrics

st.set_page_config(page_title="Classification Model Demo", layout="wide")
st.title("Classification Models Demo")
st.caption("ML Assignment 2 – upload test data, pick a model, see the evaluation.")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join("model", MODEL_FILES[name]))


@st.cache_resource
def load_meta():
    return joblib.load(os.path.join("model", "meta.joblib"))


meta = load_meta()
le, target, features = meta["label_encoder"], meta["target"], meta["features"]

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV (features + target column)", type="csv")
    model_name = st.selectbox("Select model", list(MODEL_FILES))
    st.markdown(f"**Target column:** `{target}`")
    if os.path.exists("test_data.csv"):
        with open("test_data.csv", "rb") as f:
            st.download_button("Download sample test_data.csv", f, "test_data.csv", "text/csv")

if os.path.exists("model/metrics.csv"):
    with st.expander("Comparison of all models on the held-out test split", expanded=False):
        st.dataframe(pd.read_csv("model/metrics.csv"), width="stretch")

if uploaded is None:
    st.info("Upload a test CSV from the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in features + [target] if c not in df.columns]
if missing:
    st.error(f"Uploaded CSV is missing required columns: {missing}")
    st.stop()

st.subheader("Data preview")
st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head(), width="stretch")

known = df[target].isin(le.classes_)
if not known.all():
    st.warning(f"Dropping {(~known).sum()} rows with unseen target labels.")
    df = df[known]

X, y_true = df[features], le.transform(df[target])
model = load_model(model_name)          # pre-trained: no retraining here
y_pred, y_proba = model.predict(X), model.predict_proba(X)

st.subheader(f"Evaluation metrics – {model_name}")
try:
    m = compute_metrics(y_true, y_pred, y_proba, len(le.classes_))
except ValueError as e:
    st.error(f"Could not compute metrics (does the file contain every class?): {e}")
    st.stop()
cols = st.columns(len(m))
for col, (k, v) in zip(cols, m.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    rep = classification_report(y_true, y_pred, target_names=[str(c) for c in le.classes_],
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")
