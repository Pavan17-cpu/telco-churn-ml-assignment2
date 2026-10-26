"""Train 5 classifiers on one dataset, save models, test data and metrics.

Usage:
    python train.py --data your_dataset.csv --target target_column
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """Return the 6 required metrics. Macro averaging for multi-class."""
    avg = "binary" if n_classes == 2 else "macro"
    if n_classes == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), cat_cols),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--max_train_rows", type=int, default=0,
                    help="optional cap on training rows (useful to keep SVM fast)")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df = df.dropna(subset=[args.target])
    X, y_raw = df.drop(columns=[args.target]), df[args.target]

    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)
    n_classes = len(le.classes_)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE)
    if args.max_train_rows and len(X_tr) > args.max_train_rows:
        X_tr, _, y_tr, _ = train_test_split(
            X_tr, y_tr, train_size=args.max_train_rows,
            stratify=y_tr, random_state=RANDOM_STATE)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }

    os.makedirs("model", exist_ok=True)
    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        m = compute_metrics(y_te, pipe.predict(X_te), pipe.predict_proba(X_te), n_classes)
        rows.append({"ML Model Name": name, **m})
        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipe, os.path.join("model", fname))
        print(f"{name:25s}", {k: round(v, 4) for k, v in m.items()})

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv("model/metrics.csv", index=False)
    joblib.dump({"label_encoder": le, "target": args.target,
                 "features": X.columns.tolist()}, "model/meta.joblib")

    test_df = X_te.copy()
    test_df[args.target] = le.inverse_transform(y_te)
    test_df.to_csv("test_data.csv", index=False)
    print("\n", metrics.to_markdown(index=False))
    print(f"\nSaved models, model/metrics.csv and test_data.csv ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
